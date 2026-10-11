"""Where does a warm (full APC hit) TTFT go?  One server, one long prompt, cold then warm xN.

Starts the served process through tfbench's Srv with a stack sampler injected (sitecustomize),
sends the prompt, and prints for each warm request the client TTFT plus a timeline of what the
server's threads were doing between request send and first token (runs of one stack >= 5 ms).
Follow-up runs once per fresh server: repeating it would cache the newly prefilled reply
and misrepresent continued-chat cost. Use fresh interleaved servers for comparisons.
Usage (via gpuq): ttft_warm_probe.py --model M --kind code --ctx 32768 --warm 3
    --tag base --src TREE/python --out result.jsonl --calls
"""

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def timeline(samples, t_lo, t_hi, min_ms=5.0):
    """Collapse samples of each thread in [t_lo, t_hi] to runs of an identical top-4 stack.
    Returns {thread: [(start_offset_ms, dur_ms, stack_str)]} for runs >= min_ms."""
    by_thread = {}
    for t, name, stack in samples:
        if t_lo <= t <= t_hi:
            by_thread.setdefault(name, []).append((t, " < ".join(stack[:4])))
    out = {}
    for name, rows in by_thread.items():
        rows.sort()
        runs = []
        cur = None
        for t, st in rows:
            if cur and cur[2] == st:
                cur[1] = t
            else:
                if cur:
                    runs.append(cur)
                cur = [t, t, st]
        if cur:
            runs.append(cur)
        out[name] = [
            (round((a - t_lo) * 1e3, 1), round((b - a) * 1e3 + 2, 1), st)
            for a, b, st in runs
            if (b - a) * 1e3 + 2 >= min_ms
        ]
    return out


def stage(stack):
    """Classify the active frame, giving inner work precedence over its callers.

    These are sampled wall-time estimates, not GPU execution timings. An eval
    beneath prefill includes outstanding GPU work and scheduling waits.
    """
    for frame in stack:
        lower = frame.lower()
        if any(
            x in lower
            for x in ("clone", "lookup_exact", "fetch_nearest", "_promote_warm")
        ):
            return "apc_lookup_clone"
        if any(x in lower for x in ("sample", "sampler")):
            return "sampling"
        if any(
            x in lower for x in ("tokenize", "encode:", "_format_prompt", "template")
        ):
            return "template_tokenize"
        if any(x in lower for x in ("prefill", "forward.py")):
            return "new_token_prefill"
        if any(
            x in lower for x in ("round_driver", "mtp_lane", "dflash", "tree_verify")
        ):
            return "first_round"
    return "other"


def stage_summary(samples, t_lo, t_hi, thread="yunshu-mlx"):
    """Integrate sample intervals on ONE executor thread; never sum threads.

    Unobserved time is retained, so the sum equals the client interval. Thread
    names are recorded separately for diagnosis of different executor naming.
    """
    rows = sorted((t, stack) for t, name, stack in samples if thread in name)
    totals = {
        key: 0.0
        for key in (
            "template_tokenize",
            "apc_lookup_clone",
            "new_token_prefill",
            "first_round",
            "sampling",
            "other",
            "unobserved",
        )
    }
    cursor = t_lo
    for i, (t, stack) in enumerate(rows):
        end = rows[i + 1][0] if i + 1 < len(rows) else t + 0.002
        start, end = max(t_lo, t), min(t_hi, end, t + 0.010)
        if end <= start:
            continue
        totals["unobserved"] += max(0.0, start - cursor)
        totals[stage(stack)] += end - start
        cursor = max(cursor, end)
    totals["unobserved"] += max(0.0, t_hi - cursor)
    return {key: round(value * 1000, 3) for key, value in totals.items()}


def partition_spans(spans, duration_ms, thread):
    """Partition the client interval by the innermost measured Python scope.

    The residual includes HTTP/SSE, executor queue, detokenization and scopes
    not instrumented. It must not be advertised as pure network overhead.
    GPU waits remain nested in their Python scope and are also stored raw.
    """
    scopes = [r for r in spans if thread in r["thread"] and r["stage"] != "gpu_wait"]
    edges = sorted(
        {
            0.0,
            duration_ms,
            *(
                max(0.0, min(duration_ms, r[k]))
                for r in scopes
                for k in ("start_ms", "end_ms")
            ),
        }
    )
    totals = {}
    for start, end in zip(edges, edges[1:], strict=False):
        active = [r for r in scopes if r["start_ms"] <= start and r["end_ms"] >= end]
        label = (
            min(active, key=lambda r: r["end_ms"] - r["start_ms"])["stage"]
            if active
            else "http_sse_queue_other"
        )
        totals[label] = totals.get(label, 0.0) + end - start
    return {key: round(value, 3) for key, value in totals.items()}


def clip_spans(spans, start, end, trace_id):
    """Clip spans to one delivery interval, retaining request-correlated HTTP marks."""
    return [
        dict(
            start_ms=(max(t, start) - start) * 1000,
            end_ms=(min(stop, end) - start) * 1000,
            thread=name,
            stage=label,
            detail=detail,
        )
        for t, stop, name, label, detail in spans
        if t < end
        and stop > start
        and (
            label not in ("http_receive", "http_first_sse")
            or detail.endswith("request=" + trace_id)
        )
    ]


def validate_response(result, phase, *, require_followup_hit=True):
    if result["ct"] < 1 or result["ttft_s"] <= 0:
        raise ValueError("missing generated token or TTFT")
    if phase == "cold" and result["cached"] != 0:
        raise ValueError("first request is not cold")
    if phase == "warm" and result["cached"] < result["pt"] - 8:
        raise ValueError("warm request is not a full hit")
    if phase == "followup" and require_followup_hit and not result["cached"]:
        raise ValueError("followup request did not hit APC")


def signal_dump(pid):
    """SIGUSR1 every process in pid's group (the served worker may be a child of the launcher)."""
    out = subprocess.run(
        ["pgrep", "-g", str(os.getpgid(pid))], capture_output=True, text=True
    ).stdout
    for p in out.split():
        with contextlib.suppress(ProcessLookupError):
            os.kill(int(p), signal.SIGUSR1)


def load_samples(prefix):
    rows = []
    for p in Path(prefix).parent.glob(Path(prefix).name + ".*"):
        if ".spans." not in p.name:
            rows += [json.loads(line) for line in p.read_text().splitlines() if line]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", default="code")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--src", required=True, help="python/ dir of the tree under test")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument(
        "--followup-tokens",
        type=int,
        default=2048,
        help="cold reply length used in follow-up, matching yv long",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--allow-followup-miss",
        action="store_true",
        help="tiny pilot only: cache policy may not retain a reusable historical-turn boundary",
    )
    ap.add_argument("--draft", default="dflash", choices=("dflash", "mtp", "off"))
    ap.add_argument(
        "--calls",
        action="store_true",
        help="diagnostic Python/C spans (adds profiling overhead)",
    )
    ap.add_argument(
        "--no-stacks",
        action="store_true",
        help="call/GC spans only, to test stack-sampler observer effects",
    )
    ap.add_argument(
        "--copy-index-gc",
        choices=("auto", "defer"),
        default=None,
        help="diagnostic only: defer automatic GC during copy-index build",
    )
    ap.add_argument("--thread", default="mlx", help="executor thread name substring")
    ap.add_argument(
        "--synthetic-prompt",
        action="store_true",
        help="self-contained tiny-model pilot (not a timing corpus)",
    )
    ap.add_argument("--env", action="append", default=[], help="server environment K=V")
    ap.add_argument(
        "--rehearse",
        action="store_true",
        help="before the follow-up, prefill the rendered assistant reply (next-turn rehearsal)",
    )
    a = ap.parse_args()
    if a.no_stacks and not a.calls:
        ap.error("--no-stacks requires --calls")
    server_env = {}
    for assignment in a.env:
        key, sep, value = assignment.partition("=")
        if not key or not sep:
            ap.error("--env requires K=V")
        server_env[key] = value
    gc_policy = a.copy_index_gc or server_env.get("TTFT_PROBE_COPY_INDEX_GC", "auto")
    if gc_policy not in ("auto", "defer"):
        ap.error("TTFT_PROBE_COPY_INDEX_GC must be auto or defer")
    server_env["TTFT_PROBE_COPY_INDEX_GC"] = gc_policy
    os.environ["TFB_YUNSHU_SRC"] = a.src
    os.environ["TFB_OUT"] = str(a.out.parent / a.tag)
    os.environ["TFB_PORT_LAST"] = "18996"
    sys.path.insert(0, str(HERE))
    import tfbench

    out = Path(os.environ["TFB_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    prof = str(out / "samples")
    real_popen = subprocess.Popen

    def popen(cmd, **kw):
        if not isinstance(kw.get("env"), dict) or "serve" not in cmd:
            return real_popen(cmd, **kw)  # not the served process (pgrep etc.)
        env = dict(kw["env"])
        env["PYTHONPATH"] = f"{HERE / 'ttft_probe_site'}:{env.get('PYTHONPATH', '')}"
        env["TTFT_PROBE_OUT"] = prof
        env["TTFT_PROBE_CALLS"] = "1" if a.calls else "0"
        env["TTFT_PROBE_STACKS"] = "0" if a.no_stacks else "1"
        kw["env"] = env
        return real_popen(cmd, **kw)

    tfbench.subprocess.Popen = popen
    prompt = (
        "Explain this material in detail: local inference uses a prefix cache. " * 128
        if a.synthetic_prompt
        else tfbench.load_prompt(f"{a.kind}-{a.ctx}")
    )
    extra_env = {"YUNSHU_VLM_DRAFT": tfbench.D if a.draft == "dflash" else a.draft}
    extra_env.update(server_env)
    s = tfbench.Srv("yunshu", extra_env, f"-probe-{a.tag}", model=a.model)
    try:
        rows = []
        reply = ""
        phases = ["cold"] + ["warm"] * a.warm + ["followup"]
        for i, phase in enumerate(phases):
            body = tfbench.req(
                s.model, prompt, a.followup_tokens if phase == "cold" else a.max_tokens
            )
            if phase == "followup":
                body["messages"] += [
                    {"role": "assistant", "content": reply},
                    {
                        "role": "user",
                        "content": "Continue with the next part, at the same length.",
                    },
                ]
            rehearsal = {}
            if phase == "followup" and a.rehearse:
                rehearsal = tfbench.rehearse(s, body)
                print(f"rehearsal: {rehearsal}", flush=True)
            t0 = time.perf_counter()
            trace_id = f"ttft-{os.getpid()}-{i}"
            r = tfbench.send(s.url, body, capture_timestamps=True, trace_id=trace_id)
            finished_at = time.perf_counter()
            sent_at = r.pop("_sent_at_s")
            first_at = r.pop("_first_at_s")
            if first_at is None:
                raise ValueError("stream had no visible first token")
            if phase == "cold":
                reply = r["_text"]
            r.pop("_text", None)
            validate_response(r, phase, require_followup_hit=not a.allow_followup_miss)
            rows.append(
                dict(
                    rep=i,
                    phase=phase,
                    start=sent_at,
                    first=first_at,
                    finished=finished_at,
                    client_prepare_ms=(sent_at - t0) * 1000,
                    trace_id=trace_id,
                    **rehearsal,
                    **r,
                )
            )
            print(
                f"{phase} {i}: TTFT {r['ttft_s']:.3f}s cached {r['cached']}/{r['pt']}",
                flush=True,
            )
        signal_dump(s.proc.pid)
        time.sleep(3)
        samples = load_samples(prof)
        spans = [
            json.loads(line)
            for p in out.glob("samples.spans.*")
            for line in p.read_text().splitlines()
            if line
        ]
        runtime = [json.loads(p.read_text()) for p in out.glob("runtime.*.json")]
        if not samples and not spans:
            print("FAIL: no probe evidence")
            sys.exit(1)
        mode = tfbench.engaged_spec_mode("yunshu", s.log.read_text())
        if mode is None:
            raise RuntimeError("missing engaged speculative mode")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("w") as f:
            for row in rows:
                row["sampled_stage_ms"] = stage_summary(
                    samples, row["start"], row["first"], a.thread
                )
                row["call_spans"] = clip_spans(
                    spans, row["start"], row["first"], row["trace_id"]
                )
                # Retain GC/index work after first delivery: suppression may only relocate it.
                full_spans = clip_spans(
                    spans, row["start"], row["finished"], row["trace_id"]
                )
                first_ms = (row["first"] - row["start"]) * 1000
                row["delivery_tail_spans"] = [
                    dict(r, start_ms=max(first_ms, r["start_ms"]))
                    for r in full_spans
                    if r["stage"] in ("python_gc", "copy_index")
                    and r["end_ms"] > first_ms
                ]
                row["exclusive_call_stage_ms"] = partition_spans(
                    row["call_spans"], (row["first"] - row["start"]) * 1000, a.thread
                )
                http_receive = [
                    r["start_ms"]
                    for r in row["call_spans"]
                    if r["stage"] == "http_receive"
                ]
                http_sse = [
                    r["start_ms"]
                    for r in row["call_spans"]
                    if r["stage"] == "http_first_sse"
                ]
                row["http"] = dict(
                    ingress_ms=min(http_receive) if http_receive else None,
                    first_sse_submit_ms=min(http_sse) if http_sse else None,
                    sse_submit_to_client_ms=(row["first"] - row["start"]) * 1000
                    - min(http_sse)
                    if http_sse
                    else None,
                )
                row["timeline"] = timeline(samples, row["start"], row["first"])
                row.update(
                    uncached_tokens=row["pt"] - row["cached"],
                    ctx=a.ctx,
                    kind=a.kind,
                    device=os.environ.get("GPUQ_DEVICE", "m5"),
                    engaged=mode,
                    timing_method="Python/C call spans + 2ms stack sampling; not GPU attribution",
                    diagnostic_only=True,
                    followup_apc_required=not a.allow_followup_miss,
                    stack_sampling=not a.no_stacks,
                    runtime=runtime,
                    copy_index_gc=gc_policy,
                    copy_index_gc_engaged=bool(runtime)
                    and all(r.get("copy_index_gc") == gc_policy for r in runtime),
                )
                f.write(json.dumps(row) + "\n")
            f.write(
                json.dumps(
                    {
                        "complete": True,
                        "requests": len(rows),
                        "threads": sorted({name for _, name, _ in samples}),
                    }
                )
                + "\n"
            )
    finally:
        s.kill()


if __name__ == "__main__":
    main()
