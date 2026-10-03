"""27B HTTP parity/rate receipt for constrained speculation and target logprobs.

Run only through gpuq. Explicit drafter paths survive the isolated server HOME.
Each output ends with complete only after successful requests and log engagement.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import tfbench


def workloads():
    schema = {
        "type": "object",
        "properties": {
            "records": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "label": {"type": "string"},
                        "status": {"enum": ["ready", "pending"]},
                    },
                    "required": ["id", "label", "status"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["records"],
        "additionalProperties": False,
    }
    tools = [
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    for sampled in (False, True):
        sampling = {
            "temperature": 0.7 if sampled else 0,
            "top_p": 0.95,
            "top_k": 20,
            "seed": 1234,
        }
        yield (
            "json-" + str(sampled),
            "Return six inventory records with id, label, and status. Labels are common fruit names.",
            dict(
                sampling,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "inventory", "schema": schema},
                },
            ),
        )
        yield (
            "tool-" + str(sampled),
            "Write src/squares.py containing a Python function squares(n) that returns the first n squares. Use write_file now.",
            dict(sampling, tools=tools, tool_choice="required"),
        )
        for k in (0, 5, 20):
            yield (
                f"lp{k}-{sampled}",
                "Write a Python function implementing binary search with a short docstring. Only output Python code.",
                dict(sampling, logprobs=True, top_logprobs=k),
            )
    yield (
        "cfg-forced",
        "Output the exact JSON specified by the grammar.",
        {
            "grammar": {
                "type": "cfg",
                "grammar": 'start: "{\\"action\\":\\"write_file\\",\\"path\\":\\"src/squares.py\\",\\"content\\":\\"def squares(n): return [i*i for i in range(n)]\\"}"',
            }
        },
    )
    yield (
        "tool-lp",
        "Write src/squares.py containing a Python function squares(n) that returns the first n squares. Use write_file now.",
        {
            "tools": tools,
            "tool_choice": "required",
            "logprobs": True,
            "top_logprobs": 20,
        },
    )
    yield (
        "json-lp",
        "Return six fruit inventory records with id, label, status.",
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "inventory", "schema": schema},
            },
            "logprobs": True,
            "top_logprobs": 20,
        },
    )


def read_trace(path, count, timeout=30.0):
    """Return the first ``count`` JSON lines of ``path``, waiting for the writer.

    The token receipt is written when the generator is closed, which can be
    after the HTTP stream has already ended; reading immediately raced it.
    """
    deadline = time.monotonic() + timeout
    while True:
        data = path.read_text() if path.exists() else ""
        lines = data.splitlines()
        if data and not data.endswith("\n"):
            lines = lines[:-1]
        if len(lines) >= count:
            return [json.loads(line) for line in lines[:count]]
        if time.monotonic() > deadline:
            raise RuntimeError(f"trace {path} has {len(lines)} < {count} records")
        time.sleep(0.1)


class TraceCursor:
    """Consume one receipt per request, across every workload and phase."""

    def __init__(self, path):
        self.path = path
        self.count = 0

    def next(self):
        self.count += 1
        return read_trace(self.path, self.count)[-1]


def run_agent(srv, out, args):
    """Fix auxiliary title sampling for reproducible background-request parity."""
    original = tfbench.send

    def deterministic_title(url, body, *a, **kw):
        if not body.get("tools"):
            body = dict(body, temperature=0)
        return original(url, body, *a, **kw)

    tfbench.send = deterministic_title
    try:
        return tfbench.part_agent(srv, out, args)
    finally:
        tfbench.send = original


def require_cache_reuse(result, phase):
    """A parity test must actually exercise APC on its warm/follow-up arm."""
    if phase in ("warm", "turn2") and not (result.get("xy") or {}).get("cached_tokens"):
        raise RuntimeError(f"{phase} did not reuse the cache")


def second_turn(prompt, result, extra):
    """Messages for a follow-up turn that extends the first (prefix reuse)."""
    text, _reasoning, tool_calls, _finish = result
    messages = [{"role": "user", "content": prompt}]
    if tool_calls:
        calls = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": t["name"], "arguments": t["arguments"]},
            }
            for i, t in enumerate(tool_calls)
        ]
        messages.append({"role": "assistant", "content": None, "tool_calls": calls})
        for call in calls:
            messages.append(
                {"role": "tool", "tool_call_id": call["id"], "content": "ok"}
            )
        messages.append({"role": "user", "content": "Now write tests/test_squares.py."})
    else:
        messages.append({"role": "assistant", "content": text})
        messages.append(
            {"role": "user", "content": "Do it again with different values."}
        )
    return messages


def send(srv, prompt, extra, maximum):
    body = dict(
        model=srv.model,
        messages=prompt
        if isinstance(prompt, list)
        else [{"role": "user", "content": prompt}],
        max_tokens=maximum,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        chat_template_kwargs={"enable_thinking": False},
    )
    body.update(extra)
    req = urllib.request.Request(
        srv.url + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer k"},
    )
    start = time.perf_counter()
    first = None
    content, reasoning, tools, lps = [], [], {}, []
    finish = usage = xy = None
    done = False
    with urllib.request.urlopen(req, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            p = line[5:].strip()
            if p == b"[DONE]":
                done = True
                break
            chunk = json.loads(p)
            if chunk.get("error"):
                raise RuntimeError(str(chunk["error"]))
            usage = chunk.get("usage") or usage
            xy = chunk.get("x_yunshu") or xy
            for ch in chunk.get("choices", []):
                delta = ch.get("delta", {})
                if (
                    any(
                        delta.get(k)
                        for k in ("content", "reasoning_content", "tool_calls")
                    )
                    and first is None
                ):
                    first = time.perf_counter()
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or "")
                for t in delta.get("tool_calls") or []:
                    fn = t.get("function") or {}
                    old = tools.setdefault(
                        t.get("index", 0), {"name": "", "arguments": ""}
                    )
                    old["name"] += fn.get("name") or ""
                    old["arguments"] += fn.get("arguments") or ""
                lps.extend((ch.get("logprobs") or {}).get("content") or [])
                finish = ch.get("finish_reason") or finish
    wall = time.perf_counter() - start
    if not done or usage is None or finish is None:
        raise RuntimeError("incomplete stream")
    text = "".join(content)
    value = [text, "".join(reasoning), list(tools.values()), finish]
    if finish != "length":
        for t in tools.values():
            json.loads(t["arguments"])
    if "response_format" in extra and finish != "length":
        json.loads(text)

    def sha(value):
        return hashlib.sha256(
            json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()

    return dict(
        ct=usage["completion_tokens"],
        ttft_s=(first or start + wall) - start,
        wall_s=wall,
        xy=xy,
        digest=sha(value),
        token_digest=sha([(lp["token"], lp.get("bytes")) for lp in lps])
        if lps
        else None,
        lp_digest=sha(lps) if lps else None,
        logprobs=lps,
        result=value,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rep", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--warm", action="store_true")
    ap.add_argument(
        "--turn2",
        action="store_true",
        help="after the first request send a second turn that reuses its cache",
    )
    ap.add_argument("--agent", action="store_true")
    ap.add_argument(
        "--tool-grammar-off",
        action="store_true",
        help="reproduce the original injected 185-token tool prompt (unconstrained tools)",
    )
    ap.add_argument(
        "--modes", nargs="+", choices=("mtp-ar", "mtp", "dflash-ar", "dflash")
    )
    ap.add_argument("--cache-state-check", action="store_true")
    ap.add_argument("--only-case", action="append")
    a = ap.parse_args()
    tfbench.YUNSHU_SRC = str(Path(__file__).resolve().parents[2] / "python")
    tfbench.OUT = Path(a.out).parent / (Path(a.out).stem + "-servers")
    tfbench.OUT.mkdir(parents=True, exist_ok=True)
    frozen = tfbench.OUT / "source"
    frozen.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        Path(__file__).resolve().parents[2] / "python",
        frozen / "python",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(
        Path(__file__).parent / "cspec_capture",
        frozen / "capture",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    errors = []
    receipts = {}
    with open(a.out, "w") as out:
        root = Path(__file__).resolve().parents[2]
        sources = [
            frozen / "python/yunshu_engine" / name
            for name in (
                "constrained_spec.py",
                "mtp_lane.py",
                "tool_call_grammar.py",
                "vlm_batch_runner.py",
                "vlm_engine.py",
            )
        ]
        tfbench.emit(
            out,
            part="meta",
            source_hash=hashlib.sha256(
                b"".join(p.read_bytes() for p in sources)
            ).hexdigest(),
            harness_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            git_head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            versions={
                name: importlib.metadata.version(name)
                for name in ("mlx", "mlx-vlm", "llguidance")
            },
            checkpoint=tfbench.M,
            drafter=tfbench.D,
            max_tokens=a.max_tokens,
            sampler="position-keyed seed=1234",
            tool_grammar=not a.tool_grammar_off,
            job=os.environ.get("GPUQ_JOB_ID"),
        )
        for rep in range(a.rep):
            modes = a.modes or ["mtp-ar", "mtp", "dflash-ar", "dflash"]
            for mode in modes if rep % 2 == 0 else modes[::-1]:
                srv = None
                try:
                    drafter = mode.removesuffix("-ar")
                    env = {
                        "YUNSHU_VLM_DRAFT": tfbench.D
                        if drafter == "dflash"
                        else drafter,
                        "CSPEC_DISABLE_DRAFT": "1" if mode.endswith("-ar") else "0",
                        "YUNSHU_AUTH_DISABLED": "1",
                        "YUNSHU_TOOL_GRAMMAR": "0" if a.tool_grammar_off else "1",
                    }
                    trace = tfbench.OUT / f"tokens-{mode}-r{rep}.jsonl"
                    if trace.exists():
                        trace.unlink()
                    env["CSPEC_TOKEN_TRACE"] = str(trace)
                    env["CSPEC_CACHE_STATE_CHECK"] = "1" if a.cache_state_check else "0"
                    tfbench.YUNSHU_SRC = os.pathsep.join(
                        [
                            str(frozen / "capture"),
                            str(frozen / "python"),
                        ]
                    )
                    if a.agent:
                        env["CSPEC_DISABLE_PLAIN"] = "1"
                    srv = tfbench.Srv("yunshu", env, f"{mode}-r{rep}")
                    token_cursor = TraceCursor(trace)
                    state_cursor = TraceCursor(Path(str(trace) + ".states"))
                    for case, prompt, extra in [] if a.agent else workloads():
                        if a.only_case and case not in a.only_case:
                            continue
                        if a.smoke and case not in (
                            "json-False",
                            "tool-False",
                            "lp5-False",
                            "cfg-forced",
                        ):
                            continue
                        phases = ["cold"] + (["warm"] if a.warm else [])
                        if a.turn2:
                            phases.append("turn2")
                        first_result = None
                        for phase in phases:
                            if phase == "turn2":
                                result = send(
                                    srv,
                                    second_turn(prompt, first_result["result"], extra),
                                    extra,
                                    a.max_tokens,
                                )
                            else:
                                result = send(srv, prompt, extra, a.max_tokens)
                            require_cache_reuse(result, phase)
                            if extra.get("logprobs") and not result["logprobs"]:
                                raise RuntimeError(
                                    "requested logprobs receipt is empty"
                                )
                            if first_result is None:
                                first_result = result
                            raw = token_cursor.next()
                            result["raw_token_digest"] = raw["token_digest"]
                            result["raw_token_ids"] = raw["token_ids"]
                            if a.cache_state_check:
                                state = state_cursor.next()
                                result["cache_state_digest"] = state["digest"]
                                result["cache_state_details"] = state["details"]
                            receipts[(rep, mode, case + ":" + phase)] = result
                            tfbench.emit(
                                out,
                                part="request",
                                rep=rep,
                                mode=mode,
                                case=case + ":" + phase,
                                **result,
                            )
                            print(
                                mode,
                                rep,
                                case,
                                result["ct"],
                                (result["xy"] or {}).get("decode_tps"),
                                flush=True,
                            )
                    if a.agent:
                        agent_out = tfbench.OUT / f"agent-{mode}-r{rep}.jsonl"
                        with agent_out.open("w") as agent_file:
                            run_agent(srv, agent_file, a)
                            tfbench.emit(
                                agent_file, complete=True, part="agent_complete"
                            )
                        records = [
                            json.loads(line) for line in trace.read_text().splitlines()
                        ]
                        grouped = {}
                        for record in records:
                            grouped.setdefault(record["prompt_digest"], []).append(
                                record["token_digest"]
                            )
                        receipts[(rep, mode, "agent")] = {
                            "digest": sorted(
                                (key, sorted(value)) for key, value in grouped.items()
                            )
                        }
                        tfbench.emit(
                            out,
                            part="agent_receipt",
                            mode=mode,
                            rep=rep,
                            output=str(agent_out),
                            raw_tokens=str(trace),
                            requests=len(records),
                        )
                    log = srv.log.read_text()
                    expected = f"Speculative decoding: {drafter}"
                    engaged = (
                        "Speculation disabled for parity: allow_draft=False"
                        if mode.endswith("-ar")
                        else f"Exact speculative request engaged: {drafter}"
                    )
                    if expected not in log or engaged not in log:
                        raise RuntimeError("engaged path missing from server log")
                    if (
                        not mode.endswith("-ar")
                        and not a.tool_grammar_off
                        and (
                            a.agent
                            or not a.only_case
                            or any(case.startswith("tool-") for case in a.only_case)
                        )
                    ):
                        if (
                            f"Exact speculative request engaged: {drafter} constrained=True"
                            not in log
                        ):
                            raise RuntimeError(
                                "constrained tool path missing from server log"
                            )
                    tfbench.emit(
                        out,
                        part="arm_done",
                        mode=mode,
                        rep=rep,
                        server_log=str(srv.log),
                        kernel_plan=next(
                            (
                                line
                                for line in log.splitlines()
                                if "VLM batch runner:" in line
                            ),
                            None,
                        ),
                        serial_verify_engaged="Serial target-row verify engaged:"
                        in log,
                        engaged=mode,
                        forced_engaged="Forced-token verify window engaged:" in log,
                        rc=0,
                    )
                except Exception as exc:
                    errors.append(str(exc))
                    tfbench.emit(
                        out,
                        part="arm_failed",
                        mode=mode,
                        rep=rep,
                        rc=1,
                        error=repr(exc),
                    )
                finally:
                    if srv is not None:
                        srv.kill()
        for (rep, mode, case), result in receipts.items():
            if mode.endswith("-ar"):
                continue
            baseline = receipts[(rep, mode + "-ar", case)]
            fields = (
                ("digest",)
                if a.agent
                else ("digest", "raw_token_digest", "token_digest", "lp_digest")
            )
            if (
                a.cache_state_check
                and not a.agent
                and result["cache_state_digest"] != baseline["cache_state_digest"]
            ):
                # Diagnostic only: raw cross-path state bits may differ while every
                # observable (tokens, logprobs, later turns) stays identical.
                tfbench.emit(
                    out,
                    part="cache_state_diff",
                    rep=rep,
                    mode=mode,
                    case=case,
                )
            for field in fields:
                if result[field] != baseline[field]:
                    errors.append(f"parity r{rep} {mode} {case} {field}")
        if a.warm:
            for (rep, mode, case), result in receipts.items():
                if not case.endswith(":warm"):
                    continue
                baseline = receipts[(rep, mode, case.removesuffix(":warm") + ":cold")]
                for field in (
                    "digest",
                    "raw_token_digest",
                    "token_digest",
                    "lp_digest",
                ):
                    if result[field] != baseline[field]:
                        errors.append(f"APC parity r{rep} {mode} {case} {field}")
        if errors:
            tfbench.emit(out, part="failed", errors=errors)
            raise RuntimeError(errors)
        tfbench.emit(out, complete=True, reps=a.rep, parity=True)


if __name__ == "__main__":
    main()
