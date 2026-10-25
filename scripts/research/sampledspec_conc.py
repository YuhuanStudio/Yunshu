"""Agent-shaped concurrency A/B for sampled promotion (base vs candidate source), gpuq --quiet only.

Reuses conc_probe's filler/streaming idea, but keeps token arrival times and the streamed text so
the report has the running row's tok/s while a peer runs and after it ends, the newcomer TTFT,
cross-arm text/logprob identity (seeded shape) and engagement counters from the server log.
Scenarios: ``uneven`` (A long + B short start together; B ends first so A is promoted) and
``stagger`` (A starts, B joins at +join). Shapes: ``seeded`` (T=1 top_p .95 top_k 20 seed 17, logprobs)
and ``agent`` (no sampling parameters). --dry-run / the pure helpers are CPU only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import threading
import time
from pathlib import Path

SHAPES = {
    "seeded": dict(temperature=1.0, top_p=0.95, top_k=20, seed=17, logprobs=True),
    "agent": {},
}
PROMOTED = "Promoted request onto the speculative lane"
KEYED = "DFlash fast verification: keyed=True"


def arm_order(rep):
    arms = ("base", "cand", "off")
    return arms[rep % 3 :] + arms[: rep % 3]


def plans(scenario, tokens, join):
    """(name, delay, tokens, salt) per request."""
    if scenario == "uneven":
        return [("A", 0.0, tokens, 101), ("B", 0.0, max(64, tokens // 3), 102)]
    if scenario == "stagger":
        return [("A", 0.0, tokens, 111), ("B", join, max(64, tokens // 3), 112)]
    raise ValueError(scenario)


def split_rates(a, b):
    """Running row A's tok/s while B decodes, and after B ends, plus B's TTFT (all from arrival times)."""
    ta, tb = a["times"], b["times"]
    if len(tb) < 2 or len(ta) < 4:
        raise RuntimeError("too few tokens to split rates")
    lo, hi = tb[0], tb[-1]
    during = [t for t in ta if lo <= t <= hi]
    after = [t for t in ta if t > hi]

    def rate(ts):
        return (
            (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 and ts[-1] > ts[0] else None
        )

    return dict(
        a_overlap_tps=rate(during),
        a_after_tps=rate(after),
        a_after_tokens=len(after),
        a_overlap_tokens=len(during),
        b_ttft=b["ttft"],
        a_ttft=a["ttft"],
        overlapped=bool(during),
    )


def digest(text, lps):
    h = hashlib.sha256(text.encode())
    h.update(json.dumps(lps).encode())
    return h.hexdigest()


def body(model, prompt, tokens, shape):
    b = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=tokens,
        stream=True,
        stream_options={"include_usage": True},
    )
    b.update(SHAPES[shape])
    return b


def stream(url, b, t_zero):
    import httpx

    times, text, lps = [], [], []
    usage = finish = None
    t0 = time.perf_counter()
    with httpx.Client(timeout=httpx.Timeout(1800.0)) as c:
        with c.stream("POST", url + "/v1/chat/completions", json=b) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                d = json.loads(line[6:])
                usage = d.get("usage") or usage
                for ch in d.get("choices") or []:
                    delta = ch.get("delta") or {}
                    piece = (
                        delta.get("content")
                        or delta.get("reasoning_content")
                        or delta.get("reasoning")
                    )
                    if piece:
                        times.append(time.perf_counter())
                        text.append(piece)
                    for item in (ch.get("logprobs") or {}).get("content") or []:
                        lps.append([item["token"], item["logprob"]])
                    finish = ch.get("finish_reason") or finish
    if finish is None or usage is None or not times:
        raise RuntimeError(f"incomplete stream finish={finish} usage={usage}")
    return dict(
        t0=t0 - t_zero,
        times=[t - t_zero for t in times],
        ttft=times[0] - t0,
        usage=usage,
        finish=finish,
        digest=digest("".join(text), lps),
    )


def run_scenario(url, model, scenario, shape, tokens, ctx_chars, join, salt_base):
    import conc_probe

    t_zero = time.perf_counter()
    out, errs = {}, []

    def go(name, delay, ntok, salt):
        try:
            time.sleep(delay)
            prompt = (
                conc_probe.filler(ctx_chars, salt_base + salt)
                + f"\n\n[{salt_base + salt}] Review the material above carefully. Reason step by step about "
                "its design, then write a detailed technical critique with concrete improvement proposals."
            )
            out[name] = stream(url, body(model, prompt, ntok, shape), t_zero)
        except Exception as e:  # noqa: BLE001 - surfaced after join
            errs.append(repr(e))

    th = [
        threading.Thread(target=go, args=(n, d, k, s))
        for n, d, k, s in plans(scenario, tokens, join)
    ]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if errs or set(out) != {"A", "B"}:
        raise RuntimeError(f"scenario failed: {errs}")
    rates = split_rates(out["A"], out["B"])
    return dict(
        scenario=scenario,
        shape=shape,
        prompt_tokens=out["A"]["usage"].get("prompt_tokens"),
        a_tokens=out["A"]["usage"].get("completion_tokens"),
        b_tokens=out["B"]["usage"].get("completion_tokens"),
        finish=[out["A"]["finish"], out["B"]["finish"]],
        a_digest=out["A"]["digest"],
        b_digest=out["B"]["digest"],
        a_x_yunshu=out["A"]["usage"].get("x_yunshu"),
        **rates,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-src", type=Path, required=True)
    p.add_argument("--cand-src", type=Path, required=True)
    p.add_argument("--ctx-chars", type=int, required=True)
    p.add_argument("--scenarios", nargs="+", required=True)
    p.add_argument("--shapes", nargs="+", required=True)
    p.add_argument("--rep", type=int, required=True)
    p.add_argument("--tokens", type=int, default=800)
    p.add_argument("--join", type=float, default=12.0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    for s in a.scenarios:
        plans(s, a.tokens, a.join)
    for s in a.shapes:
        SHAPES[s]
    if a.dry_run:
        print(json.dumps(dict(complete=True, order=arm_order(a.rep))))
        return
    import tfbench as t

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for arm in arm_order(a.rep):
        t.YUNSHU_SRC = str((a.base_src if arm == "base" else a.cand_src).resolve())
        label = f"sampledspec-conc-{a.out.stem}-{arm}"
        apc = Path("/Volumes/Micron/yunshu-apc") / label
        server = t.Srv(
            "yunshu",
            {
                "YUNSHU_VLM_APC_DISK": "0",
                "YUNSHU_VLM_APC_MEMORY_GB": "0",
                "YUNSHU_VLM_APC_DISK_DIR": str(apc),
                "YUNSHU_VLM_DRAFT": "off" if arm == "off" else t.D,
                "YUNSHU_NEXT_TURN_PREFILL": "off",
                "YUNSHU_CONSOLE": "0",
            },
            label,
        )
        try:
            t.send(server.url, t.req(server.model, "Say hi.", 16))
            for si, scenario in enumerate(a.scenarios):
                for shape in a.shapes:
                    start = server.log.stat().st_size
                    # distinct salt per rep/arm-independent so base and cand see identical prompts
                    res = run_scenario(
                        server.url,
                        server.model,
                        scenario,
                        shape,
                        a.tokens,
                        a.ctx_chars,
                        a.join,
                        salt_base=1000 * (a.rep + 1) + 10 * si,
                    )
                    with server.log.open("rb") as fh:
                        fh.seek(start)
                        log = fh.read().decode(errors="replace")
                    res.update(
                        arm=arm,
                        rep=a.rep,
                        promoted=log.count(PROMOTED),
                        keyed_tree=log.count(KEYED),
                    )
                    if not res["overlapped"]:
                        raise RuntimeError("requests did not overlap; invalid cell")
                    if arm == "cand" and scenario == "uneven" and shape == "seeded":
                        if not res["promoted"]:
                            raise RuntimeError("candidate never promoted a row")
                    with a.out.open("a") as fh:
                        fh.write(json.dumps(res) + "\n")
                    print(
                        json.dumps({k: v for k, v in res.items() if k != "a_x_yunshu"}),
                        flush=True,
                    )
            if arm != "off":
                server.verify_spec_mode()
        finally:
            server.kill()
            shutil.rmtree(apc, ignore_errors=True)
    with a.out.open("a") as fh:
        fh.write(json.dumps(dict(complete=True)) + "\n")


if __name__ == "__main__":
    main()
