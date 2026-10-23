#!/usr/bin/env python3
"""Per-value identity replay for the phase-2 lossless knob list (greedy, tokens + logprobs).

Run mode (inside gpuq):  servedev_identity.py run --config NAME --model DIR --out FILE.json
  starts one server with the config's settings (isolated HOME, port 18990-18999, SIGKILLed on exit),
  checks the draft mode that actually engaged, sends fixed greedy prompts at ~1K and ~32K context and
  stores exact token ids + chosen-token logprobs + finish reasons. No GPU work outside the server.
Compare mode (CPU):      servedev_identity.py compare --base FILE --cand FILE [...]
  fails closed: a missing, incomplete or differently-shaped run is a mismatch, never a pass.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

DRAFTER = "/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2"
COMMON = {
    "YUNSHU_REPLAY_TOKEN_IDS": "1",
    "YUNSHU_VLM_APC_DISK": "0",  # no cross-server cache hits: every config prefills cold
    "YUNSHU_SERVE_LOG": "0",
}
# name -> (settings, draft mode that must engage)
CONFIGS: dict[str, tuple[dict[str, str], str]] = {
    "off": ({"YUNSHU_VLM_DRAFT": "off"}, "off"),
    "mtp_base": ({"YUNSHU_VLM_DRAFT": "mtp"}, "mtp"),
    "mtp_block4": ({"YUNSHU_VLM_DRAFT": "mtp", "YUNSHU_MTP_BLOCK_SIZE": "4"}, "mtp"),
    "mtp_block8": ({"YUNSHU_VLM_DRAFT": "mtp", "YUNSHU_MTP_BLOCK_SIZE": "8"}, "mtp"),
    "gc_default": ({"YUNSHU_VLM_DRAFT": "mtp", "YUNSHU_GC_POLICY": "default"}, "mtp"),
    "gc_freeze": ({"YUNSHU_VLM_DRAFT": "mtp", "YUNSHU_GC_POLICY": "freeze"}, "mtp"),
    "df_base": ({"YUNSHU_VLM_DRAFT": DRAFTER}, "dflash"),
    "df_bits0": ({"YUNSHU_VLM_DRAFT": DRAFTER, "YUNSHU_DRAFT_BITS": "0"}, "dflash"),
}
# knob -> (baseline config, candidate configs)
KNOBS = {
    "YUNSHU_MTP_BLOCK_SIZE": ("mtp_base", ["mtp_block4", "mtp_block8"]),
    "YUNSHU_GC_POLICY": ("mtp_base", ["gc_default", "gc_freeze"]),
    "YUNSHU_DRAFT_BITS": ("df_base", ["df_bits0"]),
}
CTXS = (1024, 32768)
SEEDS = (11, 12)
MAX_TOKENS = 128


def build_prompts() -> list[dict]:
    from servedev.shapes import Shape, synthetic_messages

    out = []
    for ctx in CTXS:
        for seed in SEEDS:
            out.append(
                {
                    "ctx": ctx,
                    "seed": seed,
                    "messages": synthetic_messages(
                        Shape(ctx, MAX_TOKENS), seed, tokens_per_word=3.0
                    ),
                }
            )
    return out


def compare_rows(base: dict, cand: dict) -> dict:
    """Exact comparison of one request pair. Anything missing is a mismatch."""
    need = ("ids", "logprobs", "finish")
    for r in (base, cand):
        if any(not r.get(k) for k in need) or len(r["ids"]) != len(r["logprobs"]):
            return {"ok": False, "reason": "incomplete row"}
    n = min(len(base["ids"]), len(cand["ids"]))
    first = next((i for i in range(n) if base["ids"][i] != cand["ids"][i]), None)
    if first is None and len(base["ids"]) != len(cand["ids"]):
        first = n
    tokens_equal = first is None and base["finish"] == cand["finish"]
    lp_equal = tokens_equal and base["logprobs"] == cand["logprobs"]
    max_lp = (
        max(abs(a - b) for a, b in zip(base["logprobs"], cand["logprobs"], strict=True))
        if tokens_equal
        else None
    )
    return {
        "ok": tokens_equal and lp_equal,
        "tokens_equal": tokens_equal,
        "logprobs_equal": lp_equal,
        "first_token_divergence": first,
        "max_abs_logprob_diff": max_lp,
        "n_tokens": len(base["ids"]),
    }


def compare_runs(base: dict, cand: dict) -> dict:
    if not (base.get("complete") and cand.get("complete")):
        return {"ok": False, "reason": "run not complete", "rows": []}
    br, cr = base["rows"], cand["rows"]
    if len(br) != len(cr) or [(r["ctx"], r["seed"]) for r in br] != [
        (r["ctx"], r["seed"]) for r in cr
    ]:
        return {"ok": False, "reason": "different request sets", "rows": []}
    rows = [
        {"ctx": b["ctx"], "seed": b["seed"], **compare_rows(b, c)}
        for b, c in zip(br, cr, strict=True)
    ]
    return {"ok": bool(rows) and all(r["ok"] for r in rows), "rows": rows}


def request(url: str, model: str, messages: list[dict]) -> dict:
    import httpx

    ids: list[int] = []
    lps: list[float] = []
    finish = None
    done = False
    stats: dict = {}
    t0 = time.perf_counter()
    first = None
    with httpx.stream(
        "POST",
        url + "/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
            "logprobs": True,
        },
        timeout=900,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                done = True
                continue
            obj = json.loads(line[6:])
            stats = obj.get("x_yunshu", stats)
            if obj.get("usage"):
                stats = obj["usage"].get("x_yunshu", stats)
                stats = {
                    **stats,
                    "usage": {
                        k: v for k, v in obj["usage"].items() if isinstance(v, int)
                    },
                }
            for choice in obj.get("choices", []):
                for lp in (choice.get("logprobs") or {}).get("content", []):
                    if "token_id" not in lp or "logprob" not in lp:
                        raise RuntimeError("exact token ids / logprobs unavailable")
                    ids.append(lp["token_id"])
                    lps.append(lp["logprob"])
                    if first is None:
                        first = time.perf_counter()
                finish = choice.get("finish_reason") or finish
    if not done or not finish or not ids:
        raise RuntimeError("incomplete response")
    return {
        "ids": ids,
        "logprobs": lps,
        "finish": finish,
        "ttft_s": (first - t0) if first else None,
        "stats": stats,
    }


def run(a) -> int:
    if not (os.environ.get("GPUQ_RC") and os.environ.get("GPUQ_DEVICE")):
        raise RuntimeError("GPU work requires gpuq")
    from research.agentic.servers import Server, start_server

    settings, mode = CONFIGS[a.config]
    env_keys = {**COMMON, **settings}
    os.environ.update(env_keys)
    os.environ["AGENTIC_YUNSHU_SRC"] = str(ROOT / "python")
    log = a.out.with_suffix(".server.log")
    server = None
    try:
        server, _ports = start_server(
            1,
            lambda p: Server("yunshu", a.model, p, log),
            wait_s=120.0,
        )
        engaged = server.engaged_spec_mode()
        if engaged != mode:
            raise RuntimeError(
                f"config {a.config}: draft mode {engaged!r}, wanted {mode!r}"
            )
        rows = []
        for p in build_prompts():
            r = request(server.url, server.model_id, p["messages"])
            r.update(ctx=p["ctx"], seed=p["seed"])
            rows.append(r)
            print(
                json.dumps(
                    {
                        "config": a.config,
                        "ctx": p["ctx"],
                        "seed": p["seed"],
                        "n": len(r["ids"]),
                        "usage": r["stats"].get("usage"),
                        "spec": (r["stats"].get("speculative") or {}),
                    }
                ),
                flush=True,
            )
        value = {
            "complete": True,
            "config": a.config,
            "settings": env_keys,
            "mode": engaged,
            "rows": rows,
        }
        a.out.write_text(json.dumps(value))
        return 0
    finally:
        if server is not None:
            server.kill()


def compare(a) -> int:
    base = json.loads(a.base.read_text())
    ok_all = True
    for c in a.cand:
        res = compare_runs(base, json.loads(c.read_text()))
        ok_all = ok_all and res["ok"]
        print(json.dumps({"cand": c.name, **res}))
    return 0 if ok_all else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", choices=sorted(CONFIGS), required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--base", type=Path, required=True)
    c.add_argument("--cand", type=Path, nargs="+", required=True)
    a = ap.parse_args(argv)
    return run(a) if a.cmd == "run" else compare(a)


if __name__ == "__main__":
    raise SystemExit(main())
