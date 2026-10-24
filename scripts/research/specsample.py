"""Sampled vs greedy decode speed and speculative acceptance (DFlash / MTP / off) on long reasoning turns.

    specsample.py --mode dflash|mtp|off --port 18993 --out r.jsonl [--ctx 16384 32768] [--reps 2]

One server per mode; requests interleave rep -> ctx -> sampling. Fails closed: wrong engaged mode,
an incomplete stream or a missing usage record aborts with a nonzero exit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "agentic"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev"))

CORPUS = Path("/Volumes/P5Plus/yunshu-build/snapshot014/corpus/prompts")
DRAFTER = "/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2"
MODEL = "/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
CHARS_PER_TOK = (
    3.9  # corrected from usage.prompt_tokens on the first request of each ctx
)
# Claude Code / Codex / opencode send no temperature/top_p/top_k (S5 bodies): the gateway
# defaults apply (temperature 0.7, top_p 1.0, top_k 0).
SAMPLINGS = {
    "greedy": {"temperature": 0},
    "agent_default": {},
    "qwen_think": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
}
TASK = (
    "\n\n---\nYou are a senior engineer. Study the material above carefully, find the "
    "three most serious design problems in it, and write a detailed step-by-step plan to "
    "fix each one, with concrete code. Think the whole problem through before answering."
)
MODE_ENV = {"dflash": None, "mtp": "mtp", "off": "off"}


def build_prompt(ctx_tokens: int, chars_per_tok: float = CHARS_PER_TOK) -> str:
    src = (CORPUS / "code-32768.txt").read_text()
    n = max(int((ctx_tokens - 400) * chars_per_tok), 1000)
    return src[:n] + TASK


def sampling_body(prompt: str, name: str, max_tokens: int) -> dict:
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body.update(SAMPLINGS[name])
    return body


def decode_rate(completion_tokens: int, t_first: float, t_end: float):
    if completion_tokens > 1 and t_end > t_first:
        return (completion_tokens - 1) / (t_end - t_first)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(MODE_ENV), required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, nargs="+", default=[16384, 32768])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--samplings", nargs="+", default=list(SAMPLINGS))
    a = ap.parse_args()

    os.environ["AGENTIC_SERVER_HOME"] = "/Volumes/P5Plus/yunshu-build/specsample-home"
    os.environ["YUNSHU_VLM_APC_DISK"] = "0"
    if a.mode == "dflash":
        os.environ["YUNSHU_VLM_DRAFT"] = DRAFTER
    else:
        os.environ["YUNSHU_VLM_DRAFT"] = MODE_ENV[a.mode]
    import replay_traffic
    from servers import Server

    log = Path(a.out).with_suffix(f".{a.mode}.server.log")
    srv = Server("yunshu", MODEL, a.port, log).start()
    try:
        eng = srv.engaged_spec_mode()
        print(f"engaged draft mode: {eng} (wanted {a.mode})", flush=True)
        if eng != ("dflash" if a.mode == "dflash" else a.mode):
            print("FAIL: wrong engaged mode\n" + srv.log_tail(), flush=True)
            return 2
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        cpt = {}
        warm = sampling_body("Say hi.", "greedy", 16)
        warm["model"] = srv.model_id
        replay_traffic.send(srv.url, warm)
        for rep in range(a.reps):
            for ctx in a.ctx:
                for name in a.samplings:
                    prompt = build_prompt(ctx, cpt.get(ctx, CHARS_PER_TOK))
                    body = sampling_body(prompt, name, a.max_tokens)
                    body["model"] = srv.model_id
                    r = replay_traffic.send(srv.url, body)
                    if not r.get("stream_done") or not r.get("completion_tokens"):
                        print("FAIL: incomplete", r, flush=True)
                        return 3
                    if ctx not in cpt and r.get("prompt_tokens"):
                        cpt[ctx] = len(prompt) / r["prompt_tokens"]
                    rec = dict(
                        mode=a.mode,
                        sampling=name,
                        ctx=ctx,
                        rep=rep,
                        prompt_tokens=r["prompt_tokens"],
                        completion_tokens=r["completion_tokens"],
                        decode_tok_s=r["decode_tok_s"],
                        x_decode_tps=r["x_decode_tps"],
                        ttft_s=r["ttft_s"],
                        total_s=r["total_s"],
                        finish=r["finish"],
                        cached_tokens=r["cached_tokens"],
                        spec=r["spec"],
                        reasoning_head=r["reasoning"][:120],
                    )
                    with out.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(json.dumps(rec), flush=True)
        return 0
    finally:
        srv.kill()
        time.sleep(1)


if __name__ == "__main__":
    sys.exit(main())
