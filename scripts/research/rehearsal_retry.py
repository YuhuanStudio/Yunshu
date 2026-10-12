"""Does a retry of the same prompt still hit the cache after a next-turn rehearsal ran?

Sends prompt P, waits for the rehearsal window, resends P (a retry / regenerate) and records
how many prompt tokens were served from the cache. yv gate agent.restart_idle failed with
28159 of 41375 cached after an idle gap when the rehearsal was on.

    rehearsal_retry.py --model PATH --ctx 8192 --idle-s 8 --env YUNSHU_NEXT_TURN_PREFILL=exact --out r.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def judge(first: dict, retry: dict, minimum: float = 0.95) -> list[str]:
    """Reasons the retry failed to reuse the cache (pure; unit-tested)."""
    problems = []
    for tag, r in (("first", first), ("retry", retry)):
        if not r or not r.get("pt"):
            problems.append(f"{tag}: no prompt token count")
    if problems:
        return problems
    frac = (retry.get("cached") or 0) / retry["pt"]
    if frac < minimum:
        problems.append(
            f"retry cached {retry.get('cached') or 0} of {retry['pt']} prompt tokens ({frac:.0%} < {minimum:.0%})"
        )
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--idle-s", type=float, default=8.0)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--src", default=None)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.src:
        os.environ["TFB_YUNSHU_SRC"] = a.src
    os.environ.setdefault("TFB_OUT", str(a.out.parent / (a.out.stem + "-work")))
    sys.path.insert(0, str(HERE))
    import tfbench

    env = dict(kv.split("=", 1) for kv in a.env)
    env.setdefault("YUNSHU_VLM_DRAFT", "off")
    s = tfbench.Srv("yunshu", env, f"-retry-{a.out.stem}", model=a.model)
    try:
        text = tfbench.make_prompt("prose", a.ctx, 1)
        first = tfbench.send(s.url, tfbench.req(s.model, text, a.tokens))
        time.sleep(a.idle_s)
        retry = tfbench.send(s.url, tfbench.req(s.model, text, a.tokens))
        problems = judge(first, retry)
        rec = {
            "env": env,
            "first": {k: first.get(k) for k in ("pt", "cached", "ttft_s")},
            "retry": {k: retry.get(k) for k in ("pt", "cached", "ttft_s")},
            "problems": problems,
            "complete": True,
        }
    finally:
        s.kill()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec))
    return 0 if not rec["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
