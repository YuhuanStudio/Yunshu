"""Tool-call reliability on a recorded Claude Code request: N replays, count valid tool_use vs leaked markup.

Replays the first /v1/messages request of a census session (Claude Code's real system prompt and its 18+
tools) N times at the server's default sampling, and classifies each answer:
  tool_use   stop_reason tool_use with a parsed, known tool name
  leak       the visible text carries tool-call markup (<tool_call>, <function=, {"function": ...)
  text       an ordinary text answer with no tool call
  error      HTTP error

    python toolcall_ab.py --model M --src <python dir> --session cc_bash_edit --n 12 --out DIR
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import servers  # noqa: E402

LEAK = re.compile(r"<tool_call>|<function=|\{\s*\"function\"|<parameter=")


def classify(msg: dict, known: set[str]) -> str:
    blocks = msg.get("content") or []
    for b in blocks:
        if b.get("type") == "tool_use":
            return "tool_use" if b.get("name") in known else "leak"
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    return "leak" if LEAK.search(text) else "text"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--src", required=True, help="python/ dir to serve (worktree or a snapshot)"
    )
    ap.add_argument("--session", default="cc_bash_edit")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stream", action="store_true")
    a = ap.parse_args()
    root = sorted(
        glob.glob(
            "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/docs/research/runs/*-agent-census"
        )
    )[-1]
    with open(f"{root}/{a.session}/requests.jsonl") as f:
        rec = [json.loads(x) for x in f][a.index]
    body = rec["body"]
    known = {t["name"] for t in body.get("tools") or []}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["AGENTIC_YUNSHU_SRC"] = a.src
    port = servers.free_ports(1)[0]
    srv = servers.Server("yunshu", a.model, port, out / "server.log")
    counts: dict[str, int] = {}
    rows = []
    try:
        srv.start()
        body = {
            **body,
            "model": srv.model_id,
            "max_tokens": a.max_tokens,
            "stream": a.stream,
        }
        hdr = {
            k: v
            for k, v in rec["headers"].items()
            if k.lower().startswith(("anthropic-", "x-api-key"))
        }
        hdr["x-api-key"] = "k"
        with httpx.Client(timeout=900) as c:
            for i in range(a.n):
                t0 = time.time()
                r = c.post(srv.url + "/v1/messages", json=body, headers=hdr)
                if r.status_code != 200:
                    kind, detail = "error", r.text[:200]
                elif a.stream:
                    txt = r.text
                    kind = (
                        "tool_use"
                        if '"type": "tool_use"' in txt or '"tool_use"' in txt
                        else ("leak" if LEAK.search(txt) else "text")
                    )
                    detail = ""
                else:
                    msg = r.json()
                    kind = classify(msg, known)
                    detail = json.dumps(msg.get("content"))[:300]
                counts[kind] = counts.get(kind, 0) + 1
                rows.append(
                    {
                        "i": i,
                        "kind": kind,
                        "secs": round(time.time() - t0, 1),
                        "detail": detail,
                    }
                )
                print(i, kind, f"{time.time() - t0:.1f}s", detail[:110], flush=True)
        (out / "results.json").write_text(
            json.dumps({"counts": counts, "rows": rows}, indent=1)
        )
        print("COUNTS", counts, flush=True)
    finally:
        srv.kill()


if __name__ == "__main__":
    main()
