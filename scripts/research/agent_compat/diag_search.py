"""Why does a local model skip the search tool? Run the same 'search the web' request N times on the Messages and
Responses APIs and print, per run, whether a server tool was called and the model's own words.

    python diag_search.py --model M --src <python dir> --n 6 [--no-nudge]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(
    0, str(HERE.parent / "agentic")
)  # servers.py lives with the agentic harness
import fake_searxng  # noqa: E402
import servers  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", default="/tmp/diag_search")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sx = fake_searxng.make_server(0)
    threading.Thread(target=sx.serve_forever, daemon=True).start()
    os.environ["YUNSHU_SEARXNG_URL"] = f"http://127.0.0.1:{sx.server_address[1]}"
    os.environ["AGENTIC_YUNSHU_SRC"] = a.src
    port = servers.free_ports(1)[0]
    srv = servers.Server("yunshu", a.model, port, out / "server.log")
    try:
        srv.start()
        mid = srv.model_id
        prompts = [
            "Search the web: what is the reference release code name of Yunshu? Answer in one sentence.",
            "Perform a web search for the query: yunshu reference release code name",
        ]
        rows = []
        with httpx.Client(timeout=600) as c:
            for pi, prompt in enumerate(prompts):
                for i in range(a.n):
                    r = c.post(
                        srv.url + "/v1/messages",
                        headers={"x-api-key": "k", "anthropic-version": "2023-06-01"},
                        json={
                            "model": mid,
                            "max_tokens": 3000,
                            "tools": [
                                {"type": "web_search_20250305", "name": "web_search"}
                            ],
                            "messages": [{"role": "user", "content": prompt}],
                        },
                    )
                    m = r.json()
                    kinds = [b.get("type") for b in m.get("content", [])]
                    think = next(
                        (
                            b.get("thinking", "")
                            for b in m.get("content", [])
                            if b.get("type") == "thinking"
                        ),
                        "",
                    )
                    text = "".join(
                        b.get("text", "")
                        for b in m.get("content", [])
                        if b.get("type") == "text"
                    )
                    row = {
                        "api": "messages",
                        "prompt": pi,
                        "i": i,
                        "kinds": kinds,
                        "called": "server_tool_use" in kinds,
                        "think": think[:300],
                        "text": text[:200],
                        "rounds": (m.get("x_yunshu") or {})
                        .get("server_tools", {})
                        .get("rounds"),
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                for i in range(a.n):
                    r = c.post(
                        srv.url + "/v1/responses",
                        headers={"authorization": "Bearer k"},
                        json={
                            "model": mid,
                            "tools": [{"type": "web_search"}],
                            "input": prompt,
                            "max_output_tokens": 3000,
                        },
                    )
                    m = r.json()
                    types = [o.get("type") for o in m.get("output", [])]
                    text = m.get("output_text") or "".join(
                        cc.get("text", "")
                        for o in m.get("output", [])
                        if o.get("type") == "message"
                        for cc in o.get("content", [])
                    )
                    row = {
                        "api": "responses",
                        "prompt": pi,
                        "i": i,
                        "types": types,
                        "called": "web_search_call" in types,
                        "text": text[:200],
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
        called = sum(1 for r in rows if r["called"])
        print(f"CALLED {called}/{len(rows)}", flush=True)
        (out / "rows.json").write_text(json.dumps(rows, indent=1))
    finally:
        srv.kill()
        sx.shutdown()


if __name__ == "__main__":
    main()
