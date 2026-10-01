"""Replay saved agent request bodies against a fresh server and report how often the model
answers with a well-formed tool call (stop_reason / finish_reason) versus text.

    replay_bodies.py --checkpoint $M --bodies FILE [FILE...] --n 6 --path /v1/messages
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from servers import Server, free_ports  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bodies", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--path", default="/v1/messages")
    ap.add_argument("--log", default="/tmp/replay-server.log")
    a = ap.parse_args()
    (port,) = free_ports(1)
    srv = Server("yunshu", a.checkpoint, port, Path(a.log)).start()
    try:
        for f in a.bodies:
            body = json.loads(Path(f).read_text())
            body["stream"] = False
            kinds = []
            for _ in range(a.n):
                req = urllib.request.Request(
                    srv.url + a.path,
                    json.dumps(body).encode(),
                    {
                        "Content-Type": "application/json",
                        "x-api-key": "k",
                        "anthropic-version": "2023-06-01",
                    },
                )
                t = time.time()
                r = json.load(urllib.request.urlopen(req, timeout=600))
                blocks = [b.get("type") for b in r.get("content", [])]
                text = "".join(b.get("text", "") for b in r.get("content", []))
                kinds.append(
                    dict(
                        stop=r.get("stop_reason"),
                        blocks=blocks,
                        leaked="<tool_call>" in text,
                        s=round(time.time() - t, 1),
                        cache=(r.get("usage") or {}).get("cache_read_input_tokens"),
                    )
                )
                print(Path(f).name, kinds[-1], flush=True)
            ok = sum(1 for k in kinds if k["stop"] == "tool_use")
            print(f"{Path(f).name}: tool_use {ok}/{len(kinds)}", flush=True)
    finally:
        srv.kill()


if __name__ == "__main__":
    main()
