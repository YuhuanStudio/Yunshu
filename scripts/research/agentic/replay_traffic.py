"""Replay captured agent request bodies (Chat Completions, from ``--save-bodies`` runs) against a
fresh server and report per-request TTFT, decode rate and speculative counters.

    replay_traffic.py --serve yunshu --checkpoint $M --bodies A.json B.json --n 3 \
        [--temperature 0] [--title BODY.json] [--env YUNSHU_ROUND_DRIVER=1] --out result.jsonl

Each body is sent ``--n`` times serially (the first one is cold, later ones hit the prefix cache).
``--title`` fires a second request concurrently with the first send of every body, like the
parallel no-tools title request opencode issues per task. ``--temperature`` overrides the body's
sampling (0: greedy) so the sampled and greedy speculative paths can be compared on the same bytes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from servers import Server, free_ports  # noqa: E402


def send(url: str, body: dict) -> dict:
    body = dict(body)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": "Bearer k"},
    )
    t0 = time.time()
    t_first = None
    text = []
    calls = 0
    usage = None
    xy = None
    finish = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            d = json.loads(payload)
            if d.get("x_yunshu") is not None:
                xy = d["x_yunshu"]
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                got = delta.get("content") or delta.get("reasoning_content")
                if delta.get("tool_calls"):
                    got = got or "x"
                    calls += 1
                if got and t_first is None:
                    t_first = time.time()
                if got:
                    text.append(got)
                finish = ch.get("finish_reason") or finish
    t1 = time.time()
    ct = (usage or {}).get("completion_tokens", 0)
    dec = (ct - 1) / (t1 - t_first) if t_first and ct > 1 and t1 > t_first else None
    return dict(
        ttft_s=round((t_first or t1) - t0, 3),
        total_s=round(t1 - t0, 3),
        completion_tokens=ct,
        prompt_tokens=(usage or {}).get("prompt_tokens"),
        decode_tok_s=round(dec, 1) if dec else None,
        finish=finish,
        tool_chunks=calls,
        spec=(xy or {}).get("speculative"),
        x_decode_tps=(xy or {}).get("decode_tps"),
        text="".join(text)[:400],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", choices=["yunshu", "tensorfold"], default="yunshu")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bodies", nargs="+", required=True)
    ap.add_argument("--title")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--server-arg", action="append", default=[])
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", default="/tmp/replay-traffic-server.log")
    a = ap.parse_args()
    for kv in a.env:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    (port,) = free_ports(1)
    srv = Server(a.serve, a.checkpoint, port, Path(a.log), extra=a.server_arg).start()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    title = json.loads(Path(a.title).read_text()) if a.title else None
    try:
        # warm the kernels/compile path so the first body's cold TTFT is prefill, not warmup
        for _ in range(2):
            send(
                srv.url,
                {
                    "model": json.loads(Path(a.bodies[0]).read_text()).get("model", "m"),
                    "messages": [{"role": "user", "content": "Say hi."}],
                    "max_tokens": 24,
                },
            )
        for f in a.bodies:
            body = json.loads(Path(f).read_text())
            if a.temperature is not None:
                body["temperature"] = a.temperature
            for i in range(a.n):
                side = {}
                th = None
                if title is not None and i == 0:
                    tb = dict(title)
                    if a.temperature is not None:
                        tb["temperature"] = a.temperature
                    th = threading.Thread(
                        target=lambda: side.update(send(srv.url, tb)), daemon=True
                    )
                    th.start()
                    time.sleep(0.3)
                rec = send(srv.url, body)
                if th is not None:
                    th.join()
                    rec["title"] = side
                rec.update(body=Path(f).name, i=i, label=a.label, cold=i == 0)
                print(json.dumps({k: v for k, v in rec.items() if k != "text"}), flush=True)
                with out.open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
    finally:
        srv.kill()


if __name__ == "__main__":
    main()
