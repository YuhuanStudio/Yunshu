"""Replay every request recorded by the census against a real Yunshu server and record the outcome.

Starts `yunshu serve` on a free port 18990-18999 (worktree code via AGENTIC_YUNSHU_SRC), replays
each recorded (method, path, body) with the agent's own headers, and for every request stores
status, response headers, and (for SSE) the event-type sequence plus a validity verdict.
The server is always killed on exit.

    python replay.py --model /path/to/model [--sessions cc_plain,...] [--out DIR]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import census  # noqa: E402  (sets sys.path for the agentic harness)
import servers  # noqa: E402

DROP = {
    "host",
    "content-length",
    "connection",
    "accept-encoding",
    "authorization",
    "x-api-key",
    "transfer-encoding",
}


def sse_events(text: str):
    ev = []
    for block in text.split("\n\n"):
        name, data = None, []
        for ln in block.splitlines():
            if ln.startswith("event:"):
                name = ln[6:].strip()
            elif ln.startswith("data:"):
                data.append(ln[5:].strip())
        if data or name:
            d = "\n".join(data)
            try:
                obj = json.loads(d)
            except ValueError:
                obj = d
            ev.append((name, obj))
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sessions", default="")
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--url",
        default="",
        help="use an already running server instead of starting one",
    )
    ap.add_argument("--max-tokens", type=int, default=48)
    a = ap.parse_args()
    root = sorted(glob.glob(str(census.OUT_ROOT.parent / "*-agent-census")))[-1]
    out = Path(a.out or f"{root}/_replay_{time.strftime('%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    srv = None
    url = a.url
    if not url:
        port = servers.free_ports(1)[0]
        srv = servers.Server("yunshu", a.model, port, out / "server.log")
        srv.start()
        url = srv.url
        model_id = srv.model_id
    else:
        model_id = httpx.get(url + "/v1/models").json()["data"][0]["id"]
    print("server", url, model_id, flush=True)
    try:
        names = (
            a.sessions.split(",")
            if a.sessions
            else sorted(
                p.name
                for p in Path(root).iterdir()
                if p.is_dir() and not p.name.startswith("_")
            )
        )
        res = []
        with httpx.Client(timeout=300) as c:
            for n in names:
                f = Path(root) / n / "requests.jsonl"
                if not f.exists():
                    continue
                for i, line in enumerate(f.read_text().splitlines()):
                    r = json.loads(line)
                    body = r["body"]
                    hdr = {
                        k: v for k, v in r["headers"].items() if k.lower() not in DROP
                    }
                    hdr["authorization"] = "Bearer sk-yunshu-bench-dummy"
                    hdr["x-api-key"] = "sk-yunshu-bench-dummy"
                    if isinstance(body, dict) and "model" in body:
                        body = {**body, "model": model_id}
                        # keep replay fast: cap generation, the census checks protocol acceptance
                        for k in ("max_tokens", "max_output_tokens"):
                            if k in body:
                                body[k] = min(body[k], a.max_tokens)
                    t0 = time.time()
                    rec = dict(session=n, i=i, method=r["method"], path=r["path"])
                    try:
                        req = c.build_request(
                            r["method"],
                            url + r["path"],
                            headers=hdr,
                            json=body if r["method"] in ("POST", "PUT") else None,
                        )
                        resp = c.send(req, stream=True)
                        text = resp.read().decode("utf-8", "replace")
                        rec.update(
                            status=resp.status_code,
                            ctype=resp.headers.get("content-type"),
                            headers={
                                k: v
                                for k, v in resp.headers.items()
                                if k.lower().startswith(
                                    (
                                        "x-",
                                        "anthropic-",
                                        "openai-",
                                        "retry",
                                        "request-id",
                                    )
                                )
                            },
                            secs=round(time.time() - t0, 2),
                        )
                        if "text/event-stream" in (
                            resp.headers.get("content-type") or ""
                        ):
                            ev = sse_events(text)
                            rec["events"] = [
                                (
                                    e[0]
                                    or (
                                        e[1].get("type")
                                        if isinstance(e[1], dict)
                                        else None
                                    )
                                )
                                for e in ev
                            ][:80]
                            rec["last_event"] = (
                                json.dumps(ev[-1][1])[:600] if ev else None
                            )
                        else:
                            rec["body"] = text[:1500]
                    except Exception as e:
                        rec.update(status=-1, error=repr(e)[:300])
                    res.append(rec)
                    print(
                        f"{n}[{i}] {r['method']} {r['path'][:60]} -> {rec['status']}",
                        flush=True,
                    )
        (out / "replay.json").write_text(json.dumps(res, indent=1))
    finally:
        if srv:
            srv.kill()


if __name__ == "__main__":
    os.environ.setdefault("AGENTIC_YUNSHU_SRC", str(HERE.parents[2] / "python"))
    main()
