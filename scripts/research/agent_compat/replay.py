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
import stream_check  # noqa: E402

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


def load_requests(root, name: str) -> list[dict]:
    if name == "synthetic":
        import synthetic_requests

        return [
            dict(method="POST", path=c["path"], headers={}, body=c["body"], status=200)
            for c in synthetic_requests.cases()
        ]
    f = Path(root) / name / "requests.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines()] if f.exists() else []


def thin(reqs: list[dict], n: int) -> list[dict]:
    """First n-1 plus the last request (the long, cache-warm end of a session)."""
    return reqs if n <= 0 or len(reqs) <= n else [*reqs[: n - 1], reqs[-1]]


def verdict_stream(path: str, body, ev) -> list[str]:
    p = path.split("?")[0]
    if p == "/v1/messages":
        return stream_check.check_anthropic_stream(ev)
    if p == "/v1/chat/completions":
        usage = bool((body or {}).get("stream_options", {}).get("include_usage"))
        return stream_check.check_chat_stream(ev, expect_usage=usage)
    if p == "/v1/responses":
        return stream_check.check_responses_stream(ev)
    return []


def verdict_body(path: str, text: str) -> list[str]:
    try:
        return stream_check.check_body(path, json.loads(text))
    except ValueError:
        return [] if path.split("?")[0] not in ("/v1/messages", "/v1/responses", "/v1/chat/completions") else ["body is not JSON"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--census", default="", help="a census run dir (default: newest *-agent-census)")
    ap.add_argument("--sessions", default="")
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--url",
        default="",
        help="use an already running server instead of starting one",
    )
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--per-session", type=int, default=0, help="replay only the first N-1 and the last request of each session (0: all)")
    a = ap.parse_args()
    root = a.census or sorted(glob.glob(str(census.OUT_ROOT.parent / "*-agent-census")))[-1]
    if not a.url and not a.model:
        ap.error("--model or --url")
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
            for n in [*names, *([] if a.sessions else ["synthetic"])]:
                for i, r in enumerate(thin(load_requests(root, n), 0 if n == "synthetic" else a.per_session)):
                    body = r["body"]
                    hdr = {
                        k: v for k, v in r["headers"].items() if k.lower() not in DROP
                    }
                    hdr["authorization"] = "Bearer sk-yunshu-bench-dummy"
                    hdr["x-api-key"] = "sk-yunshu-bench-dummy"
                    if isinstance(body, dict) and ("model" in body or n == "synthetic"):
                        body = {**body, "model": model_id}
                        # keep replay fast: cap generation, the census checks protocol acceptance
                        for k in ("max_tokens", "max_output_tokens"):
                            if k in body and n != "synthetic":
                                body[k] = min(body[k], a.max_tokens)
                    t0 = time.time()
                    rec = dict(session=n, i=i, method=r["method"], path=r["path"])
                    upgrade = str(r["headers"].get("Upgrade") or r["headers"].get("upgrade") or "")
                    if r["method"] == "GET" and upgrade.lower() == "websocket":
                        rec.update(status=0, skipped="websocket upgrade (ws_probe.py covers it)")
                        res.append(rec)
                        continue
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
                            rec["problems"] = verdict_stream(r["path"], body, ev)
                        else:
                            rec["body"] = text[:1500]
                            rec["problems"] = verdict_body(r["path"], text)
                        want = r.get("status")
                        if want == 200 and not 200 <= resp.status_code < 300:
                            rec["problems"] = [
                                f"status {resp.status_code}, recorded client got {want}"
                            ] + rec.get("problems", [])
                    except Exception as e:
                        rec.update(status=-1, error=repr(e)[:300])
                    res.append(rec)
                    print(
                        f"{n}[{i}] {r['method']} {r['path'][:60]} -> {rec['status']}",
                        flush=True,
                    )
        (out / "replay.json").write_text(json.dumps(res, indent=1))
        bad = [r for r in res if r.get("status") == -1 or r.get("problems")]
        for r in bad:
            print("PROBLEM", r["session"], r["i"], r["path"], r.get("error") or r["problems"], flush=True)
        print(f"replay: {len(res)} requests, {len(bad)} with problems", flush=True)
        if not res or bad:
            sys.exit(1)
    finally:
        if srv:
            srv.kill()


if __name__ == "__main__":
    os.environ.setdefault("AGENTIC_YUNSHU_SRC", str(HERE.parents[2] / "python"))
    main()
