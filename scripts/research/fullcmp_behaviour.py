#!/usr/bin/env python3
"""B4 behaviour checks for the full comparison (manifest batch B4, one run per engine).

Seven checks, each one HTTP exchange (or two) against a running OpenAI-compatible server:

  cancel_latency   client drops a stream after 3 chunks; a follow-up request must answer within 10 s
  stop_sequences   a stop string taken from the reply stops it (finish=stop, string not in content)
  tool_validity    a tool call with JSON arguments that name a declared tool and carry its required keys
  json_validity    response_format json_schema: the content parses and carries the schema's required keys
  logprobs         logprobs=true: one entry per completion token, each a finite logprob <= 0
  anthropic_api    POST /v1/messages answers a Messages-shaped body (type=message, text block)
  responses_api    POST /v1/responses answers a Responses-shaped body (object=response, output list)

Record schema (what fullcmp_audit's behaviour guard reads), one JSONL file per engine:
  row 0      {"part": "behaviour_session", "engine", "run_id", "url", "model", "started_epoch"}
  one row per check, {"part": "behaviour", "check", "status": "pass"|"fail"|"unsupported",
                      "reason": str|null, "detail": {...}}
  "unsupported" is only a route that answers 404 (or 405) for the anthropic/responses checks; any
  other failure is "fail". The audit compares each status with the manifest's expectation.

Usage:
  fullcmp_behaviour.py --engine yunshu --url http://127.0.0.1:18990 --model q27-4bit \
      --out <raw>/behaviour/yunshu.jsonl [--run-id ID]
Exit 0 only when no check failed (unsupported is not a failure here; the audit decides).
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

CHECKS = (
    "cancel_latency",
    "stop_sequences",
    "tool_validity",
    "json_validity",
    "logprobs",
    "anthropic_api",
    "responses_api",
)
CANCEL_FOLLOWUP_MAX_S = 10.0
PROMPT = "List three colours, one per line."

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
JSON_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "ok": {"type": "boolean"}},
    "required": ["city", "ok"],
}


class RouteMissing(Exception):
    """The route answers 404/405: the engine does not serve it (reported as unsupported)."""


class Client:
    def __init__(self, url: str, model: str, timeout: float = 120.0):
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout

    def post(self, path: str, body: dict):
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            self.url + path,
            data,
            {"Content-Type": "application/json", "Authorization": "Bearer k"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as exc:
            if exc.code in (404, 405):
                raise RouteMissing(f"{path} answered {exc.code}") from exc
            raise RuntimeError(f"{path} answered {exc.code}: {exc.read()[:200]!r}") from exc

    def chat(self, **extra) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 64,
            "temperature": 0,
            **extra,
        }
        status, data = self.post("/v1/chat/completions", body)
        if status != 200:
            raise RuntimeError(f"chat answered {status}")
        return data


def check_cancel_latency(c: Client) -> dict:
    parsed = urllib.parse.urlparse(c.url)
    body = json.dumps(
        {
            "model": c.model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 512,
            "stream": True,
        }
    ).encode()
    head = (
        f"POST /v1/chat/completions HTTP/1.1\r\nHost: {parsed.hostname}:{parsed.port}\r\n"
        "Content-Type: application/json\r\nAuthorization: Bearer k\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
    ).encode()
    sock = socket.create_connection((parsed.hostname, parsed.port), timeout=c.timeout)
    try:
        sock.sendall(head + body)
        seen = b""
        while seen.count(b"data:") < 3:
            chunk = sock.recv(4096)
            if not chunk:
                break
            seen += chunk
        if seen.count(b"data:") < 3:
            return _fail("stream ended before 3 chunks", {"chunks": seen.count(b"data:")})
    finally:
        t_cancel = time.monotonic()
        sock.close()
    t0 = time.monotonic()
    c.chat(max_tokens=4)
    follow = time.monotonic() - t0
    detail = {"cancel_followup_s": round(follow, 3), "dropped_at_chunks": 3}
    if follow > CANCEL_FOLLOWUP_MAX_S:
        return _fail(f"follow-up took {follow:.1f}s after cancel", detail)
    detail["cancel_t"] = round(t_cancel, 3)
    return _pass(detail)


def check_stop_sequences(c: Client) -> dict:
    data = c.chat(max_tokens=64)
    text = data["choices"][0]["message"]["content"] or ""
    words = text.split(" ")
    if len(words) < 4 or not words[3]:
        return _fail("reply too short to pick a stop string", {"words": len(words)})
    stop = words[3]
    data = c.chat(max_tokens=64, stop=[stop])
    ch = data["choices"][0]
    content = ch["message"]["content"] or ""
    detail = {"stop": stop, "finish_reason": ch.get("finish_reason")}
    if ch.get("finish_reason") != "stop" or stop in content:
        return _fail("stop string did not end the reply", detail)
    return _pass(detail)


def check_tool_validity(c: Client) -> dict:
    data = c.chat(tools=[TOOL], tool_choice="required")
    msg = data["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if not calls:
        return _fail("no tool call returned", {"finish_reason": data["choices"][0].get("finish_reason")})
    fn = calls[0].get("function") or {}
    try:
        args = json.loads(fn.get("arguments") or "")
    except ValueError:
        return _fail("tool arguments are not JSON", {"arguments": str(fn.get("arguments"))[:120]})
    if fn.get("name") != TOOL["function"]["name"] or not isinstance(args, dict):
        return _fail("tool call names an undeclared tool or non-object arguments", {"name": fn.get("name")})
    missing = [k for k in TOOL["function"]["parameters"]["required"] if k not in args]
    if missing:
        return _fail(f"tool arguments miss required keys {missing}", {"args": args})
    return _pass({"name": fn["name"], "args_keys": sorted(args)})


def check_json_validity(c: Client) -> dict:
    data = c.chat(
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "city", "schema": JSON_SCHEMA, "strict": True},
        }
    )
    content = data["choices"][0]["message"]["content"] or ""
    try:
        obj = json.loads(content)
    except ValueError:
        return _fail("content is not JSON", {"head": content[:80]})
    missing = [k for k in JSON_SCHEMA["required"] if not isinstance(obj, dict) or k not in obj]
    if missing:
        return _fail(f"JSON misses required keys {missing}", {"head": content[:80]})
    return _pass({"keys": sorted(obj)})


def check_logprobs(c: Client) -> dict:
    data = c.chat(logprobs=True, top_logprobs=1)
    ch = data["choices"][0]
    entries = ((ch.get("logprobs") or {}).get("content")) or []
    n_tok = (data.get("usage") or {}).get("completion_tokens")
    detail = {"entries": len(entries), "completion_tokens": n_tok}
    if not entries:
        return _fail("no logprobs.content", detail)
    if n_tok is not None and len(entries) != n_tok:
        return _fail("logprob entries differ from completion tokens", detail)
    for e in entries:
        lp = e.get("logprob")
        if not isinstance(lp, (int, float)) or not math.isfinite(lp) or lp > 1e-6:
            return _fail("a logprob is not a finite value <= 0", {**detail, "bad": str(lp)})
    return _pass(detail)


def _route_check(c: Client, path: str, body: dict, shape) -> dict:
    try:
        status, data = c.post(path, body)
    except RouteMissing as exc:
        return {"status": "unsupported", "reason": str(exc), "detail": {"path": path}}
    if status != 200:
        return _fail(f"{path} answered {status}", {"path": path})
    problem = shape(data)
    if problem:
        return _fail(problem, {"path": path})
    return _pass({"path": path})


def check_anthropic_api(c: Client) -> dict:
    def shape(d):
        if d.get("type") != "message":
            return "anthropic body is not type=message"
        blocks = d.get("content") or []
        if not blocks or blocks[0].get("type") != "text":
            return "anthropic content has no text block"
        return None

    return _route_check(
        c,
        "/v1/messages",
        {"model": c.model, "max_tokens": 16, "messages": [{"role": "user", "content": PROMPT}]},
        shape,
    )


def check_responses_api(c: Client) -> dict:
    def shape(d):
        if d.get("object") != "response":
            return "responses body is not object=response"
        if not isinstance(d.get("output"), list) or not d["output"]:
            return "responses body has no output"
        return None

    return _route_check(
        c,
        "/v1/responses",
        {"model": c.model, "input": PROMPT, "max_output_tokens": 16},
        shape,
    )


RUNNERS = {
    "cancel_latency": check_cancel_latency,
    "stop_sequences": check_stop_sequences,
    "tool_validity": check_tool_validity,
    "json_validity": check_json_validity,
    "logprobs": check_logprobs,
    "anthropic_api": check_anthropic_api,
    "responses_api": check_responses_api,
}


def _pass(detail: dict) -> dict:
    return {"status": "pass", "reason": None, "detail": detail}


def _fail(reason: str, detail: dict) -> dict:
    return {"status": "fail", "reason": reason, "detail": detail}


def run_check(c: Client, name: str) -> dict:
    """One check's record body. An exception is a failure with its message, never a crash."""
    try:
        res = RUNNERS[name](c)
    except (RuntimeError, OSError, ValueError, KeyError, IndexError) as exc:
        res = _fail(f"{type(exc).__name__}: {exc}", {})
    return {"part": "behaviour", "check": name, **res}


def run(engine: str, url: str, model: str, out: Path, run_id: str, checks=CHECKS) -> int:
    c = Client(url, model)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "part": "behaviour_session",
            "engine": engine,
            "run_id": run_id,
            "url": url,
            "model": model,
            "started_epoch": round(time.time(), 3),
        }
    ]
    for name in checks:
        rows.append(run_check(c, name))
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    failed = [r["check"] for r in rows if r.get("part") == "behaviour" and r["status"] == "fail"]
    for r in rows[1:]:
        print(f"{r['check']:<15} {r['status']:<11} {r.get('reason') or ''}", flush=True)
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--checks", default=",".join(CHECKS))
    a = ap.parse_args(argv)
    wanted = [c for c in a.checks.split(",") if c]
    bad = [c for c in wanted if c not in RUNNERS]
    if bad:
        ap.error(f"unknown checks {bad}")
    return run(a.engine, a.url, a.model, a.out, a.run_id, tuple(wanted))


if __name__ == "__main__":
    sys.exit(main())
