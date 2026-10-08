#!/usr/bin/env python3
"""Capability matrix runner: every row of capmatrix.json against one OpenAI/Anthropic-compatible endpoint.

    capmatrix --base-url http://127.0.0.1:18990 --model NAME --engine yunshu --out verdict.json

Fail closed: a row is pass only when every assertion held; a transport error, an unparsable reply or a
row that never ran is a fail/error, never a skip. A row is `na` only when the matrix says the engine does not
expose that API (`applies`) or the model lacks the declared feature (`--features`). The verdict has
complete=true only when every row got a verdict; exit code 0 only when complete and no row failed.
"""

from __future__ import annotations

import argparse
import http.client
import json
import re
import sys
import time
import urllib.parse
from pathlib import Path

MATRIX = Path(__file__).with_name("capmatrix.json")
NEEDLE = "PX-7741-QUARTZ"


# ---------------------------------------------------------------- transport
class Client:
    def __init__(self, base_url: str, model: str, timeout: float = 900.0):
        u = urllib.parse.urlparse(base_url)
        self.host, self.port, self.model, self.timeout = (
            u.hostname,
            u.port or 80,
            model,
            timeout,
        )

    def conn(self):
        return http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def headers(self, path):
        h = {"content-type": "application/json", "authorization": "Bearer k"}
        if path.startswith("/v1/messages"):
            h.update({"x-api-key": "k", "anthropic-version": "2023-06-01"})
        return h

    def request(self, method, path, body=None):
        c = self.conn()
        try:
            c.request(
                method,
                path,
                json.dumps(body) if body is not None else None,
                self.headers(path),
            )
            r = c.getresponse()
            raw = r.read()
        finally:
            c.close()
        try:
            data = json.loads(raw) if raw else None
        except ValueError:
            data = {"_raw": raw[:300].decode("utf-8", "replace")}
        return {"status": r.status, "json": data}

    def stream(self, path, body, kind, abort_after=None):
        """Read an SSE stream and aggregate it (kind openai | anthropic | responses)."""
        c = self.conn()
        agg = {
            "http_status": 0,
            "text": "",
            "reasoning": "",
            "tool_calls": [],
            "finish_reason": None,
            "usage": None,
            "done": False,
            "chunks": 0,
            "events": [],
            "usage_chunk_no_choices": False,
            "error": None,
        }
        calls: dict = {}
        try:
            c.request("POST", path, json.dumps(body), self.headers(path))
            r = c.getresponse()
            agg["http_status"] = r.status
            if r.status != 200:
                agg["error"] = r.read()[:300].decode("utf-8", "replace")
                return agg
            for line in r:
                line = line.strip()
                if line.startswith(b"event:"):
                    agg["events"].append(line[6:].strip().decode())
                    continue
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    agg["done"] = True
                    break
                d = json.loads(payload)
                agg["chunks"] += 1
                if abort_after and agg["chunks"] >= abort_after:
                    return agg
                _fold(kind, d, agg, calls)
                if kind == "anthropic" and d.get("type") == "message_stop":
                    agg["done"] = True
                if kind == "responses" and d.get("type") == "response.completed":
                    agg["done"] = True
        finally:
            c.close()
        agg["tool_calls"] = [calls[k] for k in sorted(calls)]
        return agg


def _fold(kind, d, agg, calls):
    if kind == "openai":
        if d.get("usage"):
            agg["usage"] = d["usage"]
            if not d.get("choices"):
                agg["usage_chunk_no_choices"] = True
        for ch in d.get("choices") or []:
            dl = ch.get("delta") or {}
            agg["text"] += dl.get("content") or ""
            agg["reasoning"] += dl.get("reasoning_content") or dl.get("reasoning") or ""
            for tc in dl.get("tool_calls") or []:
                e = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                f = tc.get("function") or {}
                e["name"] += f.get("name") or ""
                e["arguments"] += f.get("arguments") or ""
            if ch.get("finish_reason"):
                agg["finish_reason"] = ch["finish_reason"]
    elif kind == "anthropic":
        t = d.get("type")
        if t == "content_block_start":
            b = d.get("content_block") or {}
            if b.get("type") == "tool_use":
                calls[d["index"]] = {"name": b.get("name", ""), "arguments": ""}
        elif t == "content_block_delta":
            dl = d.get("delta") or {}
            agg["text"] += dl.get("text") or ""
            agg["reasoning"] += dl.get("thinking") or ""
            if dl.get("type") == "input_json_delta":
                calls[d["index"]]["arguments"] += dl.get("partial_json") or ""
        elif t == "message_delta":
            agg["finish_reason"] = (d.get("delta") or {}).get("stop_reason")
            agg["usage"] = d.get("usage")
    else:  # responses
        t = d.get("type", "")
        if t == "response.output_text.delta":
            agg["text"] += d.get("delta") or ""
        elif t == "response.completed":
            agg["usage"] = (d.get("response") or {}).get("usage")
            agg["finish_reason"] = (d.get("response") or {}).get("status")


# ---------------------------------------------------------------- assertions
_MISSING = object()


def jpath(obj, path):
    for part in path.split("."):
        if isinstance(obj, list):
            try:
                obj = obj[int(part)]
            except (ValueError, IndexError):
                return _MISSING
        elif isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            return _MISSING
    return obj


def check_assert(a, resp):
    v = jpath(resp, a["path"])
    op = a["op"]
    if op == "exists":
        return v is not _MISSING and v is not None, v
    if op == "absent":
        return v is _MISSING or v is None or v == [] or v == "", v
    if v is _MISSING:
        return False, "missing"
    want = a.get("value")
    ok = {
        "eq": lambda: v == want,
        "in": lambda: v in want,
        "contains": lambda: isinstance(v, str) and str(want).lower() in v.lower(),
        "not_contains": lambda: (
            isinstance(v, str) and str(want).lower() not in v.lower()
        ),
        "gte": lambda: isinstance(v, (int, float)) and v >= want,
        "lte": lambda: isinstance(v, (int, float)) and v <= want,
        "len_gte": lambda: hasattr(v, "__len__") and len(v) >= want,
        "len_eq": lambda: hasattr(v, "__len__") and len(v) == want,
        "nonempty": lambda: bool(v) and (not isinstance(v, str) or bool(v.strip())),
        "regex": lambda: (
            isinstance(v, str) and re.fullmatch(want, v.strip()) is not None
        ),
        "json_object": lambda: isinstance(_loads(v), dict),
        "json_has_keys": lambda: (
            isinstance(_loads(v), dict) and all(k in _loads(v) for k in want)
        ),
        "json_arg_nonempty": lambda: (
            isinstance(_loads(v), dict) and bool(_loads(v).get(want))
        ),
    }[op]()
    return bool(ok), v


def _loads(v):
    try:
        return json.loads(v) if isinstance(v, str) else v
    except ValueError:
        return None


# ---------------------------------------------------------------- body building
def _png_b64() -> str:
    """A 64x64 solid red PNG (stdlib only)."""
    import base64
    import struct
    import zlib

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + b"\xff\x00\x00" * 64 for _ in range(64))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(png).decode()


def _wav_b64() -> str:
    import base64
    import io
    import math
    import struct
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(
            b"".join(
                struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / 16000)))
                for i in range(16000)
            )
        )
    return base64.b64encode(buf.getvalue()).decode()


def _subst(o):
    if isinstance(o, dict):
        return {k: _subst(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_subst(v) for v in o]
    if o == "IMG":
        return "data:image/png;base64," + _png_b64()
    if o == "IMGB64":
        return _png_b64()
    if o == "AUDIO":
        return _wav_b64()
    return o


def build_body(row, model, think_off_style=None):
    body = _subst(json.loads(json.dumps(row["body"])))
    body.setdefault("model", model)
    if row.get("think") is False and row.get("path") == "/v1/chat/completions":
        body.setdefault("chat_template_kwargs", {"enable_thinking": False})
        body.setdefault("enable_thinking", False)
    return body


def long_doc(lines: int) -> str:
    out = [
        f"Log entry {i}: routine check passed, batch {(i * 7919) % 100003}."
        for i in range(lines)
    ]
    out[lines * 2 // 3] = (
        f"Log entry {lines * 2 // 3}: the maintenance access code is {NEEDLE}."
    )
    return "\n".join(out)


# ---------------------------------------------------------------- script rows
def chat(cl, content, **kw):
    body = {
        "model": cl.model,
        "messages": [{"role": "user", "content": content}],
        "chat_template_kwargs": {"enable_thinking": False},
        "enable_thinking": False,
    }
    body.update(kw)
    return cl.request("POST", "/v1/chat/completions", body)


def _text(r):
    try:
        return r["json"]["choices"][0]["message"]["content"] or ""
    except (TypeError, KeyError, IndexError):
        return ""


def s_seed(cl, row):
    outs = []
    for _ in range(2):
        r = chat(
            cl,
            "Write one sentence about the sea.",
            temperature=0.9,
            seed=1234,
            max_tokens=40,
        )
        if r["status"] != 200:
            return False, f"status {r['status']}"
        outs.append(_text(r))
    return bool(outs[0].strip()) and outs[0] == outs[1], outs


def s_cancel(cl, row):
    body = {
        "model": cl.model,
        "messages": [
            {"role": "user", "content": "Count from 1 to 1500, one number per line."}
        ],
        "max_tokens": 1500,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "enable_thinking": False,
    }
    a = cl.stream("/v1/chat/completions", body, "openai", abort_after=5)
    if a["chunks"] < 5:
        return False, f"stream gave {a['chunks']} chunks"
    t0 = time.time()
    r = chat(cl, "Reply with only the word ORCHID.", max_tokens=16)
    dt = time.time() - t0
    return r["status"] == 200 and "orchid" in _text(r).lower() and dt < row.get(
        "max_follow_s", 25
    ), {"follow_s": round(dt, 2)}


def s_cache(cl, row):
    doc = long_doc(row.get("lines", 400))
    cached = []
    for _ in range(2):
        r = chat(
            cl,
            doc + "\n\nWhat is the maintenance access code? Reply with the code only.",
            max_tokens=24,
        )
        if r["status"] != 200:
            return False, f"status {r['status']}"
        u = r["json"].get("usage") or {}
        cached.append((u.get("prompt_tokens_details") or {}).get("cached_tokens"))
    return isinstance(cached[1], int) and cached[1] > 0 and not cached[0], {
        "cached_tokens": cached
    }


def s_long(cl, row):
    r = chat(
        cl,
        long_doc(row.get("lines", 2600))
        + "\n\nWhat is the maintenance access code? Reply with the code only.",
        max_tokens=48,
    )
    if r["status"] != 200:
        return False, f"status {r['status']} {str(r['json'])[:120]}"
    pt = (r["json"].get("usage") or {}).get("prompt_tokens", 0)
    return pt >= row.get("min_prompt_tokens", 28000) and NEEDLE.lower() in _text(
        r
    ).lower(), {"prompt_tokens": pt, "text": _text(r)[:60]}


_NOISE = {
    "events",
    "usage",
    "tools",
    "instructions",
    "id",
    "object",
    "created",
    "temperature",
    "top_p",
    "model",
    "output_text",
}
SCRIPTS = {"seed": s_seed, "cancel": s_cancel, "cache": s_cache, "long": s_long}


# ---------------------------------------------------------------- running
def run_row(cl, row):
    try:
        if row.get("script"):
            ok, detail = SCRIPTS[row["script"]](cl, row)
            return {"status": "pass" if ok else "fail", "detail": detail}
        body = build_body(row, cl.model, None)
        if row.get("stream"):
            resp = cl.stream(row["path"], dict(body, stream=True), row["stream"])
        else:
            path = row.get("path_override", row["path"])
            resp = cl.request(
                row.get("method", "POST"),
                path,
                body if row.get("method", "POST") != "GET" else None,
            )
            resp = {
                "http_status": resp["status"],
                **(
                    resp["json"]
                    if isinstance(resp["json"], dict)
                    else {"body": resp["json"]}
                ),
            }
            resp["output_text"] = "".join(
                c.get("text", "")
                for o in resp.get("output") or []
                if isinstance(o, dict)
                for c in o.get("content") or []
                if isinstance(c, dict) and c.get("type") == "output_text"
            )
        fails = []
        want_status = row.get("status", 200)
        want_status = want_status if isinstance(want_status, list) else [want_status]
        if resp.get("http_status") not in want_status:
            fails.append(
                f"status {resp.get('http_status')} not in {want_status}: {str(resp)[:200]}"
            )
        else:
            for a in row["asserts"]:
                ok, v = check_assert(a, resp)
                if not ok:
                    fails.append(
                        f"{a['path']} {a['op']} {a.get('value', '')!s}: got {str(v)[:80]}"
                    )
        if fails:
            fails.append(
                "response: "
                + json.dumps(
                    {k: resp[k] for k in resp if k not in _NOISE},
                    ensure_ascii=False,
                    default=str,
                )[:1200]
            )
        return {"status": "fail" if fails else "pass", "detail": fails or "ok"}
    except Exception as e:  # noqa: BLE001 - any exception is a failed row, never a skip
        return {"status": "error", "detail": f"{type(e).__name__}: {e}"}


def run(matrix, base_url, model, engine, features, only=None, timeout=900.0):
    cl = Client(base_url, model, timeout)
    results = {}
    for row in matrix["rows"]:
        if only and row["id"] not in only:
            continue
        miss = [f for f in row.get("requires", []) if f not in features]
        if engine not in row["applies"]:
            results[row["id"]] = {
                "status": "na",
                "detail": f"{engine} does not expose this API",
            }
        elif miss:
            results[row["id"]] = {"status": "na", "detail": f"model lacks {miss}"}
        else:
            t0 = time.time()
            results[row["id"]] = run_row(cl, row)
            results[row["id"]]["seconds"] = round(time.time() - t0, 1)
    return verdict(matrix, results, engine, base_url, model, only)


def verdict(matrix, results, engine, base_url, model, only=None):
    ids = [r["id"] for r in matrix["rows"] if not only or r["id"] in only]
    complete = all(
        i in results and results[i]["status"] in ("pass", "fail", "error", "na")
        for i in ids
    )
    counts = {
        s: sum(r["status"] == s for r in results.values())
        for s in ("pass", "fail", "error", "na")
    }
    applicable = counts["pass"] + counts["fail"] + counts["error"]
    return {
        "engine": engine,
        "base_url": base_url,
        "model": model,
        "matrix_version": matrix["version"],
        "rows_total": len(ids),
        "complete": complete and not only,
        "counts": counts,
        "applicable": applicable,
        "all_pass": complete
        and not only
        and counts["fail"] == 0
        and counts["error"] == 0
        and counts["pass"] > 0,
        "rows": results,
    }


def load_matrix(path=MATRIX):
    m = json.loads(Path(path).read_text())
    ids = [r["id"] for r in m["rows"]]
    assert len(ids) == len(set(ids)), "duplicate row ids"
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", default="yunshu")
    ap.add_argument(
        "--features",
        default="vision,thinking",
        help="comma list: vision,audio,embeddings,thinking",
    )
    ap.add_argument("--only", action="append")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--matrix", type=Path, default=MATRIX)
    a = ap.parse_args(argv)
    v = run(
        load_matrix(a.matrix),
        a.base_url,
        a.model,
        a.engine,
        set(filter(None, a.features.split(","))),
        set(a.only or []),
    )
    text = json.dumps(v, indent=2, ensure_ascii=False)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")
    for i, r in v["rows"].items():
        print(
            f"{r['status']:5} {i}"
            + ("" if r["status"] in ("pass", "na") else f"  {r['detail']}")
        )
    print(
        f"engine={v['engine']} pass={v['counts']['pass']} fail={v['counts']['fail']} error={v['counts']['error']} "
        f"na={v['counts']['na']} complete={v['complete']}"
    )
    return 0 if v["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
