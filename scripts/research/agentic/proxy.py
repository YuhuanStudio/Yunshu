"""Engine-neutral recording HTTP proxy that sits between a coding agent and a model server.

Streaming-transparent: bytes are forwarded as they arrive, and the proxy only *observes* them.
For every request it records time to first byte, time to first generated token, total time,
token usage (prompt / completion / cached), decode rate, finish reason, tool-call validity and
API errors. Understands OpenAI Chat Completions, Anthropic Messages and OpenAI Responses, both
streaming (SSE) and plain JSON. Error bodies are always saved; all bodies with ``save_bodies``.

It never changes what the agent sends or what the server answers.
"""

from __future__ import annotations

import contextlib
import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "host",
    "accept-encoding",
}
LEAK_MARKERS = ("<tool_call>", "<function=", "<|tool_call", "[TOOL_CALLS]")


def api_kind(path: str) -> str | None:
    p = path.split("?", 1)[0].rstrip("/")
    if p.endswith("/chat/completions"):
        return "chat"
    if p.endswith("/messages"):
        return "messages"
    if p.endswith("/responses"):
        return "responses"
    return None


def _json_ok(text: str) -> bool:
    if not text.strip():
        return True
    try:
        v = json.loads(text)
    except ValueError:
        return False
    return isinstance(v, dict)


class Tracker:
    """Accumulates one generation request's observations from parsed events / bodies."""

    def __init__(self, kind: str | None, t0: float):
        self.kind = kind
        self.t0 = t0
        self.t_first_token: float | None = None
        self.t_last_token: float | None = None
        self.prompt = self.completion = self.cached = self.cache_write = None
        self.finish: str | None = None
        self.tool_args: dict[str, str] = {}
        self.tool_done: dict[str, str] = {}
        self.leaked = False
        self._leak_tail = ""
        self.text_parts: list[str] = []
        self.delta_events = 0
        self.x_yunshu = None
        self.error_event = None
        self._blocks: dict[int, str] = {}

    def _tok(self, t: float):
        if self.t_first_token is None:
            self.t_first_token = t
        self.t_last_token = t
        self.delta_events += 1

    def _text(self, s: str):
        if s:
            self.text_parts.append(s)
            window = self._leak_tail + s
            if not self.leaked and any(m in window for m in LEAK_MARKERS):
                self.leaked = True
            self._leak_tail = window[-(max(map(len, LEAK_MARKERS)) - 1) :]

    def feed_event(self, ev: dict, t: float):
        if not isinstance(ev, dict):
            return
        if ev.get("x_yunshu") is not None:
            self.x_yunshu = ev["x_yunshu"]
        if ev.get("error") and self.kind != "responses":
            self.error_event = ev["error"]
        if self.kind == "chat":
            self._chat(ev, t)
        elif self.kind == "messages":
            self._messages(ev, t)
        elif self.kind == "responses":
            self._responses(ev, t)

    def _chat(self, ev, t):
        u = ev.get("usage")
        if u:
            self._chat_usage(u)
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            got = False
            for k in ("content", "reasoning_content", "reasoning"):
                if d.get(k):
                    got = True
                    if k == "content":
                        self._text(d[k])
            for tc in d.get("tool_calls") or []:
                got = True
                key = str(tc.get("index", 0))
                fn = tc.get("function") or {}
                self.tool_args[key] = self.tool_args.get(key, "") + (
                    fn.get("arguments") or ""
                )
            if got:
                self._tok(t)
            if ch.get("finish_reason"):
                self.finish = ch["finish_reason"]

    def _chat_usage(self, u):
        self.prompt = u.get("prompt_tokens")
        self.completion = u.get("completion_tokens")
        det = u.get("prompt_tokens_details") or {}
        self.cached = det.get("cached_tokens")

    def _anthropic_usage(self, u):
        if not u:
            return
        if u.get("output_tokens") is not None:
            self.completion = u["output_tokens"]
        if "input_tokens" in u:
            base = u.get("input_tokens") or 0
            read = u.get("cache_read_input_tokens") or 0
            write = u.get("cache_creation_input_tokens") or 0
            # Anthropic input_tokens excludes cached tokens; report the full prompt.
            self.prompt = base + read + write
            self.cached = read
            self.cache_write = write

    def _messages(self, ev, t):
        ty = ev.get("type")
        if ty == "message_start":
            self._anthropic_usage((ev.get("message") or {}).get("usage"))
        elif ty == "content_block_start":
            cb = ev.get("content_block") or {}
            idx = ev.get("index", 0)
            self._blocks[idx] = cb.get("type", "")
            if cb.get("type") == "tool_use":
                self.tool_args.setdefault(str(idx), "")
                self._tok(t)
        elif ty == "content_block_delta":
            d = ev.get("delta") or {}
            idx = str(ev.get("index", 0))
            if d.get("type") == "input_json_delta":
                self.tool_args[idx] = self.tool_args.get(idx, "") + (
                    d.get("partial_json") or ""
                )
                self._tok(t)
            elif d.get("type") == "text_delta":
                self._text(d.get("text") or "")
                self._tok(t)
            elif d.get("type") == "thinking_delta":
                self._tok(t)
        elif ty == "message_delta":
            self._anthropic_usage(ev.get("usage"))
            sr = (ev.get("delta") or {}).get("stop_reason")
            if sr:
                self.finish = sr
        elif ty == "error":
            self.error_event = ev.get("error")

    def _responses(self, ev, t):
        ty = ev.get("type", "")
        if ty.endswith(".delta"):
            if ty == "response.output_text.delta":
                self._text(ev.get("delta") or "")
            self._tok(t)
        elif ty == "response.output_item.done":
            it = ev.get("item") or {}
            if it.get("type") == "function_call":
                self.tool_done[it.get("call_id") or it.get("id") or "?"] = (
                    it.get("arguments") or ""
                )
        elif ty in ("response.completed", "response.incomplete", "response.failed"):
            r = ev.get("response") or {}
            u = r.get("usage") or {}
            self.prompt = u.get("input_tokens")
            self.completion = u.get("output_tokens")
            self.cached = (u.get("input_tokens_details") or {}).get("cached_tokens")
            self.finish = ty.split(".", 1)[1]
            if r.get("x_yunshu") is not None:
                self.x_yunshu = r["x_yunshu"]
            if r.get("error"):
                self.error_event = r["error"]
            for it in r.get("output") or []:
                if it.get("type") == "function_call":
                    self.tool_done.setdefault(
                        it.get("call_id") or it.get("id") or "?",
                        it.get("arguments") or "",
                    )
        elif ty == "error":
            self.error_event = ev

    def feed_json(self, body: dict, t: float):
        """A non-streaming JSON response."""
        if not isinstance(body, dict):
            return
        self.t_first_token = self.t_last_token = t
        if body.get("x_yunshu") is not None:
            self.x_yunshu = body["x_yunshu"]
        if self.kind == "chat":
            if body.get("usage"):
                self._chat_usage(body["usage"])
            for i, ch in enumerate(body.get("choices") or []):
                m = ch.get("message") or {}
                self._text(m.get("content") or "")
                for j, tc in enumerate(m.get("tool_calls") or []):
                    self.tool_args[f"{i}.{j}"] = (tc.get("function") or {}).get(
                        "arguments"
                    ) or ""
                self.finish = ch.get("finish_reason") or self.finish
        elif self.kind == "messages":
            self._anthropic_usage(body.get("usage"))
            self.finish = body.get("stop_reason")
            for i, b in enumerate(body.get("content") or []):
                if b.get("type") == "text":
                    self._text(b.get("text") or "")
                elif b.get("type") == "tool_use":
                    self.tool_done[str(i)] = json.dumps(b.get("input") or {})
        elif self.kind == "responses":
            self._responses({"type": "response.completed", "response": body}, t)
            for it in body.get("output") or []:
                if it.get("type") == "message":
                    for c in it.get("content") or []:
                        self._text(c.get("text") or "")

    def summary(self, t_end: float) -> dict:
        args = {**self.tool_args, **self.tool_done}
        malformed = sum(1 for v in args.values() if not _json_ok(v))
        out = dict(
            prompt_tokens=self.prompt,
            completion_tokens=self.completion,
            cached_tokens=self.cached,
            finish_reason=self.finish,
            tool_calls=len(args),
            malformed_tool_calls=malformed,
            leaked_tool_markup=self.leaked,
            delta_events=self.delta_events,
        )
        if self.cache_write:
            out["cache_write_tokens"] = self.cache_write
        if self.t_first_token is not None:
            out["ttft_s"] = round(self.t_first_token - self.t0, 4)
            n = self.completion if self.completion else self.delta_events
            span = (self.t_last_token or t_end) - self.t_first_token
            # decode rate excludes the first token (it ends prefill)
            if n and n > 1 and span > 0:
                out["decode_tok_s"] = round((n - 1) / span, 2)
        if self.x_yunshu is not None:
            out["x_yunshu"] = self.x_yunshu
        if self.error_event:
            out["error_event"] = self.error_event
        return out


class RecordingProxy:
    def __init__(
        self,
        upstream: str,
        host: str = "127.0.0.1",
        port: int = 0,
        bodies_dir: Path | None = None,
        save_bodies: bool = False,
    ):
        u = urlsplit(upstream)
        self.up_host, self.up_port = u.hostname, u.port or 80
        self.bodies_dir = Path(bodies_dir) if bodies_dir else None
        self.save_bodies = save_bodies
        self.records: list[dict] = []
        self._lock = threading.Lock()
        self._seq = 0
        self.pending = 0
        self.active: dict[int, tuple] = {}
        proxy = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _serve(self):
                proxy._handle(self)

        for _m in ("GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"):
            setattr(H, "do_" + _m, H._serve)  # noqa: B010

        self.httpd = ThreadingHTTPServer((host, port), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.url = f"http://{host}:{self.port}"
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def reset(self):
        with self._lock:
            self.records = []

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.records)

    def inflight(self) -> list[dict]:
        """Requests still running (e.g. when the agent was killed at the time limit)."""
        now = time.perf_counter()
        out = []
        for rec, tr, t0 in list(self.active.values()):
            d = {k: rec[k] for k in ("id", "kind", "n_tools", "n_messages") if k in rec}
            d.update(
                inflight=True,
                running_s=round(now - t0, 1),
                delta_events=tr.delta_events,
            )
            if tr.t_first_token is not None:
                d["ttft_s"] = round(tr.t_first_token - t0, 3)
            out.append(d)
        return out

    def _next_id(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    def _save(self, rid: int, suffix: str, data: bytes):
        if not self.bodies_dir:
            return None
        self.bodies_dir.mkdir(parents=True, exist_ok=True)
        p = self.bodies_dir / f"{rid:04d}-{suffix}"
        p.write_bytes(data)
        return str(p)

    def _read_request_body(self, h) -> bytes:
        if h.headers.get("Transfer-Encoding", "").lower() == "chunked":
            out = b""
            while True:
                n = int(h.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if n == 0:
                    h.rfile.readline()
                    return out
                out += h.rfile.read(n)
                h.rfile.readline()
        n = int(h.headers.get("Content-Length") or 0)
        return h.rfile.read(n) if n else b""

    def _handle(self, h):
        with self._lock:
            self.pending += 1
        t0 = time.perf_counter()
        wall0 = time.time()
        rid = self._next_id()
        body = self._read_request_body(h)
        kind = api_kind(h.path) if h.command == "POST" else None
        rec = dict(
            id=rid,
            t_wall=wall0,
            method=h.command,
            path=h.path.split("?", 1)[0],
            kind=kind,
            req_bytes=len(body),
        )
        try:
            rj = json.loads(body) if body and kind else None
        except ValueError:
            rj = None
        if isinstance(rj, dict):
            rec["model"] = rj.get("model")
            rec["stream"] = bool(rj.get("stream"))
            rec["n_tools"] = len(rj.get("tools") or [])
            msgs = rj.get("messages") or rj.get("input")
            rec["n_messages"] = len(msgs) if isinstance(msgs, list) else None
            for k in ("temperature", "top_p", "max_tokens", "max_output_tokens"):
                if rj.get(k) is not None:
                    rec.setdefault("sampling", {})[k] = rj[k]
        if self.save_bodies and kind:
            rec["req_file"] = self._save(rid, "req.json", body)
        tr = Tracker(kind, t0)
        self.active[rid] = (rec, tr, t0)
        resp_buf = bytearray()
        conn = None
        try:
            conn = http.client.HTTPConnection(self.up_host, self.up_port, timeout=3600)
            hdr = {k: v for k, v in h.headers.items() if k.lower() not in HOP}
            hdr["Accept-Encoding"] = "identity"
            conn.request(h.command, h.path, body=body or None, headers=hdr)
            resp = conn.getresponse()
            rec["ttfb_s"] = round(time.perf_counter() - t0, 4)
            rec["status"] = resp.status
            ctype = resp.getheader("Content-Type", "")
            sse = "text/event-stream" in ctype
            length = resp.getheader("Content-Length")
            h.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in HOP:
                    h.send_header(k, v)
            chunked = length is None
            if chunked:
                h.send_header("Transfer-Encoding", "chunked")
            else:
                h.send_header("Content-Length", length)
            h.end_headers()
            line_buf = b""
            keep = kind and (sse or "json" in ctype)
            while True:
                chunk = resp.read1(65536) if h.command != "HEAD" else b""
                if not chunk:
                    break
                if chunked:
                    h.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                else:
                    h.wfile.write(chunk)
                h.wfile.flush()
                if keep or resp.status >= 400:
                    resp_buf += chunk
                if kind and sse:
                    t = time.perf_counter()
                    line_buf += chunk
                    *lines, line_buf = line_buf.split(b"\n")
                    for ln in lines:
                        ln = ln.strip()
                        if ln.startswith(b"data:"):
                            d = ln[5:].strip()
                            if d and d != b"[DONE]":
                                with contextlib.suppress(ValueError):
                                    tr.feed_event(json.loads(d), t)
            if chunked:
                h.wfile.write(b"0\r\n\r\n")
                h.wfile.flush()
            t_end = time.perf_counter()
            if kind and not sse and resp.status < 400 and resp_buf:
                try:
                    tr.feed_json(json.loads(bytes(resp_buf)), t_end)
                except ValueError:
                    rec["bad_json_response"] = True
        except (BrokenPipeError, ConnectionResetError):
            rec["client_aborted"] = True
            t_end = time.perf_counter()
        except Exception as e:  # upstream refused / dropped
            rec["proxy_error"] = f"{type(e).__name__}: {e}"
            t_end = time.perf_counter()
            try:
                msg = json.dumps({"error": {"message": rec["proxy_error"]}}).encode()
                h.send_response(502)
                h.send_header("Content-Type", "application/json")
                h.send_header("Content-Length", str(len(msg)))
                h.end_headers()
                h.wfile.write(msg)
            except Exception:
                pass
        finally:
            if conn is not None:
                conn.close()
        rec["total_s"] = round(t_end - t0, 4)
        if kind:
            rec.update(tr.summary(t_end))
        status = rec.get("status", 0)
        if status >= 400 or rec.get("proxy_error"):
            rec["error"] = True
            rec["error_body"] = bytes(resp_buf[:2000]).decode(errors="replace")
            rec["error_req_file"] = self._save(rid, "error-req.json", body)
            self._save(rid, "error-resp.txt", bytes(resp_buf))
        elif self.save_bodies and kind:
            rec["resp_file"] = self._save(rid, "resp.txt", bytes(resp_buf))
        with self._lock:
            self.records.append(rec)
            self.active.pop(rid, None)
            self.pending -= 1
        h.close_connection = True
