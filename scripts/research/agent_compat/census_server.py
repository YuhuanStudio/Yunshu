"""Scripted, recording mock model server for the agent-compat census (no GPU).

Records EVERY request (method, path, query, headers, body) to a JSONL file, answers a small
script of tool calls so a real coding agent walks through a session, and returns 404 for paths
it does not know, so the census sees exactly which endpoints an agent touches.

The script is a list of steps popped one per *main-loop* turn (a request that lists tools).
A step is {"tool": <name or list of candidate names>, "input": {...}} or {"text": "..."}.
An Anthropic request that declares a server web_search / web_fetch tool (Claude Code's WebSearch
tool makes exactly such a sub-request) gets spec-shaped server_tool_use + *_tool_result blocks
so the client's parser is exercised.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _ev(name, **kw):
    return (name, {"type": name, **kw})


class Census:
    def __init__(self, log: Path, script: list[dict], port=0, model="census-model"):
        self.log = log
        self.script = list(script)
        self.model = model
        self.lock = threading.Lock()
        self.log.parent.mkdir(parents=True, exist_ok=True)
        srv = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _body(self):
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                try:
                    return json.loads(raw) if raw else None
                except ValueError:
                    return {"_raw": raw[:2000].decode("utf-8", "replace")}

            def do_GET(self):
                srv.handle(self, "GET", None)

            def do_DELETE(self):
                srv.handle(self, "DELETE", None)

            def do_POST(self):
                srv.handle(self, "POST", self._body())

            def do_PUT(self):
                srv.handle(self, "PUT", self._body())

        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def record(self, h, method, body, status, note=""):
        rec = dict(
            t=time.time(),
            method=method,
            path=h.path,
            headers=dict(h.headers.items()),
            body=body,
            status=status,
            note=note,
        )
        with self.lock, self.log.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def send_json(self, h, obj, status=200, headers=None):
        b = json.dumps(obj).encode()
        h.send_response(status)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(b)))
        for k, v in (headers or {}).items():
            h.send_header(k, v)
        h.end_headers()
        h.wfile.write(b)

    def sse(self, h, events):
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Transfer-Encoding", "chunked")
        h.end_headers()
        for name, ev in events:
            data = (f"event: {name}\n" if name else "") + "data: "
            data += (ev if isinstance(ev, str) else json.dumps(ev)) + "\n\n"
            raw = data.encode()
            h.wfile.write(b"%x\r\n%s\r\n" % (len(raw), raw))
            h.wfile.flush()
        h.wfile.write(b"0\r\n\r\n")

    def handle(self, h, method, body):
        path = h.path.split("?")[0].rstrip("/")
        status, note = 200, ""
        try:
            if method == "GET" and path.endswith("/models"):
                m = {
                    "id": self.model,
                    "object": "model",
                    "type": "model",
                    "display_name": self.model,
                    "created_at": "2026-01-01T00:00:00Z",
                }
                self.send_json(
                    h,
                    {
                        "data": [m],
                        "object": "list",
                        "has_more": False,
                        "first_id": self.model,
                        "last_id": self.model,
                    },
                )
            elif method == "GET" and path.rsplit("/", 2)[-2] == "models":
                self.send_json(
                    h,
                    {"id": path.rsplit("/", 1)[-1], "object": "model", "type": "model"},
                )
            elif method == "POST" and path.endswith("/messages/count_tokens"):
                self.send_json(h, {"input_tokens": 1234})
            elif method == "POST" and path.endswith("/messages"):
                self.messages(h, body)
            elif method == "POST" and path.endswith("/responses"):
                self.responses(h, body)
            elif method == "POST" and path.endswith("/chat/completions"):
                self.chat(h, body)
            else:
                status = 404
                self.send_json(
                    h,
                    {
                        "error": {
                            "message": f"no route {method} {path}",
                            "type": "not_found",
                        }
                    },
                    404,
                )
        except (BrokenPipeError, ConnectionResetError):
            note = "client-disconnect"
        finally:
            self.record(h, method, body, status, note)

    def next_step(self):
        with self.lock:
            return self.script.pop(0) if self.script else {"text": "Done."}

    @staticmethod
    def _tool_names(body):
        out = []
        for t in body.get("tools") or []:
            out.append(
                t.get("name") or (t.get("function") or {}).get("name") or t.get("type")
            )
        return out

    @staticmethod
    def _pick(step, names):
        want = step["tool"] if isinstance(step["tool"], list) else [step["tool"]]
        for w in want:
            if w in names:
                return w
        return None

    # ---- Anthropic Messages ----------------------------------------------------------
    def messages(self, h, body):
        tools = body.get("tools") or []
        stream = bool(body.get("stream"))
        server_types = [
            t.get("type")
            for t in tools
            if str(t.get("type", "")).startswith(("web_search", "web_fetch"))
        ]
        usage = {
            "input_tokens": 100,
            "output_tokens": 12,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        blocks: list[dict] = []
        stop = "end_turn"
        if server_types:
            wid = "srvtoolu_01census"
            if server_types[0].startswith("web_search"):
                blocks = [
                    {
                        "type": "server_tool_use",
                        "id": wid,
                        "name": "web_search",
                        "input": {"query": "yunshu"},
                    },
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": wid,
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": "https://example.com/a",
                                "title": "Example A",
                                "encrypted_content": "enc",
                                "page_age": "1 day ago",
                            }
                        ],
                    },
                    {
                        "type": "text",
                        "text": "Found it. Example A says hi.",
                        "citations": [
                            {
                                "type": "web_search_result_location",
                                "url": "https://example.com/a",
                                "title": "Example A",
                                "encrypted_index": "idx",
                                "cited_text": "hi",
                            }
                        ],
                    },
                ]
                usage["server_tool_use"] = {"web_search_requests": 1}
            else:
                blocks = [{"type": "text", "text": "fetched"}]
        else:
            names = self._tool_names(body)
            step = self.next_step() if len(names) >= 3 else {"text": "ok"}
            nm = self._pick(step, names) if "tool" in step else None
            if nm:
                blocks = [
                    {
                        "type": "tool_use",
                        "id": f"toolu_census{int(time.time() * 1000) % 10**8}",
                        "name": nm,
                        "input": step.get("input", {}),
                    }
                ]
                stop = "tool_use"
            else:
                blocks = [
                    {
                        "type": "text",
                        "text": step.get("text") or f"(no tool {step.get('tool')})",
                    }
                ]
        msg = {
            "id": "msg_census",
            "type": "message",
            "role": "assistant",
            "model": body.get("model", self.model),
            "content": blocks,
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": usage,
        }
        if not stream:
            return self.send_json(h, msg)
        ev = [
            _ev(
                "message_start",
                message={
                    **msg,
                    "content": [],
                    "stop_reason": None,
                    "usage": {**usage, "output_tokens": 1},
                },
            )
        ]
        for i, b in enumerate(blocks):
            if b["type"] == "text":
                ev.append(
                    _ev(
                        "content_block_start",
                        index=i,
                        content_block={"type": "text", "text": ""},
                    )
                )
                ev.append(
                    _ev(
                        "content_block_delta",
                        index=i,
                        delta={"type": "text_delta", "text": b["text"]},
                    )
                )
                for c in b.get("citations") or []:
                    ev.append(
                        _ev(
                            "content_block_delta",
                            index=i,
                            delta={"type": "citations_delta", "citation": c},
                        )
                    )
            elif b["type"] in ("tool_use", "server_tool_use"):
                ev.append(
                    _ev(
                        "content_block_start", index=i, content_block={**b, "input": {}}
                    )
                )
                ev.append(
                    _ev(
                        "content_block_delta",
                        index=i,
                        delta={
                            "type": "input_json_delta",
                            "partial_json": json.dumps(b["input"]),
                        },
                    )
                )
            else:
                ev.append(_ev("content_block_start", index=i, content_block=b))
            ev.append(_ev("content_block_stop", index=i))
        ev.append(
            _ev(
                "message_delta",
                delta={"stop_reason": stop, "stop_sequence": None},
                usage={"output_tokens": usage["output_tokens"]},
            )
        )
        ev.append(_ev("message_stop"))
        self.sse(h, ev)

    # ---- OpenAI Responses ------------------------------------------------------------
    def responses(self, h, body):
        names = self._tool_names(body)
        step = self.next_step() if names else {"text": "ok"}
        out: list[dict] = []
        nm = self._pick(step, names) if "tool" in step else None
        if nm:
            out = [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": f"call_{int(time.time() * 1000) % 10**8}",
                    "name": nm,
                    "arguments": json.dumps(step.get("input", {})),
                    "status": "completed",
                }
            ]
        else:
            txt = step.get("text") or f"(no tool {step.get('tool')} in {names})"
            out = [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": txt, "annotations": []}
                    ],
                }
            ]
        usage = {
            "input_tokens": 100,
            "output_tokens": 12,
            "total_tokens": 112,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }
        resp = {
            "id": "resp_census",
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "model": body.get("model", self.model),
            "output": out,
            "usage": usage,
        }
        if not body.get("stream"):
            return self.send_json(h, resp)
        empty = {**resp, "status": "in_progress", "output": []}
        ev = [
            _ev("response.created", response=empty),
            _ev("response.in_progress", response=empty),
        ]
        for i, it in enumerate(out):
            if it["type"] == "function_call":
                added = {**it, "arguments": "", "status": "in_progress"}
            else:
                added = {**it, "content": [], "status": "in_progress"}
            ev.append(_ev("response.output_item.added", output_index=i, item=added))
            if it["type"] == "function_call":
                ev.append(
                    _ev(
                        "response.function_call_arguments.delta",
                        item_id=it["id"],
                        output_index=i,
                        delta=it["arguments"],
                    )
                )
                ev.append(
                    _ev(
                        "response.function_call_arguments.done",
                        item_id=it["id"],
                        output_index=i,
                        arguments=it["arguments"],
                    )
                )
            else:
                part = it["content"][0]
                ev.append(
                    _ev(
                        "response.content_part.added",
                        item_id=it["id"],
                        output_index=i,
                        content_index=0,
                        part={**part, "text": ""},
                    )
                )
                ev.append(
                    _ev(
                        "response.output_text.delta",
                        item_id=it["id"],
                        output_index=i,
                        content_index=0,
                        delta=part["text"],
                    )
                )
                ev.append(
                    _ev(
                        "response.output_text.done",
                        item_id=it["id"],
                        output_index=i,
                        content_index=0,
                        text=part["text"],
                    )
                )
                ev.append(
                    _ev(
                        "response.content_part.done",
                        item_id=it["id"],
                        output_index=i,
                        content_index=0,
                        part=part,
                    )
                )
            ev.append(_ev("response.output_item.done", output_index=i, item=it))
        ev.append(_ev("response.completed", response=resp))
        self.sse(h, ev)

    # ---- Chat Completions ------------------------------------------------------------
    def chat(self, h, body):
        names = self._tool_names(body)
        step = self.next_step() if len(names) >= 3 else {"text": "ok"}
        usage = {"prompt_tokens": 100, "completion_tokens": 12, "total_tokens": 112}
        call = None
        nm = self._pick(step, names) if "tool" in step else None
        if nm:
            call = {
                "id": f"call_{int(time.time() * 1000) % 10**8}",
                "type": "function",
                "function": {
                    "name": nm,
                    "arguments": json.dumps(step.get("input", {})),
                },
            }
        text = None if call else (step.get("text") or "ok")
        fin = "tool_calls" if call else "stop"
        if not body.get("stream"):
            m = {"role": "assistant", "content": text}
            if call:
                m["tool_calls"] = [call]
            return self.send_json(
                h,
                {
                    "id": "chatcmpl-census",
                    "object": "chat.completion",
                    "model": self.model,
                    "choices": [{"index": 0, "message": m, "finish_reason": fin}],
                    "usage": usage,
                },
            )

        def ch(delta, **kw):
            return (
                None,
                {
                    "id": "chatcmpl-census",
                    "object": "chat.completion.chunk",
                    "model": self.model,
                    "choices": [{"index": 0, "delta": delta, **kw}],
                },
            )

        ev = [ch({"role": "assistant", "content": ""})]
        if call:
            ev.append(
                ch(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": call["id"],
                                "type": "function",
                                "function": {
                                    "name": call["function"]["name"],
                                    "arguments": "",
                                },
                            }
                        ]
                    }
                )
            )
            ev.append(
                ch(
                    {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {
                                    "arguments": call["function"]["arguments"]
                                },
                            }
                        ]
                    }
                )
            )
        else:
            ev.append(ch({"content": text}))
        ev.append(ch({}, finish_reason=fin))
        ev.append(
            (
                None,
                {
                    "id": "chatcmpl-census",
                    "object": "chat.completion.chunk",
                    "model": self.model,
                    "choices": [],
                    "usage": usage,
                },
            )
        )
        ev.append((None, "[DONE]"))
        self.sse(h, ev)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18991)
    ap.add_argument("--log", default="/tmp/census.jsonl")
    ap.add_argument("--script", default="[]")
    a = ap.parse_args()
    s = Census(Path(a.log), json.loads(a.script), a.port)
    print(s.url, flush=True)
    threading.Event().wait()
