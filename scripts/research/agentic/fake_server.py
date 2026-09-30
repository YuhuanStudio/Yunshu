"""Tiny fake model server (Chat Completions, Messages, Responses; streaming and JSON).

Used to test the recording proxy and the agent wiring without a GPU. Answers with fixed text;
when the requested model name contains "tool" and the request lists tools, it calls the first
tool with malformed (``bad``) or valid JSON arguments instead.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TEXT = ["Hello", " from", " the", " fake", " engine"]


def _tool(body):
    tools = body.get("tools") or []
    if "tool" not in str(body.get("model", "")) or not tools:
        return None
    t = tools[0]
    name = t.get("name") or (t.get("function") or {}).get("name") or "tool"
    return name, ("{bad" if "bad" in str(body.get("model")) else '{"x": 1}')


def _ev(kind, **kw):
    return (kind, {"type": kind, **kw})


class Fake:
    def __init__(self, port=0, delay=0.01):
        self.delay = delay
        self.requests: list[dict] = []
        fake = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                fake._send_json(
                    self, {"data": [{"id": "fake-model", "object": "model"}]}
                )

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                fake.requests.append(
                    dict(path=self.path, body=body, headers=dict(self.headers))
                )
                if "err" in str(body.get("model")):
                    return fake._send_json(self, {"error": {"message": "boom"}}, 500)
                fake._answer(self, body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def _send_json(self, h, obj, status=200):
        b = json.dumps(obj).encode()
        h.send_response(status)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(b)))
        h.end_headers()
        h.wfile.write(b)

    def _sse(self, h, events):
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Transfer-Encoding", "chunked")
        h.end_headers()
        for name, ev in events:
            time.sleep(self.delay)
            data = (f"event: {name}\n" if name else "") + "data: "
            data += (ev if isinstance(ev, str) else json.dumps(ev)) + "\n\n"
            raw = data.encode()
            h.wfile.write(b"%x\r\n%s\r\n" % (len(raw), raw))
            h.wfile.flush()
        h.wfile.write(b"0\r\n\r\n")

    def _chat(self, h, body, tool, stream):
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 60},
        }
        fin = "tool_calls" if tool else "stop"
        if not stream:
            msg = {"role": "assistant", "content": None if tool else "".join(TEXT)}
            if tool:
                msg["tool_calls"] = [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": tool[0], "arguments": tool[1]},
                    }
                ]
            return self._send_json(
                h,
                {
                    "choices": [{"index": 0, "message": msg, "finish_reason": fin}],
                    "usage": usage,
                },
            )

        def ch(delta, **kw):
            return (None, {"choices": [{"index": 0, "delta": delta, **kw}]})

        ev = []
        if tool:
            first = {
                "index": 0,
                "id": "c1",
                "function": {"name": tool[0], "arguments": ""},
            }
            ev.append(ch({"tool_calls": [first]}))
            more = {"index": 0, "function": {"arguments": tool[1]}}
            ev.append(ch({"tool_calls": [more]}))
        else:
            ev += [ch({"content": t}) for t in TEXT]
        ev.append(ch({}, finish_reason=fin))
        ev.append((None, {"choices": [], "usage": usage}))
        ev.append((None, "[DONE]"))
        self._sse(h, ev)

    def _messages(self, h, body, tool, stream):
        u = {
            "input_tokens": 40,
            "cache_read_input_tokens": 60,
            "cache_creation_input_tokens": 0,
            "output_tokens": 5,
        }
        stop = "tool_use" if tool else "end_turn"
        if not stream:
            content = (
                [{"type": "tool_use", "id": "t1", "name": tool[0], "input": {"x": 1}}]
                if tool
                else [{"type": "text", "text": "".join(TEXT)}]
            )
            return self._send_json(
                h,
                {
                    "type": "message",
                    "content": content,
                    "stop_reason": stop,
                    "usage": u,
                },
            )
        ev = [_ev("message_start", message={"usage": {**u, "output_tokens": 1}})]
        if tool:
            blk = {"type": "tool_use", "id": "t1", "name": tool[0], "input": {}}
            ev.append(_ev("content_block_start", index=0, content_block=blk))
            d = {"type": "input_json_delta", "partial_json": tool[1]}
            ev.append(_ev("content_block_delta", index=0, delta=d))
        else:
            blk = {"type": "text", "text": ""}
            ev.append(_ev("content_block_start", index=0, content_block=blk))
            for t in TEXT:
                d = {"type": "text_delta", "text": t}
                ev.append(_ev("content_block_delta", index=0, delta=d))
        ev.append(_ev("content_block_stop", index=0))
        ev.append(
            _ev(
                "message_delta", delta={"stop_reason": stop}, usage={"output_tokens": 5}
            )
        )
        ev.append(_ev("message_stop"))
        self._sse(h, ev)

    def _responses(self, h, body, tool, stream):
        out = (
            [
                {
                    "type": "function_call",
                    "call_id": "c1",
                    "name": tool[0],
                    "arguments": tool[1],
                }
            ]
            if tool
            else [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "".join(TEXT)}],
                }
            ]
        )
        resp = {
            "id": "resp_1",
            "status": "completed",
            "output": out,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 5,
                "input_tokens_details": {"cached_tokens": 60},
            },
        }
        if not stream:
            return self._send_json(h, resp)
        ev = [_ev("response.created", response={"id": "resp_1"})]
        if tool:
            ev.append(_ev("response.function_call_arguments.delta", delta=tool[1]))
            ev.append(_ev("response.output_item.done", item=out[0]))
        else:
            ev += [_ev("response.output_text.delta", delta=t) for t in TEXT]
        ev.append(_ev("response.completed", response=resp))
        self._sse(h, ev)

    def _answer(self, h, body):
        path = h.path.split("?")[0]
        stream = bool(body.get("stream"))
        tool = _tool(body)
        if path.endswith("/chat/completions"):
            return self._chat(h, body, tool, stream)
        if path.endswith("/messages"):
            return self._messages(h, body, tool, stream)
        if path.endswith("/responses"):
            return self._responses(h, body, tool, stream)
        self._send_json(h, {}, 404)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18990)
    a = ap.parse_args()
    f = Fake(a.port)
    print(f.url, flush=True)
    threading.Event().wait()
