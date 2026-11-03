"""Test-only request-family stub: the fullcmp stub plus the reply shapes of the B2 family cells.

Subclasses the base stub (fullcmp_stub_server) and answers only the family requests itself; every
other request goes to the base handler unchanged. Family replies are what a conforming engine sends:

- tool-chat: a tool_calls delta naming a declared tool, JSON arguments, finish=tool_calls
- json-schema: content that conforms to the request's response_format schema
- thinking on (chat_template_kwargs.enable_thinking): reasoning_content deltas, then content
- images (image_url parts): accepted, prompt usage grows by 256 tokens per image and
  prompt_tokens_details.image_tokens reports them; a text-only engine (tf-*) rejects them with 400
- cjk (CJK ask in the prompt): CJK text

Faults (in-process tests only, via start(faults=...)) make one reply invalid in one named way.
Used by tests only: CPU, 127.0.0.1, no model.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import fullcmp_stub_server as base

IMAGE_TOKENS = 256
CJK_TOKENS = ["我們", "模型", "推理", "快取", "解碼", "。"]
FAULTS = {
    "bad_tool_json",  # tool arguments cut off mid-object
    "bad_json",  # json-schema reply cut off
    "bad_schema",  # json-schema reply with a wrong field type
    "no_reasoning",  # thinking on, but no reasoning_content split
    "no_image_tokens",  # image parts accepted, but no image tokens reported
}


def family_of(body: dict) -> str | None:
    """The request family of a chat body, or None for a plain decode request."""
    if body.get("tools"):
        return "tool-chat"
    rf = body.get("response_format") or {}
    if rf.get("type") == "json_schema":
        return "json-schema"
    for m in body.get("messages", []):
        c = m.get("content")
        if isinstance(c, list) and any(
            isinstance(p, dict) and p.get("type") == "image_url" for p in c
        ):
            return "images"
    if (body.get("chat_template_kwargs") or {}).get("enable_thinking") is True:
        return "thinking"
    text = base.prompt_text(body)
    if "請用" in text or "繁體中文" in text:
        return "cjk"
    return None


def image_count(body: dict) -> int:
    n = 0
    for m in body.get("messages", []):
        c = m.get("content")
        if isinstance(c, list):
            n += sum(1 for p in c if isinstance(p, dict) and p.get("type") == "image_url")
    return n


def schema_instance(schema: dict):
    """A minimal instance that conforms to a schema (the required fields; first enum value)."""
    if "enum" in schema:
        return schema["enum"][0]
    t = schema.get("type")
    if t == "object":
        return {
            k: schema_instance(v)
            for k, v in schema.get("properties", {}).items()
            if k in schema.get("required", [])
        }
    if t == "array":
        return [schema_instance(schema.get("items", {"type": "string"}))]
    if t == "integer":
        return 1
    if t == "number":
        return 1.5
    if t == "boolean":
        return True
    return "x"


def _cut(text: str) -> str:
    return text[: max(1, len(text) // 2)]


def plan(body: dict, kind: str, faults=frozenset()):
    """(deltas, finish, pieces, image_tokens) of one family reply. pieces counts completion tokens."""
    max_tokens = int(body.get("max_tokens") or 16)
    if kind == "tool-chat":
        text = base.prompt_text(body)
        name, args = (
            ("read_file", {"path": "notes/todo.md"})
            if "notes/todo.md" in text
            else ("run_shell", {"command": "ls src/"})
        )
        arguments = json.dumps(args)
        if "bad_tool_json" in faults:
            arguments = _cut(arguments)
        deltas = [
            {"tool_calls": [{"index": 0, "id": "call_0", "type": "function",
                             "function": {"name": name, "arguments": ""}}]},
            {"tool_calls": [{"index": 0, "function": {"arguments": arguments}}]},
        ]
        return deltas, "tool_calls", 2, 0
    if kind == "json-schema":
        schema = body["response_format"]["json_schema"]["schema"]
        obj = schema_instance(schema)
        if "bad_schema" in faults:
            obj["title"] = 1
        content = json.dumps(obj, ensure_ascii=False)
        if "bad_json" in faults:
            content = _cut(content)
        pieces = content.split(" ") if " " in content else [content]
        return [{"content": p} for p in pieces], "stop", len(pieces), 0
    if kind == "cjk":
        pieces = [CJK_TOKENS[i % len(CJK_TOKENS)] for i in range(max_tokens)]
        return [{"content": p} for p in pieces], "length", len(pieces), 0
    if kind == "images":
        n_img = image_count(body)
        img = 0 if "no_image_tokens" in faults else IMAGE_TOKENS * n_img
        toks = base.make_tokens(base.prompt_text(body), max_tokens)
        return [{"content": t} for t in toks], "length", len(toks), img
    # thinking on: reasoning split first, then the answer; the budget covers both (as in the engines)
    deltas = []
    reason = [] if "no_reasoning" in faults else ["先", "想", "一下"][: max(0, max_tokens - 1)]
    deltas += [{"reasoning_content": r} for r in reason]
    toks = base.make_tokens(base.prompt_text(body), max(1, max_tokens - len(reason)))
    deltas += [{"content": t} for t in toks]
    return deltas, "length", len(reason) + len(toks), 0


def make_family_handler(state, faults=frozenset()):
    Base = base.make_handler(state)

    class Handler(Base):
        def do_POST(self):
            if not self.path.startswith("/v1/chat/completions"):
                return super().do_POST()
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.loads(raw)
            kind = family_of(body)
            if kind is None:
                self.rfile = io.BytesIO(raw)  # the base handler reads its own body
                return super().do_POST()
            if kind == "images" and state.engine.startswith("tf-"):
                return self._json(400, {"error": {"message": "image input is not supported"}})
            deltas, finish, pieces, img = plan(body, kind, faults)
            with state.lock:
                state.round += 1
                n = state.round
            state.log(base.request_lines(state.engine, n, pieces, drafted=4, accepted=3))
            prompt_tokens = len(base.prompt_text(body).split()) + img
            details = {"cached_tokens": 0}
            if img:
                details["image_tokens"] = img
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": pieces,
                "total_tokens": prompt_tokens + pieces,
                "prompt_tokens_details": details,
            }
            self._sse(f"chatcmpl-stub-{n}", state.model, deltas, finish, usage)

        def _sse(self, cid, model, deltas, finish, usage):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for d in deltas:
                chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0, "delta": d}],
                }
                self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
            final = {
                "id": cid,
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                "usage": usage,
            }
            self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return Handler


def start(engine, model, log_path, port=0, version="stub", faults=frozenset()):
    """Start the family stub in a background thread. Returns (server, port, state)."""
    unknown = set(faults) - FAULTS
    if unknown:
        raise ValueError(f"unknown faults {sorted(unknown)}")
    state = base.StubState(engine, model, Path(log_path), version)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_family_handler(state, frozenset(faults)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1], state


def main(argv=None) -> int:
    """Same command line as fullcmp_stub_server (tfbench launches it with TFB_STUB_SCRIPT)."""
    ap = argparse.ArgumentParser(description="fullcmp request-family test stub")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", default="stub-model")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--version", default="stub")
    a = ap.parse_args(argv)
    state = base.StubState(a.engine, a.model, a.log, a.version)
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_family_handler(state))
    # exit with the parent (the test or tfbench that launched this stub), as the base stub does
    base.watch_parent(os.getppid(), max_lifetime=base.MAX_LIFETIME_S)
    print(f"stub {a.engine} (families) listening on 127.0.0.1:{a.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
