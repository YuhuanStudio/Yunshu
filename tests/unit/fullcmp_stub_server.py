"""Test-only OpenAI-compatible stub server for the fullcmp dry run (no model, no GPU).

Serves /v1/models, /health and POST /v1/chat/completions (stream and non-stream). Tokens are
deterministic (a function of the prompt and the position), usage counts prompt and completion
tokens, max_tokens and stop strings are honoured. Every request appends engine-style log lines to
the log file: the load/engagement line of the engine type (the expected-mode probe's high or
medium pattern from fullcmp_audit_manifest.EXPECTED_MODE_PROBE, or the round-profile line for
yunshu and TensorFold) and one per-request round line.

Used by tests only. Launched by tfbench through the TFB_STUB_SERVER hook (see tfbench.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROUND_LINE = (
    "[round-profile] engine={engine} round={n} drafts=4 accepted=3 tokens={tokens}"
)


def engine_load_lines(engine: str) -> list[str]:
    """The engine's load/engagement lines, written once at startup. Each one matches the
    high- or medium-confidence engaged-mode pattern for that engine (fullcmp_audit_manifest)."""
    if engine == "mlx-serve-26.11.1":
        return ["[mtp] sidecar loaded (mtp head ready)"]
    if engine == "kapitsa-0.2.26":
        return ["DFlash2 drafter /stub/draft: 4 layers loaded in 12 ms"]
    if engine == "omlx-0.7.3":
        return ["DFlash enabled for stub-model as a batched drafter, draft=/stub/draft"]
    if engine == "omlx-0.7.3rc1":
        return ["DFlash enabled for stub-model as a batched drafter, draft=/stub/draft"]
    if engine == "yunshu-main":
        # tfbench.engaged_spec_mode_line reads the initialized runner line
        return ["VLM batch runner: model=stub-model draft=dflash"]
    if engine.startswith("tf-"):
        return ["drafts: DFlash2 block=16 (stub)"]
    return []


def request_lines(
    engine: str, n: int, tokens: int, drafted: int, accepted: int
) -> list[str]:
    """Per-request engagement lines: what the engine prints when a round drafted and accepted."""
    if engine == "mlx-serve-26.11.1":
        return [f"[mtp-round] off0=0 t1=1 m=1 drafts={drafted} accepted={accepted}"]
    if engine == "kapitsa-0.2.26":
        return [f"spec: {accepted}/{drafted} accepted/drafted over 1 cycles"]
    if engine.startswith("omlx"):
        return [f"DFlash round drafted={drafted} accepted={accepted}"]
    return [ROUND_LINE.format(engine=engine, n=n, tokens=tokens)]


def make_tokens(prompt: str, n: int) -> list[str]:
    seed = zlib.crc32(prompt.encode())
    return [f"t{(seed + i * 7919) % 9973}" for i in range(n)]


def apply_stop(text: str, stops: list[str]) -> tuple[str, bool]:
    cut = None
    for s in stops:
        if s and s in text:
            idx = text.index(s)
            cut = idx if cut is None else min(cut, idx)
    if cut is None:
        return text, False
    return text[:cut], True


def prompt_text(body: dict) -> str:
    parts = []
    for m in body.get("messages", []):
        c = m.get("content", "")
        if isinstance(c, list):
            c = " ".join(x.get("text", "") for x in c if isinstance(x, dict))
        parts.append(str(c))
    return "\n".join(parts)


class StubState:
    def __init__(self, engine: str, model: str, log_path: Path, version: str = "stub"):
        self.engine = engine
        self.model = model
        self.version = version
        self.log_path = Path(log_path)
        self.lock = threading.Lock()
        self.round = 0
        # prompts this process has already prefilled: a repeat is served from cache, a first
        # sighting is a cold miss (server-reported cached_tokens, as the real engines report them)
        self.seen: set[str] = set()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as fh:
            for line in engine_load_lines(engine):
                fh.write(line + "\n")

    def log(self, lines: list[str]) -> None:
        with self.lock, self.log_path.open("a") as fh:
            for line in lines:
                fh.write(line + "\n")


def make_handler(state: StubState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # keep test output quiet
            pass

        def _json(self, code: int, obj) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.startswith("/health"):
                return self._json(200, {"status": "ok"})
            if self.path.startswith("/v1/yunshu/status"):
                return self._json(200, {"version": state.version})
            if self.path.startswith("/v1/models"):
                return self._json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": state.model,
                                "object": "model",
                                "meta": {"mtp_loaded": True},
                            }
                        ],
                    },
                )
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.startswith("/v1/chat/completions"):
                return self._json(404, {"error": "not found"})
            body = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
            )
            prompt = prompt_text(body)
            max_tokens = int(body.get("max_tokens") or 16)
            stops = body.get("stop") or []
            if isinstance(stops, str):
                stops = [stops]
            toks = make_tokens(prompt, max_tokens)
            text, stopped = apply_stop(" ".join(toks), list(stops))
            emitted = text.split(" ") if text else []
            finish = "stop" if stopped else "length"
            prompt_tokens = len(prompt.split())
            with state.lock:
                state.round += 1
                n = state.round
                cached = prompt_tokens - 1 if prompt in state.seen else 0
                state.seen.add(prompt)
            state.log(
                request_lines(state.engine, n, len(emitted), drafted=4, accepted=3)
            )
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(emitted),
                "total_tokens": prompt_tokens + len(emitted),
                "prompt_tokens_details": {"cached_tokens": max(0, cached)},
            }
            cid = f"chatcmpl-stub-{n}"
            if not body.get("stream"):
                return self._json(
                    200,
                    {
                        "id": cid,
                        "object": "chat.completion",
                        "model": state.model,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": finish,
                            }
                        ],
                        "usage": usage,
                    },
                )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for i, tok in enumerate(emitted):
                chunk = {
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "model": state.model,
                    "choices": [
                        {"index": 0, "delta": {"content": (" " if i else "") + tok}}
                    ],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            final = {
                "id": cid,
                "object": "chat.completion.chunk",
                "model": state.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                "usage": usage,
            }
            self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return Handler


def start(engine: str, model: str, log_path, port: int = 0, version: str = "stub"):
    """Start the stub in a background thread. Returns (server, port, state)."""
    state = StubState(engine, model, Path(log_path), version)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1], state


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="fullcmp test stub server")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", default="stub-model")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--version", default="stub")
    a = ap.parse_args(argv)
    state = StubState(a.engine, a.model, a.log, a.version)
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(state))
    print(f"stub {a.engine} listening on 127.0.0.1:{a.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
