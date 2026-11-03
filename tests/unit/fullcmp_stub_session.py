"""Session stub for the fullcmp e2e cell types (conc, follow-up, idle follow-up, restart).

Extends tests/unit/fullcmp_stub_server.py by composition: the base module supplies the engine
load lines, the request/round log lines, the token maths and the GET routes; this module adds
  * a prefix cache that persists in a cache directory (a relaunched stub with the same directory
    serves prefix hits, so cached_tokens > 0 after a restart and 0 when cold);
  * the per-engine usage shape table (what each engine puts in the final usage object);
  * a launcher that tfbench can run through TFB_STUB_SERVER (see shim_dir()).

Test-only. No GPU, no model. tfbench's Srv launches <TFB_STUB_SERVER>/fullcmp_stub_server.py, so
shim_dir() writes a one-line file of that name that runs this module's main().
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

UNIT_DIR = Path(__file__).resolve().parent
if str(UNIT_DIR) not in sys.path:
    sys.path.insert(0, str(UNIT_DIR))

import fullcmp_stub_server as base  # noqa: E402

CACHE_ENV = "FULLCMP_STUB_CACHE_DIR"
CACHE_FILE = "prefix-cache.json"

# Usage shape each engine's final usage object carries (what tfbench's cached_tokens() reads):
#   details      usage.prompt_tokens_details.cached_tokens = count
#   unavailable  no cache count at all (tfbench records cached=None plus the reason)
# yunshu reports the OpenAI details shape. tf-1.2.1 is listed as details too: tfbench reads that
# shape for every TensorFold build (bench_engine_matrix reads both shapes); not verified on the wire.
USAGE_SHAPE = {
    "yunshu": "details",
    "tf-1.2.1": "details",
}


class PrefixStore:
    """Prompts this cache has prefilled, persisted as JSON. A new prompt hits the longest stored
    prompt that is a word-prefix of it (the way a prefix cache serves a follow-up turn)."""

    def __init__(self, cache_dir: Path | None):
        self.path = Path(cache_dir) / CACHE_FILE if cache_dir else None
        self.prompts: list[str] = []
        if self.path and self.path.exists():
            self.prompts = json.loads(self.path.read_text())

    def hit_words(self, prompt: str) -> int:
        best = 0
        for stored in self.prompts:
            if prompt.startswith(stored):
                best = max(best, len(stored.split()))
        return best

    def add(self, prompt: str) -> None:
        if prompt in self.prompts:
            return
        self.prompts.append(prompt)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.prompts))
            tmp.replace(self.path)


class SessionState(base.StubState):
    def __init__(self, engine, model, log_path, version="stub", cache_dir=None):
        super().__init__(engine, model, log_path, version)
        self.cache = PrefixStore(cache_dir)
        self.shape = USAGE_SHAPE.get(engine, "details")


def usage_object(
    shape: str, prompt_tokens: int, completion_tokens: int, cached: int | None
):
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if shape == "details":
        usage["prompt_tokens_details"] = {"cached_tokens": int(cached or 0)}
    elif shape != "unavailable":
        raise ValueError(f"unknown usage shape {shape!r}")
    return usage


def make_session_handler(state: SessionState):
    BaseHandler = base.make_handler(state)

    class Handler(BaseHandler):
        def do_POST(self):
            if not self.path.startswith("/v1/chat/completions"):
                return self._json(404, {"error": "not found"})
            body = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
            )
            prompt = base.prompt_text(body)
            max_tokens = int(body.get("max_tokens") or 16)
            stops = body.get("stop") or []
            if isinstance(stops, str):
                stops = [stops]
            text, stopped = base.apply_stop(
                " ".join(base.make_tokens(prompt, max_tokens)), list(stops)
            )
            emitted = text.split(" ") if text else []
            finish = "stop" if stopped else "length"
            prompt_tokens = len(prompt.split())
            with state.lock:
                state.round += 1
                n = state.round
                hit = min(state.cache.hit_words(prompt), max(prompt_tokens - 1, 0))
                state.cache.add(prompt)
            state.log(
                base.request_lines(state.engine, n, len(emitted), drafted=4, accepted=3)
            )
            usage = usage_object(state.shape, prompt_tokens, len(emitted), hit)
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


def start(engine, model, log_path, port=0, version="stub", cache_dir=None):
    """In-process session stub on 127.0.0.1 (ephemeral port by default). Returns (server, port, state).
    Stop it with server.shutdown(); a relaunch with the same cache_dir is a restart."""
    state = SessionState(engine, model, Path(log_path), version, cache_dir)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_session_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1], state


def shim_dir(root: Path) -> Path:
    """A directory tfbench accepts as TFB_STUB_SERVER: it holds fullcmp_stub_server.py, which runs
    this module's main() (tfbench launches <dir>/fullcmp_stub_server.py)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "fullcmp_stub_server.py").write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(UNIT_DIR)!r})\n"
        "import fullcmp_stub_session as s\n"
        "raise SystemExit(s.main())\n"
    )
    return root


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="fullcmp session test stub")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", default="stub-model")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--version", default="stub")
    a = ap.parse_args(argv)
    import os

    state = SessionState(
        a.engine, a.model, a.log, a.version, os.environ.get(CACHE_ENV) or None
    )
    server = ThreadingHTTPServer(("127.0.0.1", a.port), make_session_handler(state))
    print(f"stub {a.engine} listening on 127.0.0.1:{a.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
