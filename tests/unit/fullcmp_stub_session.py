"""Session stub for the fullcmp e2e cell types (conc, follow-up, idle follow-up, restart).

Extends tests/unit/fullcmp_stub_server.py by composition: the base module supplies the engine
load lines, the request/round log lines, the token maths and the GET routes; this module adds
  * a prefix cache that persists under the server's HOME (a relaunched stub on the same HOME serves
    prefix hits, so cached_tokens > 0 after a restart; a wiped HOME gives 0);
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
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

UNIT_DIR = Path(__file__).resolve().parent
if str(UNIT_DIR) not in sys.path:
    sys.path.insert(0, str(UNIT_DIR))

import fullcmp_stub_server as base  # noqa: E402

# The prefix cache is the server's disk tier: it lives under the server's HOME, exactly like a real
# engine's SSD tier (tfbench gives each server its own HOME and wipes it between cells). A relaunch
# on the same HOME (tfbench restart phase) finds the cache; a relaunch on a wiped HOME does not.
CACHE_SUBDIR = "fullcmp-stub-cache"
PREFILL_FLOOR_S = 0.02
CACHE_FILE = "prefix-cache.json"

# Usage shape each engine's final usage object carries (what tfbench's cached_tokens() reads):
#   details      usage.prompt_tokens_details.cached_tokens = count
#   unavailable  no cache count at all (tfbench records cached=None plus the reason)
# The table is tests/unit/fullcmp_usage_shapes.json, so a GPU smoke's usage_sample can update it
# (keys starting with "_" are notes). tf-1.2.1 is listed as details: tfbench reads that shape for
# every TensorFold build; not verified on the wire.
USAGE_SHAPES_PATH = UNIT_DIR / "fullcmp_usage_shapes.json"


def load_usage_shapes(path: Path = USAGE_SHAPES_PATH) -> dict:
    raw = json.loads(Path(path).read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


USAGE_SHAPE = load_usage_shapes()


class SessionServer(ThreadingHTTPServer):
    """The default listen backlog (5) resets connections when a mixed cell sends 8 long requests at
    once; a real engine accepts them, so the stub does too."""

    request_queue_size = 64
    daemon_threads = True


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
            prompt_tokens = count_prompt_tokens(prompt)
            with state.lock:
                state.round += 1
                n = state.round
                hit = min(state.cache.hit_words(prompt), max(prompt_tokens - 1, 0))
                state.cache.add(prompt)
            state.log(
                base.request_lines(
                    state.engine, n, len(emitted), drafted=4, accepted=3, prompt_tokens=prompt_tokens
                )
            )
            # a stub answers instantly: a floor on prefill time keeps TTFT measurable (tfbench rounds
            # ttft_s to ms, so a 0.0 TTFT is a fixture artefact, not a measurement)
            time.sleep(PREFILL_FLOOR_S)
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
                # per-request acceptance (the engaged evidence for yunshu): drafted 4, accepted 3
                "x_yunshu": {"speculative": {"rounds": 2, "drafted": 4, "accepted": 3}},
            }
            self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    return Handler


def count_prompt_tokens(prompt: str) -> int:
    """Prompt tokens the stub reports: one per word. A variant module replaces this (the mixed
    variant counts long prompts without a tokenizer)."""
    return len(prompt.split())


def cache_dir_for_home(home: str | Path) -> Path:
    """The disk tier of a server whose HOME is `home` (what the real engines use for theirs)."""
    return Path(home) / CACHE_SUBDIR


def start(engine, model, log_path, port=0, version="stub", cache_dir=None):
    """In-process session stub on 127.0.0.1 (ephemeral port by default). Returns (server, port, state).
    Stop it with server.shutdown(); a relaunch with the same cache_dir is a restart."""
    state = SessionState(engine, model, Path(log_path), version, cache_dir)
    server = SessionServer(("127.0.0.1", port), make_session_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1], state


def shim_dir(root: Path, module: str = "fullcmp_stub_session") -> Path:
    """A directory tfbench accepts as TFB_STUB_SERVER: it holds fullcmp_stub_server.py, which runs
    <module>.main() (tfbench launches <dir>/fullcmp_stub_server.py). module=fullcmp_stub_mixed
    selects the mixed-load stub variant."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "fullcmp_stub_server.py").write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(UNIT_DIR)!r})\n"
        f"import {module} as s\n"
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

    # the disk tier is under the HOME the server was launched with (tfbench sets HOME per server)
    home = os.environ.get("HOME") or None
    state = SessionState(
        a.engine,
        a.model,
        a.log,
        a.version,
        cache_dir_for_home(home) if home else None,
    )
    server = SessionServer(("127.0.0.1", a.port), make_session_handler(state))
    # exit with the parent (the test or tfbench that launched this stub), as the base stub does
    base.watch_parent(os.getppid(), max_lifetime=base.MAX_LIFETIME_S)
    print(f"stub {a.engine} listening on 127.0.0.1:{a.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
