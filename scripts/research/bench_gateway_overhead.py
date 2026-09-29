"""Per-token cost of the serving path above the model (CPU only, no GPU).

Serves the real gateway app with a real ``VLMEngine`` whose batch runner is
replaced by a stub that emits a fixed token sequence at a fixed cadence (or as
fast as possible with ``--interval 0``). Everything above ``iter_tokens`` is the
production code: detokenizer + reasoning split, delivery to the event loop,
the chat router, SSE formatting, middleware, uvicorn. A streaming client then
measures what arrives:

- ``interval 0``: the most chunks per second the path can deliver, i.e. its
  per-token CPU cost;
- ``interval N ms``: per-token lateness vs the producer (how much the path
  adds per token when the GPU would produce one every N ms).

    python scripts/research/bench_gateway_overhead.py --tokenizer <ckpt> \\
        --interval 0 12 --tokens 512 --output runs/gateway.jsonl
"""

import argparse
import http.client
import json
import os
import socket
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class StubRunner:
    """Emits ``ids`` one by one, ``interval`` seconds apart (busy-wait accurate)."""

    def __init__(self, ids, interval):
        self.ids = ids
        self.interval = interval
        self.emitted: list[float] = []

    def iter_tokens(self, input_ids, *, max_tokens, stats=None, **_):
        self.emitted.clear()
        nxt = time.perf_counter()
        for i, tok in enumerate(self.ids[:max_tokens]):
            if self.interval:
                nxt += self.interval
                while time.perf_counter() < nxt:
                    time.sleep(min(0.0005, max(0.0, nxt - time.perf_counter())))
            if stats is not None:
                if i == 0:
                    stats.first_token_s = 0.0
                stats.generated = i + 1
            self.emitted.append(time.perf_counter())
            yield tok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument(
        "--interval",
        type=float,
        nargs="+",
        default=[0.0, 12.0],
        help="ms between tokens",
    )
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--note", default="")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    os.environ.setdefault("YUNSHU_AUTH_DISABLED", "1")
    import uvicorn
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    from yunshu_engine.vlm_engine import VLMEngine
    from yunshu_gateway.engine import set_engine
    from yunshu_gateway.main import create_app

    tok = load_tokenizer(Path(a.tokenizer))
    text = (
        "def lru_cache(capacity: int):\n    '''A thread-safe LRU cache.'''\n"
        "    import threading\n    lock = threading.Lock()\n    data = {}\n"
    ) * 64
    ids = tok.encode(text, add_special_tokens=False)
    engine = VLMEngine(a.tokenizer)
    engine._model = object()
    engine._tokenizer = tok
    engine._processor = None
    engine._config = {"model_type": "qwen3_5"}
    engine._running = True
    runner = StubRunner(ids, 0.0)
    engine._batch_runner = runner

    app = create_app()
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    while not server.started:
        time.sleep(0.05)
    set_engine(engine)
    model = engine.model_name

    a.output.parent.mkdir(parents=True, exist_ok=True)
    out = a.output.open("a")
    for interval in a.interval:
        runner.interval = interval / 1000.0
        for rep in range(a.repeats):
            body = {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": a.tokens,
                "temperature": 0.0,
                "stream": True,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
            cpu0 = time.process_time()
            conn.request(
                "POST",
                "/v1/chat/completions",
                json.dumps(body),
                {"Content-Type": "application/json"},
            )
            resp = conn.getresponse()
            arrived = []
            for line in resp:
                if (
                    line.startswith(b"data: ")
                    and b'"content"' in line
                    and b'"content": ""' not in line
                ):
                    if b'"content": null' in line:
                        continue
                    arrived.append(time.perf_counter())
            conn.close()
            cpu = time.process_time() - cpu0
            em = list(runner.emitted)
            n = min(len(em), len(arrived))
            lag = [arrived[i] - em[i] for i in range(n)]
            span = arrived[-1] - arrived[0] if len(arrived) > 1 else 0
            row = {
                "note": a.note,
                "interval_ms": interval,
                "rep": rep,
                "tokens": len(em),
                "chunks": len(arrived),
                "chunks_per_s": round((len(arrived) - 1) / span, 1) if span else None,
                "cpu_ms_per_token": round(1000 * cpu / max(1, len(em)), 3),
                "lag_ms_median": round(1000 * statistics.median(lag), 3)
                if lag
                else None,
                "lag_ms_p95": round(1000 * sorted(lag)[int(0.95 * (n - 1))], 3)
                if lag
                else None,
            }
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps(row), flush=True)
    server.should_exit = True
    th.join(timeout=10)


if __name__ == "__main__":
    main()
