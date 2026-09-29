"""Where does single-request decode time go between the runner and the client?

Runs the same prompts (greedy, no thinking) through three layers and reports
decode tok/s (tokens after the first / time after the first) for each:

- ``runner``: ``VLMBatchRunner.iter_tokens`` on the prompt ids, consumed on a
  plain thread (what the engine's consumer thread sees);
- ``engine``: ``VLMEngine.generate_stream`` (detokenizer, reasoning split,
  delivery to the event loop), no HTTP;
- ``http``: a running server's ``/v1/chat/completions`` stream (``--url``).

``runner`` also records every runner step on the MLX thread (wall time per
``gen.next()`` and tokens it produced), so step time vs per-token delivery
overhead can be separated.

    python scripts/research/probe_server_path.py --model <ckpt> --layers runner engine \\
        --repeats 3 --output runs/server-path.jsonl
    python scripts/research/probe_server_path.py --url http://127.0.0.1:18990 \\
        --layers http --repeats 3 --output runs/server-path.jsonl
"""

import argparse
import asyncio
import http.client
import json
import statistics
import sys
import threading
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

PROMPTS = {
    "code": "Write a Python module implementing a thread-safe LRU cache class with "
    "get, put, delete and resize methods, full docstrings and type hints, then "
    "pytest tests for every method.",
    "prose": "Write a long, vivid short story about a lighthouse keeper who finds a "
    "message in a bottle during a winter storm.",
}


def filler(n_tokens: int) -> str:
    if n_tokens <= 0:
        return ""
    words = [
        f"Record {i}: the sensor reported nominal values." for i in range(n_tokens // 9)
    ]
    return "Background log (ignore):\n" + "\n".join(words) + "\n\n"


def messages_for(task: str, context: int) -> list[dict]:
    return [{"role": "user", "content": filler(context) + PROMPTS[task]}]


def rate(times: list[float]) -> float | None:
    if len(times) < 2:
        return None
    return round((len(times) - 1) / (times[-1] - times[0]), 2)


def run_http(url: str, model: str, msgs, max_tokens: int) -> dict:
    u = urllib.parse.urlparse(url)
    body = {
        "model": model,
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream_options": {"include_usage": True},
    }
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=3600)
    t0 = time.perf_counter()
    conn.request(
        "POST",
        "/v1/chat/completions",
        json.dumps(body),
        {"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    times, text, completion = [], [], None
    for line in resp:
        if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
            continue
        ev = json.loads(line[6:])
        if ev.get("usage"):
            completion = ev["usage"].get("completion_tokens")
        for ch in ev.get("choices") or []:
            c = (ch.get("delta") or {}).get("content")
            if c:
                times.append(time.perf_counter())
                text.append(c)
    conn.close()
    span = times[-1] - times[0] if len(times) > 1 else 0.0
    return {
        "ttft_s": round(times[0] - t0, 3) if times else None,
        "chunks": len(times),
        "tokens": completion,
        # Tokens after the first over the time after the first content chunk
        # (usage counts tokens; a chunk can carry several or none).
        "decode_tps": round((completion - 1) / span, 2)
        if completion and span
        else None,
        "decode_chunk_rate": rate(times),
        "wall_s": round(time.perf_counter() - t0, 3),
        "text": "".join(text),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", help="checkpoint dir (runner / engine layers)")
    ap.add_argument("--url", help="server URL (http layer)")
    ap.add_argument("--served-name", default="Qwen3.8-27B")
    ap.add_argument("--layers", nargs="+", default=["runner", "engine"])
    ap.add_argument("--tasks", nargs="+", default=["code", "prose"])
    ap.add_argument("--contexts", type=int, nargs="+", default=[0])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--note", default="")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    out = a.output.open("a")

    def emit(row):
        row = {"note": a.note, **row}
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps(row), flush=True)

    engine = None
    loop = None
    if any(layer in ("runner", "engine") for layer in a.layers):
        from yunshu_engine.vlm_engine import VLMEngine

        engine = VLMEngine(a.model)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(engine.start())
        runner = engine._batch_runner
        steps: list = []
        orig = runner._step_generator

        def timed(group):
            t = time.perf_counter()
            before = sum(j.stats.generated for j in group.jobs.values())
            orig(group)
            after = sum(j.stats.generated for j in group.jobs.values())
            steps.append((t, time.perf_counter() - t, after - before))

        runner._step_generator = timed

    for ctx in a.contexts:
        for task in a.tasks:
            msgs = messages_for(task, ctx)
            for rep in range(a.repeats):
                for layer in a.layers:
                    if layer == "http":
                        r = run_http(a.url, a.served_name, msgs, a.max_tokens)
                        r["text_head"] = r.pop("text")[:80]
                        emit(
                            {
                                "layer": "http",
                                "task": task,
                                "context": ctx,
                                "rep": rep,
                                **r,
                            }
                        )
                        continue
                    if layer == "runner":
                        ids, pkw, salt = engine._executor.submit(
                            engine._runner_input, msgs, [], [], False, {}
                        ).result()
                        times: list[float] = []
                        steps.clear()
                        t0 = time.perf_counter()

                        def consume():
                            for _tok in runner.iter_tokens(
                                ids,
                                max_tokens=a.max_tokens,
                                temperature=0.0,
                                prompt_kwargs=pkw,
                                apc_semantic_hash=salt,
                            ):
                                times.append(time.perf_counter())

                        th = threading.Thread(target=consume)
                        th.start()
                        th.join()
                        dec = [s for s in steps if s[0] >= times[0]] if times else []
                        step_s = sum(s[1] for s in dec)
                        span = times[-1] - times[0] if len(times) > 1 else 0.0
                        emit(
                            {
                                "layer": "runner",
                                "task": task,
                                "context": ctx,
                                "rep": rep,
                                "prompt_tokens": len(ids),
                                "tokens": len(times),
                                "ttft_s": round(times[0] - t0, 3) if times else None,
                                "decode_tps": rate(times),
                                "decode_steps": len(dec),
                                "tokens_per_step": round(
                                    sum(s[2] for s in dec) / max(1, len(dec)), 2
                                ),
                                "step_ms_mean": round(
                                    1000 * statistics.mean(s[1] for s in dec), 2
                                )
                                if dec
                                else None,
                                "step_share": round(step_s / span, 3) if span else None,
                            }
                        )
                        continue
                    if layer == "engine":

                        async def go(msgs=msgs):
                            times, texts = [], []
                            t0 = time.perf_counter()
                            async for o in engine.generate_stream(
                                messages=msgs,
                                max_tokens=a.max_tokens,
                                temperature=0.0,
                                enable_thinking=False,
                            ):
                                if o.new_text:
                                    times.append(time.perf_counter())
                                    texts.append(o.new_text)
                            return t0, times, texts

                        t0, times, texts = loop.run_until_complete(go())
                        emit(
                            {
                                "layer": "engine",
                                "task": task,
                                "context": ctx,
                                "rep": rep,
                                "chunks": len(times),
                                "ttft_s": round(times[0] - t0, 3) if times else None,
                                "decode_chunk_rate": rate(times),
                                "text_head": "".join(texts)[:80],
                            }
                        )
    if engine is not None:
        loop.run_until_complete(engine.stop())


if __name__ == "__main__":
    main()
