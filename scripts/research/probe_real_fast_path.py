"""Small local-model fast-path baseline; not an HTTP or omni validation."""

import asyncio
import hashlib
import json
import os
import platform
import time
from pathlib import Path

from yunshu_engine.batched_engine import BatchedEngine


async def main():
    if (
        os.environ.get("YUNSHU_ENGINE_LOOP") == "1"
        or os.environ.get("YUNSHU_SPEC_UNVERIFIED") == "mlxvlm_mtp"
    ):
        raise SystemExit("Run with default fast path (no ENGINE_LOOP or MTP override)")
    path = Path("models/Qwen2.5-3B-Instruct-4bit")
    engine = BatchedEngine(str(path))
    results = []
    fast_calls = 0
    original = engine._stream_generate_fast

    async def traced(*args, **kwargs):
        nonlocal fast_calls
        fast_calls += 1
        async for item in original(*args, **kwargs):
            yield item

    engine._stream_generate_fast = traced
    start = time.monotonic()
    try:
        await engine.start()
        load_s = time.monotonic() - start
        messages = [{"role": "user", "content": "Reply with exactly: hello"}]
        for run in range(2):
            chunks = []
            t = time.monotonic()
            first = None
            async for out in engine.stream_chat(messages, max_tokens=16, temperature=0):
                if out.new_text and first is None:
                    first = time.monotonic() - t
                chunks.append(out)
            results.append(
                dict(
                    run=run,
                    first_text_s=first,
                    total_s=time.monotonic() - t,
                    chunks=len(chunks),
                    text="".join(c.new_text for c in chunks),
                    finish_reason=chunks[-1].finish_reason,
                    completion_tokens=chunks[-1].completion_tokens,
                )
            )
        print(
            json.dumps(
                dict(
                    model=str(path),
                    config_sha256=hashlib.sha256(
                        (path / "config.json").read_bytes()
                    ).hexdigest(),
                    python=platform.python_version(),
                    load_s=load_s,
                    engine_loop=engine._should_use_engine_loop(None),
                    fast_path_calls=fast_calls,
                    runs=results,
                ),
                indent=2,
            )
        )
    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
