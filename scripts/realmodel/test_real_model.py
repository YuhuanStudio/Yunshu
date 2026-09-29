"""Real model smoke test for the default serving path (BatchedEngine).

Run manually with: PYTHONPATH=. uv run python scripts/realmodel/test_real_model.py [model]

Loads a small MLX model and checks:
1. BatchedEngine.start() — model + tokenizer load
2. stream_chat() — streaming deltas, formatted as OpenAI SSE chunks
3. chat() — non-streaming completion

Default model is a small 4-bit checkpoint (~300 MB download on first run).
"""

import asyncio
import sys
import time

TEST_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


async def main(model: str) -> int:
    from yunshu_engine.batched_engine import BatchedEngine
    from yunshu_gateway.streaming import format_openai_chunk, format_openai_done

    print("=== Yunshu Real Model Smoke Test ===")
    print(f"Model: {model}\n")

    print("[1/3] Loading model...")
    engine = BatchedEngine(model)
    t0 = time.time()
    await engine.start()
    print(f"      Loaded in {time.time() - t0:.1f}s")

    print("[2/3] Streaming chat...")
    t0 = time.time()
    first = None
    text = ""
    finish = None
    async for out in engine.stream_chat(
        messages=[{"role": "user", "content": "What is 2+2? Answer briefly."}],
        max_tokens=50,
        temperature=0.0,
    ):
        if out.new_text:
            first = first or time.time() - t0
            format_openai_chunk(
                completion_id="smoke",
                model=model,
                delta_content=out.new_text,
                finish_reason=out.finish_reason,
            )
            sys.stdout.write(out.new_text)
            sys.stdout.flush()
            text += out.new_text
        if out.finished:
            finish = out.finish_reason
            break
    format_openai_done()
    print(
        f"\n      finish={finish} first_token={first or 0:.3f}s total={time.time() - t0:.1f}s"
    )

    print("[3/3] Non-streaming chat...")
    t0 = time.time()
    out = await engine.chat(
        messages=[{"role": "user", "content": "Say hello in French."}],
        max_tokens=20,
        temperature=0.0,
    )
    print(
        f"      {out.text!r} finish={out.finish_reason} tokens={out.completion_tokens}"
    )
    print(f"      Time: {time.time() - t0:.1f}s")

    await engine.stop()
    ok = "4" in text and bool(out.text.strip())
    print("\n=== PASS ===" if ok else "\n=== FAIL ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else TEST_MODEL)))
