"""Real model integration test.

Run manually with: uv run python scripts/test_real_model.py

This script loads a small MLX model and verifies the full pipeline:
1. Engine.load() — model + tokenizer + BatchGenerator
2. Engine.start() — step loop
3. generate_stream() — per-request detokenizer + output queue
4. OpenAI SSE formatting

Uses a small 4-bit quantized model (~2GB download on first run).
"""

import asyncio
import sys
import time

# Test model — small quantized model for Apple Silicon
TEST_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


async def main():
    from python.yunshu_engine.engine import Engine, EngineConfig
    from python.yunshu_gateway.streaming import (
        format_openai_chunk,
        format_openai_done,
    )

    print(f"=== Yunshu Real Model Integration Test ===")
    print(f"Model: {TEST_MODEL}")
    print()

    # 1. Create and load engine
    print("[1/4] Loading model...")
    engine = Engine(EngineConfig(
        completion_batch_size=8,
        prefill_batch_size=4,
        prefill_step_size=1024,
    ))
    t0 = time.time()
    engine.load(TEST_MODEL)
    print(f"      Loaded in {time.time() - t0:.1f}s")

    # 2. Start step loop
    print("[2/4] Starting step loop...")
    await engine.start()
    print("      Step loop running")

    # 3. Generate streaming response
    print("[3/4] Generating response...")
    messages = [
        {"role": "user", "content": "What is 2+2? Answer briefly."},
    ]

    t0 = time.time()
    full_text = ""
    token_count = 0

    async for output in engine.generate_stream(
        prompt=messages,
        max_tokens=50,
        temperature=0.0,
    ):
        token_count += 1
        chunk = format_openai_chunk(
            completion_id="test-123",
            model=TEST_MODEL,
            delta_content=output.token_text,
            finish_reason=output.finish_reason,
        )
        # Print token text (simulating SSE output)
        sys.stdout.write(output.token_text)
        sys.stdout.flush()
        full_text += output.token_text

        if output.finish_reason:
            print(f"\n      Finish: {output.finish_reason}")
            break

    done = format_openai_done()
    print(f"\n      Generated {token_count} tokens in {time.time() - t0:.1f}s")

    # 4. Non-streaming test
    print("[4/4] Non-streaming generate...")
    t0 = time.time()
    state = await engine.generate(
        prompt=[{"role": "user", "content": "Say hello in French."}],
        max_tokens=20,
        temperature=0.0,
    )
    print(f"      Response: {state.generated_text!r}")
    print(f"      Finish: {state.finish_reason}, tokens: {state.completion_token_count}")
    print(f"      Time: {time.time() - t0:.1f}s")

    # Cleanup
    await engine.stop()
    print()
    print("=== All tests passed ===")


if __name__ == "__main__":
    asyncio.run(main())
