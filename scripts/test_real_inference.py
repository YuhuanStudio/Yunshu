"""Real model integration test — loads actual Qwen3.5-9B and runs inference.

Tests the full engine stack: Engine → Scheduler → BatchGenerator → token output.
This is NOT a unit test — it requires the model to be downloaded.

Usage:
    PYTHONPATH=. uv run python scripts/test_real_inference.py
"""

import sys
import time
import json
import traceback
import asyncio

MODELS_DIR = "models"
LLM_MODEL = f"{MODELS_DIR}/Qwen3.5-9B-MLX-4bit"


def test_batch_generator_direct():
    """Test 1: Direct BatchGenerator usage with real model."""
    print("=" * 60)
    print("TEST 1: Direct BatchGenerator with Qwen3.5-9B")
    print("=" * 60)

    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator, generation_stream
    from mlx_lm.sample_utils import make_sampler

    t0 = time.time()
    model, tokenizer = load(LLM_MODEL)
    print(f"Model loaded in {time.time()-t0:.1f}s")

    sampler = make_sampler(temp=0.7)
    bg = BatchGenerator(
        model, max_tokens=64, sampler=sampler,
        prefill_batch_size=4, completion_batch_size=32,
        prefill_step_size=2048, stream=generation_stream,
    )

    # Insert prompt
    prompt_tokens = tokenizer.encode("What is 2+2? Answer with just the number.", add_special_tokens=False)
    print(f"Prompt: {len(prompt_tokens)} tokens")

    uids = bg.insert(prompts=[prompt_tokens], max_tokens=[32], samplers=[sampler])
    uid = uids[0]
    print(f"Inserted uid={uid}")

    # Step 0: prefill
    prompt_res, gen_res = bg.next()
    print(f"Prefill done: {len(prompt_res)} prompt responses")

    # Generate tokens
    all_tokens = []
    t0 = time.time()
    for step in range(50):
        gen_res = bg.next_generated()
        if not gen_res:
            break
        for r in gen_res:
            all_tokens.append(r.token)
            if r.finish_reason:
                elapsed = time.time() - t0
                detok = tokenizer.detokenizer
                detok.reset()
                for t in all_tokens:
                    detok.add_token(t)
                detok.finalize()
                text = detok.last_segment
                print(f"Generated {len(all_tokens)} tokens in {elapsed:.2f}s ({len(all_tokens)/elapsed:.1f} tok/s)")
                print(f"Text: {repr(text)}")
                print(f"Finish: {r.finish_reason}")
                break
        else:
            continue
        break

    bg.close()
    print("PASS: Direct BatchGenerator\n")
    return True


async def test_engine_generate():
    """Test 2: Engine.generate() with real model."""
    print("=" * 60)
    print("TEST 2: Engine.generate() with Qwen3.5-9B")
    print("=" * 60)

    from python.yunshu_engine.engine import Engine, EngineConfig

    engine = Engine(EngineConfig())

    t0 = time.time()
    engine.load(LLM_MODEL)
    print(f"Engine loaded in {time.time()-t0:.1f}s")

    await engine.start()

    # Non-streaming generation
    t0 = time.time()
    state = await engine.generate(
        prompt="Explain gravity in one sentence.",
        max_tokens=64,
        temperature=0.7,
    )
    elapsed = time.time() - t0
    print(f"Generated in {elapsed:.2f}s")
    print(f"Text: {repr(state.generated_text[:200])}")
    print(f"Finish: {state.finish_reason}")
    print(f"Prompt tokens: {state.prompt_tokens}, Output tokens: {state.completion_token_count}")

    await engine.stop()
    print("PASS: Engine.generate()\n")
    return True


async def test_engine_stream():
    """Test 3: Streaming generation with real model."""
    print("=" * 60)
    print("TEST 3: Engine streaming with Qwen3.5-9B")
    print("=" * 60)

    from python.yunshu_engine.engine import Engine, EngineConfig

    engine = Engine(EngineConfig())
    engine.load(LLM_MODEL)
    await engine.start()

    # Streaming generation
    chunks = []
    t0 = time.time()
    async for chunk in engine.generate_stream(
        prompt="Count from 1 to 5.",
        max_tokens=64,
        temperature=0.7,
    ):
        chunks.append(chunk)
    elapsed = time.time() - t0

    full_text = "".join(c.token_text for c in chunks)
    print(f"Streamed {len(chunks)} chunks in {elapsed:.2f}s")
    print(f"Text: {repr(full_text[:200])}")

    await engine.stop()
    print("PASS: Engine streaming\n")
    return True


async def test_batched_engine():
    """Test 4: BatchedEngine with multiple concurrent requests."""
    print("=" * 60)
    print("TEST 4: BatchedEngine concurrent requests")
    print("=" * 60)

    from python.yunshu_engine.batched_engine import BatchedEngine

    engine = BatchedEngine(model_name=LLM_MODEL)
    await engine.start()

    # Submit multiple requests
    prompts = [
        "What is 1+1?",
        "What is 2+2?",
        "What is 3+3?",
    ]

    t0 = time.time()
    results = []
    for prompt in prompts:
        result = await engine.generate(
            prompt=[{"role": "user", "content": prompt}],
            max_tokens=32,
            temperature=0.0,
        )
        results.append((prompt, result.text))
    elapsed = time.time() - t0

    for prompt, text in results:
        print(f"  Q: {prompt}")
        print(f"  A: {repr(text[:100])}")

    print(f"Batch completed in {elapsed:.2f}s")

    await engine.stop()
    print("PASS: BatchedEngine\n")
    return True


async def main():
    results = {}

    # Sync test
    try:
        ok = test_batch_generator_direct()
        results["batch_generator_direct"] = "PASS" if ok else "FAIL"
    except Exception as e:
        print(f"FAIL: batch_generator_direct")
        traceback.print_exc()
        results["batch_generator_direct"] = f"FAIL: {e}"

    # Async tests
    for name, test_fn in [
        ("engine_generate", test_engine_generate),
        ("engine_stream", test_engine_stream),
        ("batched_engine", test_batched_engine),
    ]:
        try:
            ok = await test_fn()
            results[name] = "PASS" if ok else "FAIL"
        except Exception as e:
            print(f"FAIL: {name}")
            traceback.print_exc()
            results[name] = f"FAIL: {e}"

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        print(f"  {name}: {status}")

    passed = sum(1 for v in results.values() if v == "PASS")
    total = len(results)
    print(f"\n{passed}/{total} tests passed")
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
