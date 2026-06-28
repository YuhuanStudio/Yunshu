"""Test VLMEngine with real Qwen3-Omni-30B-A3B-Instruct-4bit model."""
import asyncio
import sys
import time

sys.path.insert(0, ".")


async def main():
    from yunshu_engine.vlm_engine import VLMEngine

    model_path = "models/Qwen3-Omni-30B-A3B-Instruct-4bit"
    engine = VLMEngine(model_path)

    print(f"[1/4] Loading {model_path}...")
    t0 = time.monotonic()
    await engine.start()
    load_time = time.monotonic() - t0
    print(f"  Loaded in {load_time:.2f}s")
    print(f"  has_vision={engine.has_vision}, is_vlm={engine._is_vlm}")

    # Test 1: Text-only generation
    print("\n[2/4] Text-only generation...")
    t0 = time.monotonic()
    result = await engine.generate(
        messages=[{"role": "user", "content": "What is 2+3? Answer with just the number."}],
        max_tokens=32,
        temperature=0.0,
    )
    gen_time = time.monotonic() - t0
    print(f"  Generated in {gen_time:.2f}s")
    print(f"  Result: {result}")
    assert result["finish_reason"] == "stop"
    assert "5" in result["text"], f"Expected '5' in response, got: {result['text']}"

    # Test 2: Streaming text generation
    print("\n[3/4] Streaming text generation...")
    t0 = time.monotonic()
    tokens = []
    async for output in engine.generate_stream(
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        max_tokens=32,
        temperature=0.0,
    ):
        tokens.append(output)
        if output.finish_reason:
            break
    stream_time = time.monotonic() - t0
    text = "".join(t.token_text for t in tokens if t.token_text)
    print(f"  Streamed in {stream_time:.2f}s, {len(tokens)} tokens")
    print(f"  Text: {text}")

    # Test 3: Vision (using the cat image we generated earlier)
    import os
    img_path = "test_output_cat.png"
    if os.path.exists(img_path):
        print(f"\n[4/4] Vision generation with {img_path}...")
        t0 = time.monotonic()
        result = await engine.generate(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one sentence."},
                    {"type": "image_url", "image_url": {"url": img_path}},
                ],
            }],
            max_tokens=64,
            temperature=0.0,
        )
        vision_time = time.monotonic() - t0
        print(f"  Vision done in {vision_time:.2f}s")
        print(f"  Result: {result}")
    else:
        print(f"\n[4/4] FAIL: Skipping vision test — {img_path} not found")
        print("  Generate one first: uv run python scripts/test_image_engine.py")
        print(f"  This is NOT a full pass — vision validation requires {img_path}")
        # Exit non-zero so CI / validation reports do not mistake this for a full pass
        import sys
        sys.exit(2)

    await engine.stop()
    print("\nAll tests PASSED!")


if __name__ == "__main__":
    asyncio.run(main())
