"""Test ImageGenEngine with real Z-Image-Turbo-MLX-4bit model."""
import asyncio
import sys
import time

sys.path.insert(0, ".")


async def main():
    from yunshu_engine.image_engine import ImageGenEngine

    model_path = "models/Z-Image-Turbo-MLX-4bit"
    engine = ImageGenEngine(model_path)

    print(f"[1/3] Loading {model_path}...")
    t0 = time.monotonic()
    await engine.start()
    load_time = time.monotonic() - t0
    print(f"  Loaded in {load_time:.2f}s")

    print("[2/3] Generating image (256x256, 4 steps)...")
    t0 = time.monotonic()
    png_bytes = await engine.generate_image(
        prompt="A cat sitting on a windowsill at sunset.",
        width=256,
        height=256,
        num_inference_steps=4,
        seed=42,
    )
    gen_time = time.monotonic() - t0
    print(f"  Generated in {gen_time:.2f}s, {len(png_bytes)} bytes")

    # Verify PNG header
    assert png_bytes[:4] == b"\x89PNG", f"Not a valid PNG: {png_bytes[:8]}"
    print(f"  PNG header valid: {png_bytes[:8]}")

    # Save for inspection
    out_path = "test_output_cat.png"
    with open(out_path, "wb") as f:
        f.write(png_bytes)
    print(f"  Saved to {out_path}")

    print("[3/3] Streaming image gen (256x256, 4 steps)...")
    t0 = time.monotonic()
    chunks = []
    async for chunk in engine.generate_image_stream(
        prompt="A mountain landscape at dawn.",
        width=256,
        height=256,
        num_inference_steps=4,
        seed=123,
    ):
        chunks.append(chunk)
        print(f"  Step {chunk['step']}/{chunk['total_steps']}: progress={chunk['progress']:.0%}")
    stream_time = time.monotonic() - t0
    final = chunks[-1]
    assert final["is_final"], "Last chunk should be final"
    assert final["image"] is not None, "Final image should not be None"
    assert final["image"][:4] == b"\x89PNG", "Final chunk should be valid PNG"
    print(f"  Streaming done in {stream_time:.2f}s, {len(chunks)} chunks, {len(final['image'])} bytes")

    out_path2 = "test_output_mountain.png"
    with open(out_path2, "wb") as f:
        f.write(final["image"])
    print(f"  Saved to {out_path2}")

    await engine.stop()
    print("\nAll tests PASSED!")


if __name__ == "__main__":
    asyncio.run(main())
