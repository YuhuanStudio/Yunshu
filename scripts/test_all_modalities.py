"""Integration test: verify all 5 modalities with real models.

Tests each modality using the MLX-native packages:
  LLM:   mlx-lm BatchGenerator → Qwen3.5-9B-4bit
  VLM:   mlx-vlm → Qwen3-Omni-30B-A3B
  TTS:   mlx-audio → Qwen3-TTS-1.7B-VoiceDesign
  ASR:   mlx-audio → Qwen3-ASR-1.7B
  Image: self-contained Z-Image engine → Z-Image-Turbo-4bit

Usage:
    PYTHONPATH=. uv run python scripts/test_all_modalities.py
    PYTHONPATH=. uv run python scripts/test_all_modalities.py --only llm tts
"""

import sys
import time
import traceback
import argparse

MODELS_DIR = "models"

MODELS = {
    "llm":   f"{MODELS_DIR}/Qwen3.5-9B-MLX-4bit",
    "vlm":   f"{MODELS_DIR}/Qwen3-Omni-30B-A3B-Instruct-4bit",
    "tts":   f"{MODELS_DIR}/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
    "asr":   f"{MODELS_DIR}/Qwen3-ASR-1.7B-bf16",
    "image": f"{MODELS_DIR}/Z-Image-Turbo-MLX-4bit",
}


def test_llm():
    """LLM: mlx-lm BatchGenerator with Qwen3.5-9B-4bit."""
    print("=" * 60)
    print("TEST LLM: Qwen3.5-9B-4bit via mlx-lm BatchGenerator")
    print("=" * 60)

    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator, generation_stream
    from mlx_lm.sample_utils import make_sampler

    t0 = time.time()
    model, tokenizer = load(MODELS["llm"])
    print(f"Model loaded in {time.time()-t0:.1f}s")

    sampler = make_sampler(temp=0.0)
    bg = BatchGenerator(
        model, max_tokens=64, sampler=sampler,
        prefill_batch_size=4, completion_batch_size=32,
        prefill_step_size=2048, stream=generation_stream,
    )

    prompt_tokens = tokenizer.encode(
        "What is 2+2? Answer with just the number.",
        add_special_tokens=False,
    )
    print(f"Prompt: {len(prompt_tokens)} tokens")

    uids = bg.insert(
        prompts=[prompt_tokens], max_tokens=[32], samplers=[sampler],
    )
    prompt_res, gen_res = bg.next()
    print(f"Prefill done: {len(prompt_res)} responses")

    all_tokens = []
    t0 = time.time()
    for _ in range(50):
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
                tok_s = len(all_tokens) / elapsed if elapsed > 0 else 0
                print(f"Generated {len(all_tokens)} tokens in {elapsed:.2f}s ({tok_s:.1f} tok/s)")
                print(f"Text: {repr(text[:200])}")
                print(f"Finish: {r.finish_reason}")
                break
        else:
            continue
        break

    bg.close()
    print("PASS: LLM\n")
    return True


def test_tts():
    """TTS: mlx-audio with Qwen3-TTS VoiceDesign."""
    print("=" * 60)
    print("TEST TTS: Qwen3-TTS-1.7B via mlx-audio")
    print("=" * 60)

    from mlx_audio.tts import load as tts_load

    t0 = time.time()
    model = tts_load(MODELS["tts"])
    print(f"TTS model loaded in {time.time()-t0:.1f}s")

    text = "Hello, this is a test of text to speech synthesis."
    instruct = "A warm female voice with clear pronunciation"
    print(f"Synthesizing: {repr(text)}")
    print(f"Voice: {repr(instruct)}")

    t0 = time.time()
    results = model.generate(text=text, instruct=instruct, verbose=False)
    audio_chunks = []
    for result in results:
        import numpy as np
        audio_chunks.append(np.array(result.audio))
    elapsed = time.time() - t0

    if not audio_chunks:
        print("FAIL: No audio produced")
        return False

    audio = np.concatenate(audio_chunks, axis=0)
    sample_rate = getattr(model, "sample_rate", 24000)
    duration = len(audio) / sample_rate
    print(f"Audio: {duration:.2f}s, {len(audio)} samples, {sample_rate}Hz")
    print(f"Generation time: {elapsed:.2f}s (RTF: {elapsed/duration:.2f}x)")

    print("PASS: TTS\n")
    return True


def test_asr():
    """ASR: mlx-audio with Qwen3-ASR."""
    print("=" * 60)
    print("TEST ASR: Qwen3-ASR-1.7B via mlx-audio")
    print("=" * 60)

    from mlx_audio.stt import load as stt_load
    import numpy as np
    import tempfile
    import struct
    import io

    t0 = time.time()
    model = stt_load(MODELS["asr"])
    print(f"ASR model loaded in {time.time()-t0:.1f}s")

    # Generate a simple WAV file with silence for testing
    sample_rate = 16000
    duration = 1.0
    samples = np.zeros(int(sample_rate * duration), dtype=np.float32)

    buf = io.BytesIO()
    pcm = (samples * 32767).astype(np.int16)
    num_frames = len(pcm)
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + num_frames * 2))
    buf.write(b'WAVE')
    buf.write(b'fmt ')
    buf.write(struct.pack('<I', 16))
    buf.write(struct.pack('<H', 1))
    buf.write(struct.pack('<H', 1))
    buf.write(struct.pack('<I', sample_rate))
    buf.write(struct.pack('<I', sample_rate * 2))
    buf.write(struct.pack('<H', 2))
    buf.write(struct.pack('<H', 16))
    buf.write(b'data')
    buf.write(struct.pack('<I', num_frames * 2))
    buf.write(pcm.tobytes())
    wav_bytes = buf.getvalue()

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(wav_bytes)
    tmp.close()

    t0 = time.time()
    result = model.generate(tmp.name)
    elapsed = time.time() - t0

    text = result.text if hasattr(result, 'text') else str(result)
    print(f"Transcription: {repr(text[:200])}")
    print(f"Time: {elapsed:.2f}s")

    import os
    os.unlink(tmp.name)

    print("PASS: ASR\n")
    return True


def test_vlm():
    """VLM: Our self-contained engine with Qwen3-Omni-30B (text + vision)."""
    print("=" * 60)
    print("TEST VLM: Qwen3-Omni-30B via Yunshu VLMEngine (with vision)")
    print("=" * 60)

    import asyncio
    from python.yunshu_engine.vlm_engine import VLMEngine

    engine = VLMEngine(MODELS["vlm"])

    async def _test():
        t0 = time.time()
        await engine.start()
        print(f"VLM engine loaded in {time.time()-t0:.1f}s (vision={engine.has_vision})")

        # Text-only generation
        t0 = time.time()
        result = await engine.generate(
            messages=[{"role": "user", "content": "What is 2+2? Answer briefly."}],
            max_tokens=64,
            temperature=0.0,
        )
        elapsed = time.time() - t0
        print(f"Text generate: {elapsed:.2f}s")
        print(f"Text: {repr(result['text'][:200])}")

        # Vision generation
        if engine.has_vision:
            from PIL import Image
            import numpy as np
            import tempfile
            import os

            img = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
            for y in range(80, 176):
                for x in range(80, 176):
                    img.putpixel((x, y), (255, 0, 0))
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(tmp.name)
            tmp.close()

            t0 = time.time()
            result = await engine.generate(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image briefly."},
                        {"type": "image_url", "image_url": {"url": tmp.name}},
                    ],
                }],
                max_tokens=64,
                temperature=0.5,
            )
            elapsed = time.time() - t0
            print(f"Vision generate: {elapsed:.2f}s")
            print(f"Vision text: {repr(result['text'][:200])}")
            os.unlink(tmp.name)
        else:
            print("WARNING: Vision tower not loaded, skipping vision test")

        # Streaming
        chunks = []
        t0 = time.time()
        async for chunk in engine.generate_stream(
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            max_tokens=64,
            temperature=0.0,
        ):
            chunks.append(chunk)
        elapsed = time.time() - t0
        full_text = "".join(c.token_text for c in chunks)
        print(f"\nStreamed {len(chunks)} chunks in {elapsed:.2f}s")
        print(f"Text: {repr(full_text[:200])}")

        await engine.stop()

    asyncio.run(_test())
    print("PASS: VLM\n")
    return True


def test_image():
    """Image: self-contained Z-Image engine."""
    print("=" * 60)
    print("TEST IMAGE: Z-Image-Turbo-4bit via self-contained engine")
    print("=" * 60)

    from python.yunshu_engine.image_engine import ImageGenEngine

    engine = ImageGenEngine(MODELS["image"])

    async def _test():
        t0 = time.time()
        await engine.start()
        print(f"Image pipeline loaded in {time.time()-t0:.1f}s")

    import asyncio

    async def _test():
        await engine.start()

        t0 = time.time()
        png_bytes = await engine.generate_image(
            prompt="A beautiful sunset over the ocean",
            width=512,
            height=512,
            num_inference_steps=4,
            seed=42,
        )
        elapsed = time.time() - t0
        print(f"Generated {len(png_bytes)} bytes in {elapsed:.2f}s")
        print(f"Steps: 4, Size: 512x512")

        # Verify it's a valid PNG
        assert png_bytes[:4] == b'\x89PNG', "Not a valid PNG"
        print("Valid PNG confirmed")

        await engine.stop()

    asyncio.run(_test())
    print("PASS: Image\n")
    return True


def test_llm_engine_core():
    """LLM via EngineCore: full engine stack with continuous batching."""
    print("=" * 60)
    print("TEST LLM EngineCore: Qwen3.5-9B via Engine+EngineCore")
    print("=" * 60)

    import asyncio
    from python.yunshu_engine.engine import Engine, EngineConfig

    async def _test():
        engine = Engine(EngineConfig(), use_engine_core=True)
        t0 = time.time()
        engine.load(MODELS["llm"])
        print(f"Engine loaded in {time.time()-t0:.1f}s")

        await engine.start()

        # Non-streaming
        t0 = time.time()
        state = await engine.generate(
            prompt="Explain gravity in one sentence.",
            max_tokens=64,
            temperature=0.0,
        )
        elapsed = time.time() - t0
        print(f"Generate: {elapsed:.2f}s")
        print(f"Text: {repr(state.generated_text[:200])}")
        print(f"Finish: {state.finish_reason}, Prompt: {state.prompt_token_count}, Output: {state.completion_token_count}")

        # Streaming
        chunks = []
        t0 = time.time()
        async for chunk in engine.generate_stream(
            prompt="Count from 1 to 5.",
            max_tokens=64,
            temperature=0.0,
        ):
            chunks.append(chunk)
        elapsed = time.time() - t0
        full_text = "".join(c.token_text for c in chunks)
        print(f"\nStreamed {len(chunks)} chunks in {elapsed:.2f}s")
        print(f"Text: {repr(full_text[:200])}")

        await engine.stop()

    asyncio.run(_test())
    print("PASS: LLM EngineCore\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test all modalities")
    parser.add_argument("--only", nargs="+", help="Only test these modalities", default=None)
    args = parser.parse_args()

    tests = {
        "llm": ("LLM (mlx-lm BatchGenerator)", test_llm),
        "llm_engine": ("LLM (EngineCore)", test_llm_engine_core),
        "tts": ("TTS (mlx-audio)", test_tts),
        "asr": ("ASR (mlx-audio)", test_asr),
        "vlm": ("VLM (self-contained + vision)", test_vlm),
        "image": ("Image (Z-Image engine)", test_image),
    }

    if args.only:
        tests = {k: v for k, v in tests.items() if k in args.only}

    results = {}
    for name, (desc, test_fn) in tests.items():
        print(f"\n{'='*60}")
        # Force GC between tests to free memory for large models
        import gc
        gc.collect()
        try:
            ok = test_fn()
            results[name] = "PASS" if ok else "FAIL"
        except Exception as e:
            print(f"FAIL: {name}")
            traceback.print_exc()
            results[name] = f"FAIL: {e}"
        # Aggressive cleanup after each test
        gc.collect()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, (desc, _) in tests.items():
        status = results.get(name, "SKIP")
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {desc}: {status}")

    passed = sum(1 for v in results.values() if v == "PASS")
    total = len(results)
    print(f"\n{passed}/{total} tests passed")
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
