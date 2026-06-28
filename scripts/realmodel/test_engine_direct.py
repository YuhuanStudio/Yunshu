"""Real model test — direct engine test without server.

Tests the full Engine pipeline: load → start → generate → stop
using the actual MLX model on disk.

Run: PYTHONPATH=. uv run python scripts/test_engine_direct.py
"""

import asyncio
import os
import time

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def find_model(pattern: str) -> str | None:
    """Find a model directory by substring match."""
    if not os.path.isdir(MODELS_DIR):
        return None
    for d in sorted(os.listdir(MODELS_DIR)):
        if pattern.lower() in d.lower():
            return os.path.join(MODELS_DIR, d)
    return None


async def test_llm():
    """Test LLM engine with real model."""
    from yunshu_engine.engine import Engine, EngineConfig

    model_path = find_model("qwen3.5") or find_model("qwen2.5")
    if not model_path:
        print("  SKIP: No LLM model found in models/")
        return

    print(f"  Model: {os.path.basename(model_path)}")

    engine = Engine(EngineConfig(
        completion_batch_size=8,
        prefill_batch_size=4,
        prefill_step_size=1024,
    ))

    # Load
    t0 = time.time()
    engine.load(model_path)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Start
    await engine.start()

    # Streaming generate
    t0 = time.time()
    full_text = ""
    token_count = 0
    async for output in engine.generate_stream(
        prompt=[{"role": "user", "content": "What is 2+2? Answer briefly."}],
        max_tokens=30,
        temperature=0.0,
    ):
        token_count += 1
        full_text += output.token_text
        if output.finish_reason:
            break

    elapsed = time.time() - t0
    print(f"  Streaming: {full_text!r}")
    print(f"  {token_count} tokens in {elapsed:.1f}s ({token_count/elapsed:.1f} tok/s)")

    # Non-streaming
    t0 = time.time()
    state = await engine.generate(
        prompt=[{"role": "user", "content": "Say hello in French."}],
        max_tokens=20,
        temperature=0.0,
    )
    elapsed = time.time() - t0
    print(f"  Non-streaming: {state.generated_text!r} ({state.completion_token_count} tokens, {elapsed:.1f}s)")

    # Stats
    stats = engine.get_stats()
    reqs = stats.get('num_requests_processed', stats.get('scheduler_num_requests_processed', 0))
    steps = stats.get('step_counter', stats.get('scheduler_step_counter', 'N/A'))
    print(f"  Stats: {reqs} requests, {steps} steps")

    await engine.stop()
    print("  OK")


async def test_tts():
    """Test TTS engine with real model."""
    model_path = find_model("tts")
    if not model_path:
        print("  SKIP: No TTS model found in models/")
        return

    print(f"  Model: {os.path.basename(model_path)}")

    from yunshu_engine.audio_engine import TTSEngine

    engine = TTSEngine(model_path)
    t0 = time.time()
    await engine.start()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    result = await engine.synthesize(
        text="Hello, welcome to Yunshu!",
        voice="Chelsie",
        instruct="Speak in a friendly and warm tone.",
    )
    elapsed = time.time() - t0

    audio_bytes = result if isinstance(result, bytes) else result.get("audio", b"")
    print(f"  Generated {len(audio_bytes)} bytes in {elapsed:.1f}s")
    await engine.stop()
    print("  OK")


async def test_asr():
    """Test ASR engine with real model."""
    model_path = find_model("asr")
    if not model_path:
        print("  SKIP: No ASR model found in models/")
        return

    print(f"  Model: {os.path.basename(model_path)}")

    import tempfile
    import wave

    import numpy as np

    from yunshu_engine.audio_engine import ASREngine

    engine = ASREngine(model_path)
    t0 = time.time()
    await engine.start()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Create a test WAV file (silent audio)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
        with wave.open(f, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            samples = np.zeros(16000, dtype=np.int16)
            wf.writeframes(samples.tobytes())

    t0 = time.time()
    result = await engine.transcribe(audio_path=tmp_path)
    elapsed = time.time() - t0
    print(f"  Transcribed: {result.get('text', '')!r} ({elapsed:.1f}s)")
    os.unlink(tmp_path)
    await engine.stop()
    print("  OK")


async def main():
    print("=== Yunshu Direct Engine Tests ===")
    print()

    print("[1/3] LLM Engine")
    await test_llm()

    print()
    print("[2/3] TTS Engine")
    await test_tts()

    print()
    print("[3/3] ASR Engine")
    await test_asr()

    print()
    print("=== All engine tests completed ===")


if __name__ == "__main__":
    asyncio.run(main())
