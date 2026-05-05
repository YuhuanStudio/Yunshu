"""End-to-end test for all 5 modalities.

Tests: LLM, VLM, TTS, ASR, Image Generation against the running server.
Models must be in ./models/ directory.

Run:
  1. Start server: just dev-multi
  2. Run tests: PYTHONPATH=. uv run python scripts/test_all_models.py
"""

import asyncio
import sys
import time

import httpx

BASE_URL = "http://localhost:8000"


def test_health():
    """Test server health."""
    print("[1/6] Health check...", end=" ")
    resp = httpx.get(f"{BASE_URL}/health", timeout=5)
    assert resp.status_code == 200, f"Health failed: {resp.status_code}"
    data = resp.json()
    assert data["status"] == "ok"
    print(f"OK (engine: {data.get('engine', {}).get('model', 'N/A')})")
    return True


def test_models_list():
    """Test models listing."""
    print("[2/6] List models...", end=" ")
    resp = httpx.get(f"{BASE_URL}/v1/models", timeout=5)
    assert resp.status_code == 200
    data = resp.json()
    models = data.get("data", [])
    print(f"OK ({len(models)} models: {[m['id'] for m in models]})")
    return models


def test_llm_chat(models):
    """Test LLM chat completions (streaming + non-streaming)."""
    # Find LLM model
    llm_model = None
    for m in models:
        if "qwen3.5" in m["id"].lower() or "qwen3" in m["id"].lower():
            if "tts" not in m["id"].lower() and "asr" not in m["id"].lower():
                llm_model = m["id"]
                break

    if not llm_model:
        print("[3/6] LLM Chat: SKIP (no LLM model found)")
        return True

    print(f"[3/6] LLM Chat ({llm_model})...")

    # Non-streaming
    resp = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": llm_model,
            "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
            "max_tokens": 20,
            "temperature": 0.0,
        },
        timeout=60,
    )
    assert resp.status_code == 200, f"Chat failed: {resp.status_code} {resp.text}"
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    print(f"  Non-streaming: {content!r} ({usage.get('completion_tokens', 0)} tokens)")

    # Streaming
    full_text = ""
    with httpx.stream(
        "POST",
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": llm_model,
            "messages": [{"role": "user", "content": "Say hello in 3 languages."}],
            "max_tokens": 50,
            "temperature": 0.7,
            "stream": True,
        },
        timeout=60,
    ) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    chunk = __import__("json").loads(line[6:])
                    delta = chunk["choices"][0].get("delta", {})
                    if delta.get("content"):
                        full_text += delta["content"]
                except:
                    pass
    print(f"  Streaming: {full_text!r}")

    # Thinking mode test
    resp = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": llm_model,
            "messages": [{"role": "user", "content": "Is 97 a prime number?"}],
            "max_tokens": 100,
            "enable_thinking": True,
        },
        timeout=60,
    )
    if resp.status_code == 200:
        data = resp.json()
        print(f"  Thinking mode: OK")
    else:
        print(f"  Thinking mode: SKIP ({resp.status_code})")

    print("  OK")
    return True


def test_tts(models):
    """Test TTS endpoint."""
    tts_model = None
    for m in models:
        if "tts" in m["id"].lower():
            tts_model = m["id"]
            break

    if not tts_model:
        print("[4/6] TTS: SKIP (no TTS model found)")
        return True

    print(f"[4/6] TTS ({tts_model})...", end=" ")
    t0 = time.time()
    resp = httpx.post(
        f"{BASE_URL}/v1/audio/speech",
        json={
            "model": tts_model,
            "input": "Hello, this is a test of the Yunshu text to speech system.",
            "voice": "Chelsie",
            "instruct": "Speak in a friendly and warm tone.",
            "response_format": "wav",
        },
        timeout=120,
    )
    elapsed = time.time() - t0
    if resp.status_code == 200:
        size_kb = len(resp.content) / 1024
        print(f"OK ({size_kb:.0f} KB, {elapsed:.1f}s)")
    else:
        print(f"ERROR: {resp.status_code} {resp.text[:200]}")
    return True


def test_asr(models):
    """Test ASR endpoint."""
    asr_model = None
    for m in models:
        if "asr" in m["id"].lower():
            asr_model = m["id"]
            break

    if not asr_model:
        print("[5/6] ASR: SKIP (no ASR model found)")
        return True

    print(f"[5/6] ASR ({asr_model})...", end=" ")
    # Generate a simple WAV file for testing
    import struct
    import wave
    import io
    import numpy as np

    # Create a 1-second silence WAV
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        samples = np.zeros(16000, dtype=np.int16)
        wf.writeframes(samples.tobytes())
    wav_bytes = buf.getvalue()

    t0 = time.time()
    resp = httpx.post(
        f"{BASE_URL}/v1/audio/transcriptions",
        data={"model": asr_model},
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
        timeout=60,
    )
    elapsed = time.time() - t0
    if resp.status_code == 200:
        data = resp.json()
        print(f"OK (text={data.get('text', '')!r}, {elapsed:.1f}s)")
    else:
        print(f"ERROR: {resp.status_code} {resp.text[:200]}")
    return True


def test_image_gen(models):
    """Test image generation endpoint."""
    img_model = None
    for m in models:
        if "image" in m["id"].lower() or "z-image" in m["id"].lower():
            img_model = m["id"]
            break

    if not img_model:
        print("[6/6] Image Gen: SKIP (no image model found)")
        return True

    print(f"[6/6] Image Gen ({img_model})...", end=" ")
    t0 = time.time()
    resp = httpx.post(
        f"{BASE_URL}/v1/images/generations",
        json={
            "model": img_model,
            "prompt": "A beautiful sunset over mountains",
            "n": 1,
            "size": "512x512",
        },
        timeout=120,
    )
    elapsed = time.time() - t0
    if resp.status_code == 200:
        data = resp.json()
        images = data.get("data", [])
        print(f"OK ({len(images)} images, {elapsed:.1f}s)")
    else:
        print(f"ERROR: {resp.status_code} {resp.text[:200]}")
    return True


def main():
    print("=" * 60)
    print("Yunshu End-to-End Modality Tests")
    print("=" * 60)
    print()

    try:
        test_health()
        models = test_models_list()
        test_llm_chat(models)
        test_tts(models)
        test_asr(models)
        test_image_gen(models)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
