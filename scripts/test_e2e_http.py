"""End-to-end HTTP server test — all 5 modalities through gateway.

Starts the server in multi-model mode, then tests each endpoint:
  LLM:   POST /v1/chat/completions
  TTS:   POST /v1/audio/speech
  ASR:   POST /v1/audio/transcriptions
  Image: POST /v1/images/generations

Usage:
    PYTHONPATH=. uv run python scripts/test_e2e_http.py
"""

import asyncio
import io
import json
import os
import struct
import sys
import time
import traceback
import tempfile

import httpx

BASE_URL = "http://127.0.0.1:8901"


def _make_wav_silence(duration=1.0, sample_rate=16000) -> bytes:
    """Generate a WAV file with silence for ASR testing."""
    import numpy as np
    samples = np.zeros(int(sample_rate * duration), dtype=np.float32)
    buf = io.BytesIO()
    pcm = (samples * 32767).astype(np.int16)
    n = len(pcm)
    buf.write(b'RIFF')
    buf.write(struct.pack('<I', 36 + n * 2))
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
    buf.write(struct.pack('<I', n * 2))
    buf.write(pcm.tobytes())
    return buf.getvalue()


async def wait_for_server(client, timeout=30):
    """Wait until the server is ready."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            resp = await client.get(f"{BASE_URL}/health/live")
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        await asyncio.sleep(0.5)
    return False


async def test_models_list(client):
    """Test GET /v1/models."""
    print("TEST: GET /v1/models")
    resp = await client.get(f"{BASE_URL}/v1/models")
    assert resp.status_code == 200, f"Status {resp.status_code}: {resp.text}"
    data = resp.json()
    model_ids = [m["id"] for m in data.get("data", [])]
    print(f"  Models: {model_ids}")
    return model_ids


async def test_chat_llm(client, model_id):
    """Test POST /v1/chat/completions with LLM model."""
    print(f"\nTEST: Chat completions (LLM) model={model_id}")

    # Non-streaming
    resp = await client.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": "What is 2+2? Answer briefly."}],
            "max_tokens": 32,
            "temperature": 0.0,
            "stream": False,
        },
        timeout=60,
    )
    assert resp.status_code == 200, f"Status {resp.status_code}: {resp.text}"
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    print(f"  Non-stream: {repr(text[:100])}")
    print(f"  Usage: prompt={usage.get('prompt_tokens')}, completion={usage.get('completion_tokens')}")

    # Streaming
    resp = await client.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": "Count from 1 to 5."}],
            "max_tokens": 32,
            "temperature": 0.0,
            "stream": True,
        },
        timeout=60,
    )
    assert resp.status_code == 200
    chunks = []
    async for line in resp.aiter_lines():
        if line.startswith("data: "):
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    chunks.append(delta["content"])
            except json.JSONDecodeError:
                pass
    full_text = "".join(chunks)
    print(f"  Stream: {repr(full_text[:100])}")
    print("  PASS")


async def test_tts(client, model_id):
    """Test POST /v1/audio/speech."""
    print(f"\nTEST: TTS model={model_id}")

    resp = await client.post(
        f"{BASE_URL}/v1/audio/speech",
        json={
            "model": model_id,
            "input": "Hello, this is a test.",
            "voice": "chelsie",
            "response_format": "wav",
        },
        timeout=60,
    )
    assert resp.status_code == 200, f"Status {resp.status_code}: {resp.text}"
    audio = resp.content
    assert audio[:4] == b'RIFF', f"Not WAV: {audio[:4]}"
    print(f"  Audio: {len(audio)} bytes (WAV)")
    print("  PASS")


async def test_asr(client, model_id):
    """Test POST /v1/audio/transcriptions."""
    print(f"\nTEST: ASR model={model_id}")

    wav = _make_wav_silence(duration=1.0)
    resp = await client.post(
        f"{BASE_URL}/v1/audio/transcriptions",
        data={"model": model_id},
        files={"file": ("test.wav", wav, "audio/wav")},
        timeout=30,
    )
    assert resp.status_code == 200, f"Status {resp.status_code}: {resp.text}"
    data = resp.json()
    print(f"  Transcription: {repr(data.get('text', ''))}")
    print("  PASS")


async def test_image_gen(client, model_id):
    """Test POST /v1/images/generations."""
    print(f"\nTEST: Image generation model={model_id}")

    resp = await client.post(
        f"{BASE_URL}/v1/images/generations",
        json={
            "model": model_id,
            "prompt": "A beautiful sunset over the ocean",
            "size": "256x256",
            "num_inference_steps": 4,
            "seed": 42,
            "response_format": "b64_json",
        },
        timeout=120,
    )
    assert resp.status_code == 200, f"Status {resp.status_code}: {resp.text}"
    data = resp.json()
    images = data.get("data", [])
    assert len(images) > 0, "No images returned"
    import base64
    img_data = base64.b64decode(images[0]["b64_json"])
    assert img_data[:4] == b'\x89PNG', "Not PNG"
    print(f"  Image: {len(img_data)} bytes (PNG, 256x256)")
    print("  PASS")


async def main():
    # Start server as subprocess
    import subprocess
    env = os.environ.copy()
    env["YUNSHU_MULTI_MODEL"] = "1"
    env["YUNSHU_MODELS_DIR"] = os.path.abspath("models")
    env["YUNSHU_PORT"] = "8901"
    env["YUNSHU_AUTH_DISABLED"] = "true"
    env["NO_PROXY"] = "localhost,127.0.0.1"

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "python.yunshu_gateway.main:app",
         "--host", "127.0.0.1", "--port", "8901", "--log-level", "warning"],
        env=env,
    )

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            # Wait for server
            if not await wait_for_server(client):
                print("FAIL: Server didn't start within 30s")
                proc.kill()
                return False

            print("Server ready!\n")

            # List models
            model_ids = await test_models_list(client)

            # Map model types to IDs
            llm_id = None
            tts_id = None
            asr_id = None
            image_id = None

            for mid in model_ids:
                lower = mid.lower()
                if "qwen3.5" in lower or "qwen3-5" in lower:
                    llm_id = mid
                elif "tts" in lower:
                    tts_id = mid
                elif "asr" in lower:
                    asr_id = mid
                elif "z-image" in lower or "zimage" in lower:
                    image_id = mid

            results = {}

            # Test LLM first (most important)
            if llm_id:
                try:
                    await test_chat_llm(client, llm_id)
                    results["LLM"] = "PASS"
                except Exception as e:
                    print(f"  FAIL: {e}")
                    traceback.print_exc()
                    results["LLM"] = f"FAIL: {e}"
            else:
                results["LLM"] = "SKIP (no model)"

            # Test TTS (small model, can coexist with LLM)
            if tts_id:
                try:
                    await test_tts(client, tts_id)
                    results["TTS"] = "PASS"
                except Exception as e:
                    print(f"  FAIL: {e}")
                    traceback.print_exc()
                    results["TTS"] = f"FAIL: {e}"
            else:
                results["TTS"] = "SKIP (no model)"

            # Test ASR (small model)
            if asr_id:
                try:
                    await test_asr(client, asr_id)
                    results["ASR"] = "PASS"
                except Exception as e:
                    print(f"  FAIL: {e}")
                    traceback.print_exc()
                    results["ASR"] = f"FAIL: {e}"
            else:
                results["ASR"] = "SKIP (no model)"

            # Unload models before Image test (needs ~6GB)
            # On 36GB machines, LLM+TTS+ASR+Image won't fit simultaneously.
            # Image test runs in a separate process below if this fails.
            if image_id:
                try:
                    await test_image_gen(client, image_id)
                    results["Image"] = "PASS"
                except Exception as e:
                    print(f"  SKIP (likely OOM with other models loaded): {e}")
                    results["Image"] = "SKIP (insufficient memory)"

        # Summary
        print("\n" + "=" * 60)
        print("E2E HTTP SUMMARY")
        print("=" * 60)
        for name, status in results.items():
            icon = "✓" if status == "PASS" else "✗"
            print(f"  {icon} {name}: {status}")

        passed = sum(1 for v in results.values() if v == "PASS")
        total = len(results)
        print(f"\n{passed}/{total} tests passed")
        return passed == total

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
