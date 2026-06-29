"""Yunshu quickstart — copy-paste-runnable.

Prereq: a Yunshu server running locally.
    uv pip install "yunshu[all]"
    yunshu serve -m /path/to/Qwen3-Omni-7B-Instruct-4bit --port 8000
    # Set YUNSHU_OMNI_MODEL= same path for the native speech-to-speech section.

Then:
    pip install openai requests
    python examples/quickstart.py
"""

from __future__ import annotations

import base64
import json
import wave

import requests
from openai import OpenAI

BASE = "http://localhost:8000"
client = OpenAI(base_url=f"{BASE}/v1", api_key="local")

# ── Native speech-to-speech (Qwen3-Omni Thinker+Talker) ────────────────────
# One unified model: text → streamed PCM16 audio, ~1 s first-audio on M3 Max.
# No ASR+LLM+TTS cascade. Requires YUNSHU_OMNI_MODEL set on the server.
print("OMNI:", end=" ", flush=True)
pcm_chunks: list[bytes] = []
try:
    with requests.post(
        f"{BASE}/v1/omni/speech/stream",
        json={"text": "Say hello in one sentence.", "speaker": "Ethan"},
        stream=True,
        timeout=120,
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw or not raw.startswith(b"data: "):
                continue
            payload = raw[6:]
            if payload == b"[DONE]":
                break
            ev = json.loads(payload)
            if ev["type"] == "text":
                print(ev["delta"], end="", flush=True)
            elif ev["type"] == "audio":
                pcm_chunks.append(base64.b64decode(ev["delta"]))
    print()
    if pcm_chunks:
        with wave.open("omni_out.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(b"".join(pcm_chunks))
        ms = sum(len(c) for c in pcm_chunks) // 2 * 1000 // 24000
        print(f"  → wrote omni_out.wav ({ms} ms of audio)")
except Exception as e:
    print(f"\n  (omni skipped — is YUNSHU_OMNI_MODEL set on the server? {e})")

print()

# ── Text chat ────────────────────────────────────────────────────────────────
resp = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Say hi in one sentence."}],
)
print("TEXT:", resp.choices[0].message.content)

# ── Streaming text ──────────────────────────────────────────────────────────
print("STREAM:", end=" ", flush=True)
for chunk in client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Count 1 to 5."}],
    stream=True,
):
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()

# ── Vision (requires uv pip install "yunshu[vision]" and a VLM loaded) ──────
# from openai.types.chat import ChatCompletionContentPartImageParam
# img: ChatCompletionContentPartImageParam = {
#     "type": "image_url",
#     "image_url": {"url": "file:///path/to/image.png"},
# }
# vresp = client.chat.completions.create(
#     model="local",
#     messages=[{"role": "user", "content": ["What is in this image?", img]}],
# )
# print("VISION:", vresp.choices[0].message.content)

# ── Text-to-speech (requires uv pip install "yunshu[audio]") ────────────────
# speech = client.audio.speech.create(model="local", voice="alloy",
#                                     input="Hello from Yunshu.")
# speech.write_to_file("out.mp3")
# print("TTS: wrote out.mp3")

# ── Speech-to-text (requires uv pip install "yunshu[audio]") ────────────────
# with open("out.mp3", "rb") as f:
#     transcript = client.audio.transcriptions.create(model="local", file=f)
# print("ASR:", transcript.text)
