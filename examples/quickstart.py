"""quickstart.py — is my server working? A 30-second tour of the API.

No microphone needed. This hits the server over plain HTTP: it streams a
spoken reply to a WAV file, then does a text chat and a streaming chat.

    # 1. start a server (omni model enables the spoken-reply section)
    YUNSHU_OMNI_MODEL=/path/to/Qwen3-Omni-30B-A3B-Instruct-4bit \
    yunshu serve -m /path/to/Qwen3-Omni-30B-A3B-Instruct-4bit --port 8000

    # 2. run this
    pip install openai requests
    python examples/quickstart.py

To actually *talk* to it (mic in, speech out, live), see talk.py instead.
"""

from __future__ import annotations

import base64
import json
import wave

import requests
from openai import OpenAI

BASE = "http://localhost:8000"
client = OpenAI(base_url=f"{BASE}/v1", api_key="local")  # any api_key works


# ── 1. Native speech-to-speech → a WAV file ──────────────────────────────────
# One unified model turns text into streamed PCM16 audio (no TTS step). The
# server sends Server-Sent Events; we collect the audio chunks into omni_out.wav.
# Requires YUNSHU_OMNI_MODEL set on the server.
print("speech-to-speech:", end=" ", flush=True)
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
            if ev["type"] == "text":  # the spoken words, as text
                print(ev["delta"], end="", flush=True)
            elif ev["type"] == "audio":  # base64 PCM16 @ 24 kHz
                pcm_chunks.append(base64.b64decode(ev["delta"]))
    print()
    if pcm_chunks:
        with wave.open("omni_out.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(b"".join(pcm_chunks))
        ms = sum(len(c) for c in pcm_chunks) // 2 * 1000 // 24000
        print(f"  → wrote omni_out.wav ({ms} ms of audio) — open it to hear the reply")
except Exception as e:
    print(f"\n  (skipped — is YUNSHU_OMNI_MODEL set on the server? {e})")


# ── 2. Text chat (standard OpenAI API) ───────────────────────────────────────
# In single-model mode the model name is a placeholder — the server serves
# whatever you loaded, so "local" is fine.
print("\ntext chat:", end=" ", flush=True)
reply = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Say hi in one sentence."}],
)
print(reply.choices[0].message.content)


# ── 3. Streaming text ────────────────────────────────────────────────────────
print("streaming:", end=" ", flush=True)
for chunk in client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Count from 1 to 5."}],
    stream=True,
):
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print("\n\nDone. Next: examples/talk.py to hold a real spoken conversation.")
