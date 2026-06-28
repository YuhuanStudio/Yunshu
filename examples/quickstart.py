"""Yunshu quickstart — copy-paste-runnable.

Prereq: a Yunshu server running locally.
    pip install "yunshu[all]"     # or just `yunshu` for text-only
    yunshu serve -m mlx-community/Qwen2.5-0.5B-Instruct-4bit --port 8000

Then:
    pip install openai
    python examples/quickstart.py
"""

from __future__ import annotations

from openai import OpenAI

# Yunshu speaks the OpenAI protocol — point the SDK at localhost.
client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")

# ── Text chat ────────────────────────────────────────────────────────────────
resp = client.chat.completions.create(
    model="local",  # Yunshu resolves this to the single loaded model
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

# ── Vision (requires `pip install "yunshu[vision]"` and a VLM loaded) ───────
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

# ── Text-to-speech (requires `pip install "yunshu[audio]"`) ─────────────────
# speech = client.audio.speech.create(
#     model="local",
#     voice="alloy",
#     input="Hello from Yunshu.",
# )
# speech.write_to_file("out.mp3")
# print("TTS: wrote out.mp3")

# ── Speech-to-text (requires `pip install "yunshu[audio]"`) ─────────────────
# with open("out.mp3", "rb") as f:
#     transcript = client.audio.transcriptions.create(model="local", file=f)
# print("ASR:", transcript.text)
