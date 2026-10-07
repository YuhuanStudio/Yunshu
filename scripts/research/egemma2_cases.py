"""Shared EmbeddingGemma 2 test cases: texts of mixed length, a >1.5K-token text (exercises the
512-token local window), synthetic images, a speech clip and video frames. `make_media` writes the
files once; `cases(media_dir)` returns {name: item} with file-path media. CPU only."""

from __future__ import annotations

import math
import os
import random
import shutil
import struct
import subprocess
import wave

CAT = "A cat sat on the warm windowsill."
STOCK = "Quarterly earnings beat expectations as the stock market rallied."
LONG_SENT = (
    "The committee met for three days to review the harbour expansion plan, weighing dredging "
    "costs, the effect on migratory birds, ferry timetables and flood defences. "
)
SPEECH = "The weather is nice today, we walk in the park."


def long_text(repeat: int = 70) -> str:
    return "".join(f"({i}) {LONG_SENT}" for i in range(repeat))  # ~1.9K tokens


def _png(path, size, bg, shape, fill):
    from PIL import Image, ImageDraw

    im = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(im)
    w, h = size
    box = (w // 5, h // 6, w * 4 // 5, h * 5 // 6)
    (d.ellipse if shape == "ellipse" else d.rectangle)(box, fill=fill)
    im.save(path)


def _tone_wav(path, seconds=3.0, rate=16000):
    rnd = random.Random(7)
    n = int(seconds * rate)
    frames = bytearray()
    for i in range(n):
        t = i / rate
        v = 0.4 * math.sin(2 * math.pi * (220 + 80 * t) * t) + 0.02 * rnd.uniform(-1, 1)
        frames += struct.pack("<h", int(max(-1, min(1, v)) * 32767))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(frames))


def make_media(d: str) -> dict[str, str]:
    os.makedirs(d, exist_ok=True)
    m = {
        "red": os.path.join(d, "red.png"),
        "blue": os.path.join(d, "blue.png"),
        "speech": os.path.join(d, "speech.wav"),
        "tone": os.path.join(d, "tone.wav"),
    }
    _png(m["red"], (320, 240), (255, 255, 255), "ellipse", (220, 30, 30))
    _png(m["blue"], (200, 300), (30, 60, 200), "rectangle", (250, 230, 40))
    _tone_wav(m["tone"])
    if shutil.which("say") and shutil.which("afconvert"):
        aiff = os.path.join(d, "speech.aiff")
        subprocess.run(["say", "-o", aiff, SPEECH], check=True)
        subprocess.run(
            [
                "afconvert",
                "-f",
                "WAVE",
                "-d",
                "LEI16@16000",
                "-c",
                "1",
                aiff,
                m["speech"],
            ],
            check=True,
        )
    else:
        _tone_wav(m["speech"], 2.0)
    for i in range(4):  # video frames: the red ball moving right
        p = os.path.join(d, f"frame{i}.png")
        from PIL import Image, ImageDraw

        im = Image.new("RGB", (256, 192), (255, 255, 255))
        ImageDraw.Draw(im).ellipse(
            (20 + 50 * i, 60, 80 + 50 * i, 120), fill=(220, 30, 30)
        )
        im.save(p)
        m[f"frame{i}"] = p
    return m


def cases(m: dict[str, str]) -> dict[str, object]:
    frames = [m[f"frame{i}"] for i in range(4)]
    return {
        "text_cat": CAT,
        "text_hi": "Hi",
        "text_stock": STOCK,
        "text_long": long_text(),
        "image_red": {"image": m["red"]},
        "image_blue": {"image": m["blue"]},
        "image_two": {"image": [m["red"], m["blue"]]},
        "text_image": {"text": "a red circle", "image": m["red"]},
        "interleaved": {
            "text": "Photos: <|image|> and <|image|> side by side.",
            "image": [m["red"], m["blue"]],
        },
        "audio_speech": {"audio": m["speech"]},
        "audio_tone": {"audio": m["tone"]},
        "text_audio": {"text": "listen:", "audio": m["speech"]},
        "video": {"video": [frames]},
    }
