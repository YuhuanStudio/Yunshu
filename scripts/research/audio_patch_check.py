"""Real-checkpoint checks of the carried mlx-audio patches (yunshu_engine.audio_patches).

    audio_patch_check.py whisper|kokoro|qwen3 [--unpatched]

whisper: whisper-small (<|nocaptions|> vocabulary) on speech and on silence; the unpatched
         decoder puts eot in the suppress list and never stops (hits the token limit).
kokoro:  loads the local snapshot with HF_HUB_OFFLINE=1 and synthesises with a local voice.
qwen3:   Qwen3-ASR on a > 20 min English + Chinese clip with language=None; reports the CJK
         share of the transcript per quarter (unpatched: the first chunk's language is forced).
Exits nonzero on a failed expectation of the patched run.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

AUDIO = Path("/Volumes/P5Plus/yunshu-build/verdicts/audio")
MODELS = Path("/Volumes/P5Plus/models")


def cjk_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum("一" <= c <= "鿿" for c in letters) / len(letters)


def whisper(patched: bool) -> dict:
    from yunshu_engine import audio_patches

    if patched:
        audio_patches.install_stt()
    from mlx_audio.stt.utils import load_model

    m = load_model(str(MODELS / "whisper-small-mlx"))
    out: dict = {}
    import numpy as np
    import soundfile as sf

    sil = AUDIO / "silence.wav"
    sf.write(sil, np.zeros(16000 * 8, dtype="float32"), 16000)
    clip = AUDIO / "clip60.wav"
    import wave

    with wave.open(str(AUDIO / "en.wav")) as a, wave.open(str(clip), "wb") as o:
        o.setparams(a.getparams())
        o.writeframes(a.readframes(16000 * 40))
    for name, path in (("speech", clip), ("silence", sil)):
        t0 = time.time()
        r = m.generate(str(path), verbose=False)
        out[name] = {
            "text": r.text[:200],
            "chars": len(r.text),
            "s": round(time.time() - t0, 1),
        }
    ok = out["speech"]["chars"] > 100 and out["silence"]["chars"] < 40
    out["ok"] = ok
    return out


def kokoro(patched: bool) -> dict:
    from yunshu_engine import audio_patches

    os.environ["HF_HUB_OFFLINE"] = "1"
    if patched:
        audio_patches.install_tts()
    from mlx_audio.tts.utils import load_model

    m = load_model(str(MODELS / "Kokoro-82M-bf16"), strict=True)
    n = 0
    for r in m.generate(
        text="Hello, this is a local voice test.", voice="af_heart", lang_code="a"
    ):
        n += len(r.audio)
    return {
        "samples": int(n),
        "sample_rate": getattr(m, "sample_rate", None),
        "ok": n > 24000,
    }


def qwen3(patched: bool, zh_only: bool = False) -> dict:
    from yunshu_engine import audio_patches

    engaged = audio_patches.install_stt() if patched else {}
    print("patches engaged:", engaged, flush=True)
    from mlx_audio.stt.utils import load_model

    m = load_model(str(MODELS / "Qwen3-ASR-1.7B-bf16"))
    stock_extract = m.extract_language

    def traced(text):
        lang, rest = stock_extract(text)
        print("chunk decoded: language", lang, "text", rest[:30], flush=True)
        return lang, rest

    m.extract_language = traced
    t0 = time.time()
    clip = AUDIO / ("zh90.wav" if zh_only else "mix.wav")
    if zh_only:
        import wave

        with wave.open(str(AUDIO / "zh.wav")) as a, wave.open(str(clip), "wb") as o:
            o.setparams(a.getparams())
            o.writeframes(a.readframes(16000 * 90))
    r = m.generate(str(clip), language=None, verbose=False)
    print(
        "language",
        getattr(r, "language", None),
        "segments",
        len(r.segments or []),
        flush=True,
    )
    for seg in getattr(r, "segments", None) or []:
        print("segment", seg.get("language"), str(seg.get("text"))[:40], flush=True)
    text = r.text
    q = len(text) // 4
    shares = [round(cjk_share(text[i * q : (i + 1) * q]), 3) for i in range(4)]
    return {
        "engaged": engaged,
        "chars": len(text),
        "cjk_share_by_quarter": shares,
        "s": round(time.time() - t0),
        "head": text[:150],
        "tail": text[-150:],
        # first half is English (9.3 of 22.3 min), the rest Chinese
        "ok": shares[0] < 0.2 and shares[3] > 0.8,
    }


def main() -> int:
    mode = sys.argv[1]
    patched = "--unpatched" not in sys.argv
    res = {
        "whisper": whisper,
        "kokoro": kokoro,
        "qwen3": qwen3,
        "qwen3zh": lambda p: qwen3(p, True),
    }[mode](patched)
    res.update(mode=mode, patched=patched)
    print(json.dumps(res, ensure_ascii=False), flush=True)
    return 0 if res["ok"] or not patched else 1


if __name__ == "__main__":
    sys.exit(main())
