"""TTS VoiceDesign gate (Qwen3-TTS-1.7B-VoiceDesign).

Verifies the `instruct` voice-description path: synthesizing the SAME text with two
distinct voice descriptions yields valid, NON-EMPTY, and AUDIBLY DIFFERENT audio
(the instruct actually steers the voice). Guards the VoiceDesign synthesis path.

Run: PYTHONPATH=. uv run python scripts/verify_voicedesign.py
"""
from __future__ import annotations

import asyncio
import io
import os
import sys
import wave

MODEL = os.environ.get("YUNSHU_TTS_MODEL", "./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")
TEXT = "The quick brown fox jumps over the lazy dog."


def _pcm(b):
    import numpy as np
    with wave.open(io.BytesIO(b)) as w:
        n = w.getnframes()
        return np.frombuffer(w.readframes(n), dtype=np.int16).astype("float32"), w.getframerate()


async def main() -> int:
    if not os.path.isdir(MODEL):
        print(f"SKIP: TTS model not available ({MODEL})")
        return 0
    from yunshu_engine.audio_engine import TTSEngine
    from yunshu_engine.types import EngineConfig
    import numpy as np
    eng = TTSEngine(MODEL, EngineConfig())
    await eng.start()
    try:
        a = await eng.synthesize(TEXT, instruct="a deep, slow, calm male voice")
        b = await eng.synthesize(TEXT, instruct="a bright, energetic, high-pitched young female voice")
    finally:
        if hasattr(eng, "stop"):
            await eng.stop()
    pa, sr_a = _pcm(a)
    pb, _ = _pcm(b)
    valid = a[:4] == b"RIFF" and b[:4] == b"RIFF" and len(pa) > sr_a * 0.3 and len(pb) > sr_a * 0.3
    # different instructs → different waveform (compare on the shared length)
    n = min(len(pa), len(pb))
    diff = float(np.abs(pa[:n] - pb[:n]).mean()) if n else 0.0
    differs = (abs(len(pa) - len(pb)) > sr_a * 0.05) or diff > 50.0
    checks = {"both valid non-empty WAV": valid,
              "instruct steers voice (audios differ)": differs}
    for k, v in checks.items():
        print(f"  {'OK ' if v else 'BAD'} {k}")
    print(f"RESULT: dur_a={len(pa)/sr_a:.1f}s dur_b={len(pb)/_pcm(b)[1]:.1f}s diff={diff:.0f}")
    ok = all(checks.values())
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
