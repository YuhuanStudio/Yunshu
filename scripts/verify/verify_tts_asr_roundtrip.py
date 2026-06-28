"""TTS→ASR round-trip gate (real audio engines).

verify_asr_quality.py checks ASR on a fixed clip; verify_voicedesign.py checks
TTS produces distinct voices. Neither checks that the audio pipeline is
END-TO-END faithful. This gate synthesizes known text to speech, transcribes it
back, and asserts the words round-trip — a single semantic loop covering both
TTS synthesis and ASR transcription.

  - TTS produces a valid, non-trivial WAV (RIFF header, > 0.3s)
  - ASR transcribes it back so that nearly all content words survive
    (>=0.8 word-overlap) for each of two distinct phrases

Loads the TTS model, synthesizes, frees it, then loads ASR — never both at once.

Run: PYTHONPATH=. uv run python scripts/verify_tts_asr_roundtrip.py
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
import tempfile

TTS = os.environ.get("YUNSHU_TTS_MODEL", "./models/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")
ASR = os.environ.get("YUNSHU_ASR_MODEL", "./models/Qwen3-ASR-1.7B-bf16")

PHRASES = [
    "The quick brown fox jumps over the lazy dog.",
    "Paris is the capital of France and Tokyo is the capital of Japan.",
]


def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower())}


def _overlap(orig: str, got: str) -> float:
    o = _words(orig)
    return len(o & _words(got)) / len(o) if o else 0.0


async def main() -> int:
    if not os.path.isdir(TTS) or not os.path.exists(ASR):
        print(f"SKIP: audio models not available ({TTS}, {ASR})")
        return 0

    from yunshu_engine.audio_engine import TTSEngine
    from yunshu_engine.types import EngineConfig

    # ── synthesize all phrases (TTS loaded, then freed) ──────────────────────
    wav_paths: list[str] = []
    durations: list[float] = []
    eng = TTSEngine(TTS, EngineConfig())
    await eng.start()
    try:
        for ph in PHRASES:
            wav = await eng.synthesize(ph, instruct="a clear, neutral voice")
            p = tempfile.mktemp(suffix=".wav")
            with open(p, "wb") as f:
                f.write(wav)
            wav_paths.append(p)
            # crude duration: bytes after 44-byte header / (16kHz * 2 bytes), lower bound
            durations.append(max(0.0, (len(wav) - 44) / (16000 * 2)))
    finally:
        if hasattr(eng, "stop"):
            await eng.stop()

    checks: dict[str, bool] = {}
    detail: list[str] = []
    valid_wav = all(open(p, "rb").read(4) == b"RIFF" for p in wav_paths) and all(d > 0.3 for d in durations)
    checks["TTS: valid non-trivial WAV for each phrase"] = valid_wav

    # ── transcribe back (ASR loaded after TTS freed) ─────────────────────────
    try:
        from mlx_audio.stt import load
        model = load(ASR)
        overlaps = []
        for ph, p in zip(PHRASES, wav_paths):
            r = model.generate(p)
            txt = getattr(r, "text", None) or str(r)
            ov = _overlap(ph, txt)
            overlaps.append(ov)
            detail.append(f"ov={ov:.2f} :: {txt[:60]!r}")
        checks["round-trip: >=0.8 word overlap (both phrases)"] = all(o >= 0.8 for o in overlaps)
    finally:
        for p in wav_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    for k, v in checks.items():
        print(f"  {'OK ' if v else 'BAD'} {k}")
    for line in detail:
        print(f"     {line}")
    ok = all(checks.values())
    print(f"RESULT: {sum(checks.values())}/{len(checks)} passed")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
