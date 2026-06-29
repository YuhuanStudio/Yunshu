"""OmniEngine helpers — no model needed."""

from __future__ import annotations

import contextlib
import os
import wave

import pytest

from yunshu_engine.omni_engine import OmniEngine, _resolve_speaker


def test_valid_omni_speaker_passes_through():
    assert _resolve_speaker("Chelsie", "Ethan") == "Chelsie"
    assert _resolve_speaker("aiden", "Ethan") == "Aiden"  # case-normalized


def test_openai_voice_aliased_to_talker_speaker():
    assert _resolve_speaker("alloy", "Ethan") == "Ethan"  # default OpenAI voice
    assert _resolve_speaker("nova", "Ethan") == "Chelsie"  # female alias


def test_unknown_voice_falls_back_not_raises():
    # The model raises NotImplementedError for unknown speakers; the engine must
    # never forward one — fall back to the default instead.
    assert _resolve_speaker("does-not-exist", "Ethan") == "Ethan"
    assert _resolve_speaker(None, "Chelsie") == "Chelsie"
    assert _resolve_speaker("", "Ethan") == "Ethan"


def test_make_warmup_audio_writes_valid_wav():
    # The audio-encoder warmup pass needs a real readable WAV. __init__ does not
    # load a model, so the engine can be built without one.
    eng = OmniEngine("/fake/model/path")
    path = eng._make_warmup_audio()
    assert path is not None and os.path.exists(path)
    try:
        with wave.open(path, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2  # int16
            assert wf.getframerate() == 16000
            assert wf.getnframes() > 0
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)


@pytest.mark.asyncio
async def test_warmup_primes_the_audio_in_path():
    """warmup() must run at least one speech-IN pass so the audio-encoder kernels
    compile during warmup, not on the user's first real voice turn."""
    eng = OmniEngine("/fake/model/path")
    seen_audio_paths: list[str | None] = []

    def _fake_load() -> None:
        pass

    async def _fake_stream(text, *, audio_path=None, **kwargs):
        seen_audio_paths.append(audio_path)
        return
        yield  # make this an async generator

    eng.load = _fake_load  # type: ignore[method-assign]
    eng.stream = _fake_stream  # type: ignore[method-assign]

    await eng.warmup(rounds=1)

    # at least one warmup pass fed an audio_path (the speech-encoder priming pass)
    assert any(p for p in seen_audio_paths), (
        f"no audio-in warmup pass observed; audio_paths={seen_audio_paths}"
    )
    # and the throwaway clip was cleaned up
    for p in seen_audio_paths:
        if p:
            assert not os.path.exists(p), f"warmup clip {p} leaked"
