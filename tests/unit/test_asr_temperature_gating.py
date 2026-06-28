"""(deferred-LOW from the audio hunt): the ASR transcription path forced
temperature=0.0 (the router default) onto model.generate, which for Whisper DISABLED its
temperature-FALLBACK decoding (Whisper's default is a tuple 0.0,0.2,…,1.0 that retries on
compression/logprob failure) — silently hurting robustness on hard audio. Now temperature
defaults to None (router) and a None temperature is dropped before model.generate (engine),
so the model keeps its own (fallback) default; an explicit value is still honored.
"""
from __future__ import annotations

import inspect

from yunshu_engine import audio_engine
from yunshu_gateway.routers import audio as audio_router


def test_engine_drops_none_temperature():
    src = inspect.getsource(audio_engine.ASREngine.transcribe)
    # the None-temperature is popped so the model uses its own default
    assert 'if gen_kwargs.get("temperature") is None:' in src
    assert 'gen_kwargs.pop("temperature", None)' in src


def test_router_transcription_temperature_defaults_none():
    # scope to the transcription endpoint (the TTS path legitimately keeps Form(0.0))
    src = inspect.getsource(audio_router.create_transcription)
    assert "temperature: float | None = Form(None, ge=0.0, le=1.0)" in src
    assert "temperature: float = Form(0.0" not in src
