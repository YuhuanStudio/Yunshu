"""the ASR (/v1/audio/transcriptions) and streaming-TTS (/v1/audio/speech
stream) paths grabbed the FIRST loaded engine of the right type and ignored `model` —
so with two ASR (or two TTS) models loaded, a request for `whisper` could be served by
`qwen3-asr` (wrong model, wrong sample-rate). _enforce_no_auto_load only proves the
model is loaded; it does NOT bind engine selection. _select_audio_engine now matches by
model_id (mirroring the non-streaming create_speech). Also: the OpenAI SDK's
`timestamp_granularities[]` form key now binds via an alias."""
from __future__ import annotations

import inspect
import types

from yunshu_gateway.routers import (
    audio as A,  # noqa: N812  # intentional short module alias
)
from yunshu_gateway.routers.audio import _select_audio_engine


class _TTSEngine:
    pass


class _OtherTTS(_TTSEngine):
    pass


class _ASREngine:
    pass


def _mgr(entries):
    return types.SimpleNamespace(list_entries=lambda: entries)


def _entry(model_id, engine, loaded=True):
    return types.SimpleNamespace(model_id=model_id, engine=engine, is_loaded=loaded)


def test_selects_matching_model_not_first():
    a, b = _TTSEngine(), _TTSEngine()
    mgr = _mgr([_entry("kokoro", a), _entry("dia", b)])
    # request for the SECOND-registered model must NOT return the first
    assert _select_audio_engine(mgr, "dia", _TTSEngine) is b
    assert _select_audio_engine(mgr, "kokoro", _TTSEngine) is a


def test_case_insensitive_match():
    a = _TTSEngine()
    mgr = _mgr([_entry("Kokoro-82M", a)])
    assert _select_audio_engine(mgr, "kokoro-82m", _TTSEngine) is a


def test_no_match_with_model_returns_none():
    mgr = _mgr([_entry("kokoro", _TTSEngine())])
    assert _select_audio_engine(mgr, "whisper-large", _TTSEngine) is None


def test_empty_model_falls_back_to_first_of_type():
    a, b = _TTSEngine(), _TTSEngine()
    mgr = _mgr([_entry("kokoro", a), _entry("dia", b)])
    assert _select_audio_engine(mgr, "", _TTSEngine) is a


def test_skips_unloaded_and_wrong_type():
    asr = _ASREngine()
    tts = _TTSEngine()
    mgr = _mgr([
        _entry("asr-1", asr),                 # wrong type
        _entry("kokoro", tts, loaded=False),  # right type but unloaded
        _entry("kokoro", tts),                # the live one
    ])
    assert _select_audio_engine(mgr, "kokoro", _TTSEngine) is tts
    assert _select_audio_engine(mgr, "asr-1", _ASREngine) is asr


def test_asr_and_stream_paths_use_selector():
    src = inspect.getsource(A)
    assert "_select_audio_engine(manager, model, ASREngine)" in src
    assert "_select_audio_engine(manager, req.model, TTSEngine)" in src


def test_timestamp_granularities_bracket_alias():
    sig = inspect.signature(A.create_transcription)
    p = sig.parameters.get("timestamp_granularities_bracket")
    assert p is not None
    assert p.default.alias == "timestamp_granularities[]"
