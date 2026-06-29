"""(realtime hunt): RBAC model-access was enforced for the TEXT engine in
_resolve_engine but BYPASSED on the realtime ASR/TTS audio paths — both
_synthesize_audio_response and _handle_input_audio_buffer_commit iterated
manager.list_entries() and grabbed the first synthesize/transcribe engine with NO key
check. A key scoped away from an audio model could still drive it over the socket (the
model-isolation keystone, un-propagated to audio).

Fix: a shared _key_allows(model_id) gate, applied at all three engine-selection sites.
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from unittest import mock

from yunshu_gateway.routers import realtime
from yunshu_gateway.routers.realtime import RealtimeSession


class _DenyKey:
    def __init__(self, denied):
        self._denied = denied

    def can_access_model(self, model_id):
        return model_id not in self._denied


def _session(api_key):
    s = RealtimeSession.__new__(RealtimeSession)
    s._api_key = api_key
    return s


def test_key_allows_semantics():
    # no key (auth disabled / static token) → permissive
    assert _session(None)._key_allows("anything") is True
    # key that denies "tts-x"
    s = _session(_DenyKey({"tts-x"}))
    assert s._key_allows("tts-x") is False
    assert s._key_allows("tts-ok") is True
    # a key whose check raises → fail closed
    raising = SimpleNamespace(
        can_access_model=lambda m: (_ for _ in ()).throw(RuntimeError())
    )
    assert _session(raising)._key_allows("m") is False


def _entry(model_id):
    eng = mock.MagicMock()
    eng.synthesize_stream = mock.MagicMock(
        side_effect=AssertionError("denied engine was driven")
    )
    eng.synthesize = mock.MagicMock(
        side_effect=AssertionError("denied engine was driven")
    )
    return SimpleNamespace(is_loaded=True, model_id=model_id, engine=eng), eng


def test_tts_loop_skips_denied_engine():
    s = _session(_DenyKey({"tts-x"}))
    s.session = SimpleNamespace(voice="v", output_audio_format="pcm16")
    s._g711_resample_remainder = b"carry"
    entry, eng = _entry("tts-x")
    mgr = SimpleNamespace(list_entries=lambda: [entry])
    with (
        mock.patch.object(realtime, "get_model_manager", create=True, return_value=mgr),
        mock.patch("yunshu_gateway.engine.get_model_manager", return_value=mgr),
    ):
        # the only synth engine is denied → loop skips it, engine never driven,
        # function returns without raising the AssertionError side_effects
        asyncio.run(
            s._synthesize_audio_response("hello", "resp_1", "item_1", voice="v")
        )
    eng.synthesize_stream.assert_not_called()
    eng.synthesize.assert_not_called()
    # the fresh-response carry reset still happened (we entered the method body)
    assert s._g711_resample_remainder == b""


def test_both_audio_loops_gate_on_key_allows():
    src = inspect.getsource(realtime)
    # TTS loop
    i_tts = src.index("hasattr(entry.engine, 'synthesize')")
    assert "_key_allows" in src[i_tts : i_tts + 400], (
        "TTS loop missing _key_allows gate"
    )
    # ASR loop
    i_asr = src.index("hasattr(entry.engine, 'transcribe')")
    assert "_key_allows" in src[i_asr : i_asr + 400], (
        "ASR loop missing _key_allows gate"
    )
