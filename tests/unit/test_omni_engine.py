"""OmniEngine helpers — no model needed."""

from __future__ import annotations

from yunshu_engine.omni_engine import _resolve_speaker


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
