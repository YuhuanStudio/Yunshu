"""Multi-turn context in the realtime omni prompt builders.

The native omni path feeds ONE text turn (+ optional raw audio) to the model, so
prior conversation turns must be folded into that text or the model forgets the
conversation. These guard that prior turns are included (bounded for TTFT), while
single-turn prompts stay unchanged.

- text-in  → _messages_to_omni_prompt: persona + prior transcript + latest user.
- speech-in → _omni_system_text: persona + prior transcript (the live query is the
  raw audio, which is NOT present in `messages` as text on the native path).
"""

from yunshu_gateway.routers.realtime import (
    _OMNI_DEFAULT_PERSONA,
    _OMNI_HISTORY_MAX_CHARS,
    _messages_to_omni_prompt,
    _omni_persona,
    _omni_system_text,
)


def test_text_in_single_turn_unchanged():
    msgs = [
        {"role": "system", "content": "You are Yun."},
        {"role": "user", "content": "Hello"},
    ]
    assert _messages_to_omni_prompt(msgs) == "You are Yun.\n\nHello"


def test_text_in_multi_turn_includes_prior():
    msgs = [
        {"role": "system", "content": "You are Yun."},
        {"role": "user", "content": "My name is Bo."},
        {"role": "assistant", "content": "Nice to meet you, Bo."},
        {"role": "user", "content": "What's my name?"},
    ]
    out = _messages_to_omni_prompt(msgs)
    assert "Bo" in out  # prior turn retained → model can answer "Bo"
    assert out.rstrip().endswith("User: What's my name?")  # latest turn last


def test_speech_in_first_turn_is_persona_only():
    # native speech-in: the live query is raw audio (not in messages); first turn
    # has no prior text turns → persona only.
    msgs = [{"role": "system", "content": "You are Yun."}]
    assert _omni_system_text(msgs) == "You are Yun."


def test_speech_in_multi_turn_includes_prior_text():
    msgs = [
        {"role": "system", "content": "You are Yun."},
        {"role": "user", "content": "My name is Bo."},
        {"role": "assistant", "content": "Hi Bo!"},
        # current turn is raw audio → not present as text
    ]
    out = _omni_system_text(msgs)
    assert "You are Yun." in out and "Bo" in out


def test_history_is_bounded_for_ttft():
    # a long conversation must not inflate the prompt without bound
    msgs = [{"role": "system", "content": "P"}]
    for i in range(500):
        msgs.append({"role": "user", "content": f"question number {i} padding text"})
        msgs.append({"role": "assistant", "content": f"answer number {i} padding text"})
    msgs.append({"role": "user", "content": "final"})
    out = _messages_to_omni_prompt(msgs)
    # bounded near the cap (+ persona + framing), and keeps the most recent turns
    assert len(out) < _OMNI_HISTORY_MAX_CHARS + 200
    assert out.rstrip().endswith("User: final")
    assert "question number 499" in out  # recent kept
    assert "question number 0" not in out  # oldest elided


# ── default conversational persona (no system message) ──────────────────────
def test_text_in_no_system_prepends_default_persona():
    out = _messages_to_omni_prompt([{"role": "user", "content": "hi there"}])
    assert out.startswith(_OMNI_DEFAULT_PERSONA)
    assert out.rstrip().endswith("hi there")


def test_speech_in_no_system_uses_default_persona():
    # native speech-in, no system, no prior turns → the default voice persona
    assert _omni_system_text([]) == _OMNI_DEFAULT_PERSONA


def test_user_system_message_overrides_default_persona():
    msgs = [
        {"role": "system", "content": "You are Yun."},
        {"role": "user", "content": "hi"},
    ]
    out = _messages_to_omni_prompt(msgs)
    assert out.startswith("You are Yun.")
    assert _OMNI_DEFAULT_PERSONA not in out  # user's persona wins, default not added


def test_persona_env_override(monkeypatch):
    monkeypatch.setenv("YUNSHU_OMNI_PERSONA", "Be terse.")
    assert _omni_persona() == "Be terse."
    monkeypatch.setenv("YUNSHU_OMNI_PERSONA", "")  # explicit empty disables it
    assert _omni_persona() == ""
    monkeypatch.delenv("YUNSHU_OMNI_PERSONA", raising=False)
    assert _omni_persona() == _OMNI_DEFAULT_PERSONA
