"""Native-omni realtime path — event-sequence coverage with a fake OmniEngine.

The real path needs a resident Qwen3-Omni (~22GB); these drive
_generate_response_omni with a stub engine so CI verifies the OpenAI-Realtime
event contract (and the opt-in gate) without a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import yunshu_gateway.routers.realtime as rt
from yunshu_gateway.routers import omni


@dataclass
class _Chunk:
    kind: str
    data: Any
    elapsed_s: float = 0.0


class _FakeOmniEngine:
    """Yields two text fragments then one audio chunk then done — like the real
    OmniEngine.stream, but with no model. Records the call args it was given."""

    def __init__(self):
        self.calls = []

    async def stream(
        self,
        text,
        image_path=None,
        audio_path=None,
        speaker=None,
        thinker_max_new_tokens=None,
    ):
        self.calls.append({"text": text, "audio_path": audio_path, "speaker": speaker})
        yield _Chunk("text", "Hi ")
        yield _Chunk("text", "there")
        yield _Chunk("audio", np.linspace(-0.3, 0.3, 2400, dtype=np.float32))
        yield _Chunk(
            "done", {"first_audio_s": 0.1, "audio_seconds": 0.1, "total_s": 0.2}
        )


def _session_with_user_turn():
    ws = MagicMock()
    ws.send_json = AsyncMock()
    session = rt.RealtimeSession(ws)
    session.conversation.add_item(
        rt.ConversationItem(
            "u1",
            "message",
            role="user",
            content=[{"type": "input_text", "text": "hello"}],
        )
    )
    return session, ws


def test_gate_default_policy(monkeypatch):
    # No env and no speakable served model → cascade (off): non-omni models unaffected.
    monkeypatch.delenv("YUNSHU_OMNI_MODEL", raising=False)
    monkeypatch.delenv("YUNSHU_REALTIME_OMNI", raising=False)
    monkeypatch.setattr(omni, "_shared_speakable_model", lambda: None)
    assert rt._omni_realtime_enabled() is False
    # A configured omni model → native voice ON by default (no second flag needed).
    monkeypatch.setenv("YUNSHU_OMNI_MODEL", "/x")
    assert rt._omni_realtime_enabled() is True
    # Explicit opt-out forces the cascade even with a model configured.
    monkeypatch.setenv("YUNSHU_REALTIME_OMNI", "0")
    assert rt._omni_realtime_enabled() is False
    # Explicit opt-in forces it on even without a model path.
    monkeypatch.delenv("YUNSHU_OMNI_MODEL", raising=False)
    monkeypatch.setenv("YUNSHU_REALTIME_OMNI", "1")
    assert rt._omni_realtime_enabled() is True
    # Reuse: the served model itself can speak (no env) → on by default.
    monkeypatch.delenv("YUNSHU_REALTIME_OMNI", raising=False)
    monkeypatch.setattr(omni, "_shared_speakable_model", lambda: (object(), object()))
    assert rt._omni_realtime_enabled() is True


def test_realtime_active_requires_loadable_talker(monkeypatch):
    """_omni_realtime_active() = enabled AND a Talker model actually loadable.
    Enabled but the probe failing (no Talker / load error) → inactive, so the path
    falls back to the ASR→LLM→TTS cascade instead of erroring (B1)."""
    monkeypatch.setenv("YUNSHU_OMNI_MODEL", "/x")
    monkeypatch.setenv("YUNSHU_REALTIME_OMNI", "1")
    monkeypatch.setattr(rt, "_omni_speech_ready", lambda: False)
    assert rt._omni_realtime_active() is False
    monkeypatch.setattr(rt, "_omni_speech_ready", lambda: True)
    assert rt._omni_realtime_active() is True
    # explicit opt-out (=0) short-circuits before the probe → inactive even if ready
    monkeypatch.setenv("YUNSHU_REALTIME_OMNI", "0")
    assert rt._omni_realtime_active() is False


def test_messages_to_omni_prompt_prepends_system():
    msgs = [
        {"role": "system", "content": "You are Yun."},
        {"role": "user", "content": "first"},
        {"role": "user", "content": "latest"},
    ]
    p = rt._messages_to_omni_prompt(msgs)
    # persona + prior-turn context ("first") + the latest user turn last —
    # multi-turn continuity (prior turns are no longer dropped).
    assert p.startswith("You are Yun.\n\n")
    assert "first" in p
    assert p.rstrip().endswith("User: latest")
    # single-turn still collapses to persona + the lone user text (unchanged)
    assert (
        rt._messages_to_omni_prompt(
            [
                {"role": "system", "content": "You are Yun."},
                {"role": "user", "content": "only"},
            ]
        )
        == "You are Yun.\n\nonly"
    )
    assert rt._messages_to_omni_prompt([]) == ""


@pytest.mark.asyncio
async def test_omni_response_emits_full_event_chain(monkeypatch):
    monkeypatch.setattr(omni, "_get_omni_engine", lambda: _FakeOmniEngine())
    session, ws = _session_with_user_turn()

    await session._generate_response_omni("resp_1", "item_1", ["text", "audio"], {})

    types = [c[0][0]["type"] for c in ws.send_json.call_args_list]
    # Required OpenAI-Realtime lifecycle order.
    expected_order = [
        "response.output_item.added",
        "response.content_part.added",
        "response.audio_transcript.done",
        "response.audio.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.done",
    ]
    positions = [types.index(t) for t in expected_order]
    assert positions == sorted(positions), types
    # text + transcript + audio deltas all fired
    assert "response.text.delta" in types
    assert "response.audio_transcript.delta" in types
    assert "response.audio.delta" in types
    # exactly one terminal of each
    assert types.count("response.done") == 1
    assert types.count("response.audio.done") == 1
    assert session._response_done_emitted is True


@pytest.mark.asyncio
async def test_omni_response_stores_assistant_audio_turn(monkeypatch):
    monkeypatch.setattr(omni, "_get_omni_engine", lambda: _FakeOmniEngine())
    session, _ = _session_with_user_turn()

    await session._generate_response_omni("resp_1", "item_1", ["text", "audio"], {})

    items = session.conversation.items
    assistant = [i for i in items if i.role == "assistant"]
    assert len(assistant) == 1
    part = assistant[0].content[0]
    assert part["type"] == "audio"  # stored as audio so barge-in truncate works
    assert part["transcript"] == "Hi there"


@pytest.mark.asyncio
async def test_omni_uses_raw_audio_for_native_speech_in(monkeypatch):
    """When raw user audio is stashed, omni gets audio_path (native speech-in),
    the temp WAV is cleaned up, and the stash is consumed (not reused next turn)."""
    import os

    fake = _FakeOmniEngine()
    monkeypatch.setattr(omni, "_get_omni_engine", lambda: fake)
    session, _ = _session_with_user_turn()
    # simulate a committed user-audio buffer (pcm16 @ 24k)
    pcm = (np.zeros(4800, dtype="<i2")).tobytes()
    session._last_user_audio = (pcm, 24000)

    await session._generate_response_omni("resp_1", "item_1", ["text", "audio"], {})

    assert len(fake.calls) == 1
    audio_path = fake.calls[0]["audio_path"]
    assert audio_path is not None  # native speech-in path taken
    assert not os.path.exists(audio_path)  # temp WAV cleaned up in finally
    assert session._last_user_audio is None  # consumed


@pytest.mark.asyncio
async def test_omni_falls_back_to_text_without_audio(monkeypatch):
    fake = _FakeOmniEngine()
    monkeypatch.setattr(omni, "_get_omni_engine", lambda: fake)
    session, _ = _session_with_user_turn()
    assert session._last_user_audio is None

    await session._generate_response_omni("resp_1", "item_1", ["text", "audio"], {})

    assert fake.calls[0]["audio_path"] is None  # text path
    assert "hello" in fake.calls[0]["text"]  # the user's transcript/text


@pytest.mark.asyncio
async def test_omni_response_text_only_skips_audio(monkeypatch):
    monkeypatch.setattr(omni, "_get_omni_engine", lambda: _FakeOmniEngine())
    session, ws = _session_with_user_turn()

    await session._generate_response_omni("resp_1", "item_1", ["text"], {})

    types = [c[0][0]["type"] for c in ws.send_json.call_args_list]
    assert "response.text.delta" in types
    assert "response.audio.delta" not in types  # text-only modality
    assert "response.audio.done" not in types
    assert types.count("response.done") == 1


class _FakeOmniToolEngine:
    """Yields a Thinker turn that emits <tool_call> markup, like a real omni model
    deciding to call a tool. Records the prompt it was given so the test can assert
    the tool definitions were injected."""

    model_path = "/models/qwen3-omni"

    def __init__(self):
        self.calls = []

    async def stream(
        self,
        text,
        image_path=None,
        audio_path=None,
        speaker=None,
        thinker_max_new_tokens=None,
    ):
        self.calls.append({"text": text, "audio_path": audio_path, "speaker": speaker})
        yield _Chunk("text", "Let me check. ")
        yield _Chunk(
            "text",
            '<tool_call>{"name": "get_weather", "arguments": {"city": "SF"}}</tool_call>',
        )
        yield _Chunk(
            "done", {"first_audio_s": 0.0, "audio_seconds": 0.0, "total_s": 0.1}
        )


_WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


@pytest.mark.asyncio
async def test_omni_injects_tools_and_emits_function_call(monkeypatch):
    fake = _FakeOmniToolEngine()
    monkeypatch.setattr(omni, "_get_omni_engine", lambda: fake)
    session, ws = _session_with_user_turn()
    session.session.tools = [_WEATHER_TOOL]

    await session._generate_response_omni("resp_1", "item_1", ["text"], {})

    # 1) tool definitions were injected into the omni prompt (so the Thinker can call)
    prompt = fake.calls[0]["text"]
    assert "Available tools:" in prompt
    assert "get_weather" in prompt

    # 2) the function_call is emitted as its OWN output item with arguments + a fresh call_id
    events = [c[0][0] for c in ws.send_json.call_args_list]
    types = [e["type"] for e in events]
    added = [
        e
        for e in events
        if e["type"] == "response.output_item.added"
        and e.get("item", {}).get("type") == "function_call"
    ]
    assert len(added) == 1
    assert added[0]["item"]["name"] == "get_weather"
    assert added[0]["item"]["call_id"].startswith("call_")
    assert "response.function_call_arguments.done" in types
    assert types.count("response.done") == 1

    # 3) function_call output is included in response.done and stored in history
    done = [e for e in events if e["type"] == "response.done"][0]
    fc_out = [o for o in done["response"]["output"] if o.get("type") == "function_call"]
    assert len(fc_out) == 1
    history_fc = [
        i for i in session.conversation.items if i.item_type == "function_call"
    ]
    assert len(history_fc) == 1
    assert history_fc[0].name == "get_weather"

    # 4) the markup is stripped from the stored transcript so it isn't re-fed/spoken
    assistant_msg = [i for i in session.conversation.items if i.role == "assistant"][0]
    assert "<tool_call>" not in assistant_msg.content[0]["text"]


@pytest.mark.asyncio
async def test_omni_tool_choice_none_suppresses_call(monkeypatch):
    """tool_choice='none' must NOT emit function_call items even if the Thinker
    emits markup — and the markup is still stripped from the transcript."""
    fake = _FakeOmniToolEngine()
    monkeypatch.setattr(omni, "_get_omni_engine", lambda: fake)
    session, ws = _session_with_user_turn()
    session.session.tools = [_WEATHER_TOOL]
    session.session.tool_choice = "none"

    await session._generate_response_omni("resp_1", "item_1", ["text"], {})

    # prompt tells the model NOT to call tools
    assert "must NOT call" in fake.calls[0]["text"]
    events = [c[0][0] for c in ws.send_json.call_args_list]
    fc_added = [
        e
        for e in events
        if e["type"] == "response.output_item.added"
        and e.get("item", {}).get("type") == "function_call"
    ]
    assert fc_added == []  # no function call emitted
    history_fc = [
        i for i in session.conversation.items if i.item_type == "function_call"
    ]
    assert history_fc == []
    # stray markup still stripped from the stored transcript
    assistant_msg = [i for i in session.conversation.items if i.role == "assistant"][0]
    assert "<tool_call>" not in assistant_msg.content[0]["text"]


@pytest.mark.asyncio
async def test_omni_no_tools_leaves_prompt_untouched(monkeypatch):
    """The common no-tools voice path must not gain any tool-instruction text."""
    fake = _FakeOmniEngine()
    monkeypatch.setattr(omni, "_get_omni_engine", lambda: fake)
    session, _ = _session_with_user_turn()
    assert session.session.tools == []

    await session._generate_response_omni("resp_1", "item_1", ["text", "audio"], {})

    assert "Available tools" not in fake.calls[0]["text"]


def test_omni_tools_prompt_honors_tool_choice():
    # auto: advisory, includes the markup format + tool name
    p = rt._omni_tools_prompt([_WEATHER_TOOL], "auto")
    assert "get_weather" in p and "<tool_call>" in p and "Decide whether" in p
    # required: must call at least one
    assert "MUST call at least one" in rt._omni_tools_prompt(
        [_WEATHER_TOOL], "required"
    )
    # named dict: force a specific tool
    forced = rt._omni_tools_prompt(
        [_WEATHER_TOOL], {"type": "function", "function": {"name": "get_weather"}}
    )
    assert "MUST call the tool 'get_weather'" in forced
    # none: forbid, no format guidance
    none_p = rt._omni_tools_prompt([_WEATHER_TOOL], "none")
    assert "must NOT call" in none_p and "<tool_call>" not in none_p
    # empty tools → empty string (no injection)
    assert rt._omni_tools_prompt([], "auto") == ""
