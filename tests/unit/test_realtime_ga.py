"""GA <-> internal Realtime event translation (pure functions)."""

from yunshu_gateway import realtime_ga as ga


def test_wants_beta():
    assert ga.wants_beta({"openai-beta": "realtime=v1"})
    assert not ga.wants_beta({})


def test_session_update_from_ga():
    ev = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "output_modalities": ["audio"],
            "instructions": "hi",
            "max_output_tokens": 50,
            "audio": {
                "input": {
                    "format": {"type": "audio/pcmu"},
                    "turn_detection": {"type": "semantic_vad", "eagerness": "high"},
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": "marin",
                },
            },
        },
    }
    s = ga.from_ga(ev)["session"]
    assert s == {
        "modalities": ["audio"],
        "instructions": "hi",
        "max_response_output_tokens": 50,
        "input_audio_format": "g711_ulaw",
        "output_audio_format": "pcm16",
        "voice": "marin",
        "turn_detection": {"type": "server_vad"},
    }


def test_response_create_from_ga():
    r = ga.from_ga(
        {
            "type": "response.create",
            "response": {
                "output_modalities": ["text"],
                "max_output_tokens": 9,
                "audio": {"output": {"voice": "cedar"}},
                "instructions": "x",
            },
        }
    )["response"]
    assert r == {
        "modalities": ["text"],
        "max_response_output_tokens": 9,
        "voice": "cedar",
        "instructions": "x",
    }


def test_server_events_to_ga():
    assert ga.to_ga({"type": "conversation.created"}) == []
    assert ga.to_ga({"type": "response.text.delta", "delta": "a"})[0]["type"] == (
        "response.output_text.delta"
    )
    assert ga.to_ga({"type": "response.audio_transcript.done"})[0]["type"] == (
        "response.output_audio_transcript.done"
    )
    added, done = ga.to_ga(
        {
            "type": "conversation.item.created",
            "event_id": "e1",
            "item": {
                "id": "i",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "x"}],
            },
        }
    )
    assert (added["type"], done["type"]) == (
        "conversation.item.added",
        "conversation.item.done",
    )
    assert added["item"]["content"][0]["type"] == "output_text"
    resp = ga.to_ga(
        {"type": "response.done", "response": {"modalities": ["audio"], "output": []}}
    )[0]["response"]
    assert resp["output_modalities"] == ["audio"] and "modalities" not in resp
    sess = ga.to_ga(
        {
            "type": "session.created",
            "session": {
                "modalities": ["text"],
                "input_audio_format": "g711_alaw",
                "voice": "alloy",
                "turn_detection": None,
            },
        }
    )[0]["session"]
    assert sess["type"] == "realtime"
    assert sess["audio"]["input"]["format"] == {"type": "audio/pcma"}
    assert sess["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}
