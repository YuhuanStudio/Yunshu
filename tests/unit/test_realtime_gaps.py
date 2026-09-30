"""Realtime GA gaps: idle_timeout_ms, noise_reduction, output_audio_buffer.*,
rate_limits.updated, response.create.input, audio content_part shape."""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from yunshu_engine import settings
from yunshu_gateway import realtime_ga
from yunshu_gateway.realtime_dsp import NoiseReducer, supported
from yunshu_gateway.routers import realtime as rt


def _session(dialect="ga"):
    ws = MagicMock()
    sent: list[dict] = []

    async def send_json(ev):
        sent.append(ev)

    ws.send_json = AsyncMock(side_effect=send_json)
    s = rt.RealtimeSession(ws, dialect=dialect)
    s._resolve_engine = lambda: None  # type: ignore[method-assign]
    return s, sent


def _types(sent):
    return [e["type"] for e in sent]


@pytest.fixture(autouse=True)
def _clean():
    yield
    settings.clear_overrides()


# ── GA translator ──


def test_ga_session_maps_idle_timeout_and_noise_reduction():
    internal = realtime_ga.from_ga(
        {
            "type": "session.update",
            "session": {
                "audio": {
                    "input": {
                        "noise_reduction": {"type": "far_field"},
                        "turn_detection": {
                            "type": "server_vad",
                            "idle_timeout_ms": 6000,
                            "create_response": True,
                        },
                    }
                }
            },
        }
    )["session"]
    assert internal["input_audio_noise_reduction"] == {"type": "far_field"}
    assert internal["turn_detection"]["idle_timeout_ms"] == 6000
    back = realtime_ga._session_to_ga(
        {
            "input_audio_noise_reduction": {"type": "far_field"},
            "turn_detection": {"type": "server_vad", "idle_timeout_ms": 6000},
        }
    )
    assert back["audio"]["input"]["noise_reduction"] == {"type": "far_field"}
    assert back["audio"]["input"]["turn_detection"]["idle_timeout_ms"] == 6000
    assert realtime_ga._session_to_ga({})["audio"]["input"]["noise_reduction"] is None


def test_ga_response_create_keeps_input():
    out = realtime_ga.from_ga(
        {"type": "response.create", "response": {"input": [], "conversation": "none"}}
    )
    assert out["response"]["input"] == []


def test_session_validates_new_fields():
    s, _ = _session()
    s.session.update({"input_audio_noise_reduction": {"type": "bogus"}})
    assert s.session.input_audio_noise_reduction is None
    s.session.update({"input_audio_noise_reduction": {"type": "near_field"}})
    assert s.session.input_audio_noise_reduction == {"type": "near_field"}
    s.session.update({"input_audio_noise_reduction": None})
    assert s.session.input_audio_noise_reduction is None
    s.session.update({"turn_detection": {"type": "server_vad", "idle_timeout_ms": -5}})
    assert "idle_timeout_ms" not in s.session.turn_detection
    s.session.update(
        {"turn_detection": {"type": "server_vad", "idle_timeout_ms": 5000}}
    )
    assert s.session.turn_detection["idle_timeout_ms"] == 5000


# ── rate limits / output audio buffer / content part ──


@pytest.mark.asyncio
async def test_rate_limits_updated_follows_response_created():
    for dialect in ("ga", "beta"):
        s, sent = _session(dialect)
        await s._handle_response_create(
            {"type": "response.create", "response": {"modalities": ["text"]}}
        )
        types = _types(sent)
        assert types.index("rate_limits.updated") == types.index("response.created") + 1
        limits = {
            r["name"]: r
            for r in sent[types.index("rate_limits.updated")]["rate_limits"]
        }
        assert set(limits) == {"requests", "tokens"}
        assert all(isinstance(v["limit"], int) for v in limits.values())
        if s._active_response:
            s._active_response.cancel()


@pytest.mark.asyncio
async def test_rate_limits_use_configured_rpm():
    settings.set_override("YUNSHU_RATE_LIMIT_RPM", 30)
    s, _ = _session()
    ev = s._rate_limits_event()
    req = ev["rate_limits"][0]
    assert req["limit"] == 30 and req["remaining"] == 29


@pytest.mark.asyncio
async def test_output_audio_buffer_lifecycle_ga_only():
    s, sent = _session("ga")
    delta = base64.b64encode(b"\0" * 96).decode()
    await s.send_event(
        {"type": "response.audio.delta", "response_id": "r1", "delta": delta}
    )
    await s.send_event(
        {"type": "response.audio.delta", "response_id": "r1", "delta": delta}
    )
    await s.send_event({"type": "response.audio.done", "response_id": "r1"})
    assert _types(sent).count("output_audio_buffer.started") == 1
    assert _types(sent)[-1] == "output_audio_buffer.stopped"
    assert s._resp_audio_bytes == 192

    # a cut response reports cleared
    sent.clear()
    await s.send_event(
        {"type": "response.audio.delta", "response_id": "r2", "delta": delta}
    )
    s._obuf_cleared = True
    await s.send_event({"type": "response.audio.done", "response_id": "r2"})
    assert _types(sent)[-1] == "output_audio_buffer.cleared"

    beta, bsent = _session("beta")
    await beta.send_event(
        {"type": "response.audio.delta", "response_id": "r", "delta": delta}
    )
    await beta.send_event({"type": "response.audio.done", "response_id": "r"})
    assert not [t for t in _types(bsent) if t.startswith("output_audio_buffer")]


@pytest.mark.asyncio
async def test_output_audio_buffer_clear_cancels_active_response():
    s, _ = _session()
    s._handle_response_cancel = AsyncMock()  # type: ignore[method-assign]
    await s._handle_output_audio_buffer_clear({"type": "output_audio_buffer.clear"})
    s._handle_response_cancel.assert_not_called()  # nothing playing: no-op

    async def forever():
        await asyncio.sleep(30)

    s._active_response = asyncio.create_task(forever())
    await s._handle_output_audio_buffer_clear({"type": "output_audio_buffer.clear"})
    s._handle_response_cancel.assert_awaited_once()
    s._active_response.cancel()


@pytest.mark.asyncio
async def test_audio_content_part_shape():
    s, sent = _session()
    await s.send_event(
        {"type": "response.created", "response": {"id": "r", "modalities": ["audio"]}}
    )
    assert s._content_part("hi") == {"type": "audio", "transcript": "hi"}
    ga = realtime_ga.to_ga(
        {"type": "response.content_part.done", "part": s._content_part("hi")}
    )
    assert ga[0]["part"] == {"type": "output_audio", "transcript": "hi"}
    await s.send_event(
        {"type": "response.created", "response": {"id": "r2", "modalities": ["text"]}}
    )
    assert s._content_part("hi") == {"type": "text", "text": "hi"}


# ── idle timeout ──


@pytest.mark.asyncio
async def test_idle_timeout_triggers_a_response():
    s, sent = _session("beta")
    s.session.update({"turn_detection": {"type": "server_vad", "idle_timeout_ms": 60}})
    s._handle_response_create = AsyncMock()  # type: ignore[method-assign]
    await s.send_event({"type": "response.done", "response": {"status": "completed"}})
    await asyncio.sleep(0.25)
    types = _types(sent)
    assert types[-3:] == [
        "input_audio_buffer.timeout_triggered",
        "input_audio_buffer.committed",
        "conversation.item.created",
    ]
    trig = sent[-3]
    assert trig["item_id"] == sent[-2]["item_id"]
    assert {"audio_start_ms", "audio_end_ms"} <= set(trig)
    s._handle_response_create.assert_awaited_once()
    assert s.conversation.items[-1].role == "user"


@pytest.mark.asyncio
async def test_idle_timeout_cancelled_by_speech_or_new_response():
    s, sent = _session()
    s.session.update({"turn_detection": {"type": "server_vad", "idle_timeout_ms": 80}})
    s._handle_response_create = AsyncMock()  # type: ignore[method-assign]
    await s.send_event({"type": "response.done", "response": {"status": "completed"}})
    assert s._idle_task is not None
    s._cancel_idle_timer()
    await asyncio.sleep(0.2)
    assert "input_audio_buffer.timeout_triggered" not in _types(sent)
    # a response that is not completed does not arm it; nor does a session without it
    await s.send_event({"type": "response.done", "response": {"status": "cancelled"}})
    assert s._idle_task is None
    s2, _ = _session()
    await s2.send_event({"type": "response.done", "response": {"status": "completed"}})
    assert s2._idle_task is None


# ── response.create.input ──


def _item(text, role="user", id_=None):
    d = {
        "type": "message",
        "role": role,
        "content": [{"type": "input_text", "text": text}],
    }
    if id_:
        d["id"] = id_
    return d


def test_response_input_replaces_conversation_context():
    s, _ = _session()
    s.conversation.add_item(
        rt.ConversationItem(
            "i1", "message", "user", [{"type": "input_text", "text": "old"}]
        )
    )
    default = s._build_messages()
    assert [m["content"] for m in default] == ["old"]
    msgs = s._build_messages(input_items=[_item("out of band")])
    assert [m["content"] for m in msgs] == ["out of band"]
    assert s._build_messages(input_items=[]) == []  # [] clears the context
    ref = s._build_messages(
        input_items=[{"type": "item_reference", "id": "i1"}, _item("and this")]
    )
    assert [m["content"] for m in ref] == ["old", "and this"]
    out = s._build_messages(
        input_items=[
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "earlier answer"}],
            }
        ]
    )
    assert out[0]["role"] == "assistant" and out[0]["content"] == "earlier answer"


@pytest.mark.asyncio
async def test_response_input_must_be_a_list():
    s, sent = _session()
    await s._handle_response_create(
        {"type": "response.create", "response": {"input": "nope"}}
    )
    assert sent[-1]["type"] == "error" and "input" in sent[-1]["error"]["message"]
    assert "response.created" not in _types(sent)


# ── noise reduction ──


def _tone_plus_noise(rate=24000, seconds=1.0, noise=300, tone=6000, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(int(rate * seconds)) / rate
    tone_sig = tone * np.sin(2 * np.pi * 440 * t)
    noise_sig = rng.normal(0, noise, len(t))
    return tone_sig, noise_sig


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))


@pytest.mark.parametrize("kind", ["near_field", "far_field"])
def test_noise_reducer_attenuates_noise_and_keeps_speech(kind):
    assert supported(kind) and not supported("x")
    rate = 24000
    nr = NoiseReducer(kind, rate)
    tone, noise = _tone_plus_noise(rate)
    quiet = np.clip(noise, -32768, 32767).astype("<i2")
    loud = np.clip(tone + noise, -32768, 32767).astype("<i2")
    # 0.5 s of room noise (learns the floor), then speech over the noise, chunked like a client
    stream = np.concatenate([quiet[: rate // 2], loud[: rate // 2]])
    out = []
    for i in range(0, len(stream), 2400):
        chunk = stream[i : i + 2400].tobytes()
        y = nr.process(chunk)
        assert len(y) == len(chunk)
        out.append(np.frombuffer(y, dtype="<i2"))
    out = np.concatenate(out)
    noise_in, noise_out = _rms(stream[: rate // 4]), _rms(out[rate // 4 : rate // 2])
    speech_in, speech_out = _rms(stream[-rate // 4 :]), _rms(out[-rate // 4 :])
    assert noise_out < 0.75 * noise_in
    assert speech_out > 0.8 * speech_in


def test_noise_reducer_removes_dc_and_handles_edge_cases():
    nr = NoiseReducer("near_field", 24000)
    dc = np.full(4800, 3000, dtype="<i2").tobytes()
    out = np.frombuffer(nr.process(dc), dtype="<i2")
    assert abs(float(out[-480:].mean())) < 300
    assert nr.process(b"") == b""
    assert len(nr.process(b"\x01\x02\x03")) == 2  # odd byte: whole samples only
    with pytest.raises(ValueError):
        NoiseReducer("bogus")


@pytest.mark.asyncio
async def test_append_applies_noise_reduction_when_configured():
    s, _ = _session()
    rate = 24000
    _, noise = _tone_plus_noise(rate, 0.6, noise=200)
    pcm = np.clip(noise, -32768, 32767).astype("<i2").tobytes()
    b64 = base64.b64encode(pcm).decode()
    s.session.turn_detection = None  # no VAD: only the buffer is under test
    await s._handle_input_audio_buffer_append({"audio": b64})
    plain = bytes(s._audio_buffer)
    assert plain == pcm  # off by default: untouched
    s._audio_buffer = bytearray()
    s.session.update({"input_audio_noise_reduction": {"type": "far_field"}})
    for i in range(0, len(pcm), 4800):
        await s._handle_input_audio_buffer_append(
            {"audio": base64.b64encode(pcm[i : i + 4800]).decode()}
        )
    denoised = np.frombuffer(bytes(s._audio_buffer), dtype="<i2")
    assert len(denoised) * 2 == len(pcm)
    assert _rms(denoised[-4800:]) < 0.5 * _rms(np.frombuffer(pcm, dtype="<i2")[-4800:])


# ── full response flow with a fake engine and a fake TTS ──


class _FakeEngine:
    is_loaded = True

    async def generate_stream(self, prompt=None, cancel_event=None, **kw):
        self.prompt = prompt
        for w in ("Hello", " there", "."):
            yield SimpleNamespace(
                token_text=w, finish_reason=None, prompt_tokens=7, completion_tokens=1
            )
        yield SimpleNamespace(
            token_text="", finish_reason="stop", prompt_tokens=7, completion_tokens=3
        )


@pytest.mark.asyncio
async def test_spoken_response_flow_ga():
    s, sent = _session("ga")
    engine = _FakeEngine()
    s._resolve_engine = lambda: engine  # type: ignore[method-assign]

    async def fake_tts(text, response_id, item_id, voice=None, out_fmt=None):
        delta = base64.b64encode(b"\0" * 4800).decode()
        await s.send_event(
            {"type": "response.audio.delta", "response_id": response_id, "delta": delta}
        )
        await s.send_event({"type": "response.audio.done", "response_id": response_id})

    s._synthesize_audio_response = fake_tts  # type: ignore[method-assign]
    s.conversation.add_item(
        rt.ConversationItem(
            "u1", "message", "user", [{"type": "input_text", "text": "hi"}]
        )
    )
    await s._handle_response_create(
        {"type": "response.create", "response": {"modalities": ["audio"]}}
    )
    await s._active_response
    types = _types(sent)
    parts = [e["part"] for e in sent if e["type"].startswith("response.content_part")]
    assert parts and all(p["type"] == "output_audio" for p in parts)
    assert parts[-1]["transcript"].strip() == "Hello there."
    assert "output_audio_buffer.started" in types
    assert types.index("output_audio_buffer.stopped") > types.index(
        "response.output_audio.done"
    )
    assert types.index("rate_limits.updated") == types.index("response.created") + 1
    done = next(e for e in sent if e["type"] == "response.done")
    assert done["response"]["status"] == "completed"
    item_done = next(e for e in sent if e["type"] == "response.output_item.done")
    assert item_done["item"]["content"][0]["type"] == "output_audio"


@pytest.mark.asyncio
async def test_response_input_reaches_the_prompt_and_is_out_of_band():
    s, sent = _session("ga")
    engine = _FakeEngine()
    s._resolve_engine = lambda: engine  # type: ignore[method-assign]
    s.conversation.add_item(
        rt.ConversationItem(
            "u1", "message", "user", [{"type": "input_text", "text": "BANANA"}]
        )
    )
    await s._handle_response_create(
        {
            "type": "response.create",
            "response": {
                "conversation": "none",
                "input": [_item("PINEAPPLE")],
                "modalities": ["text"],
            },
        }
    )
    await s._active_response
    assert "PINEAPPLE" in str(engine.prompt) and "BANANA" not in str(engine.prompt)
    assert len(s.conversation.items) == 1  # out of band: nothing appended
