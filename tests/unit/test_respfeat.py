"""SDK computer actions and incremental citation contracts (CPU only)."""

import asyncio
import json

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import TypeAdapter

from yunshu_gateway.anthropic_documents import Document, stream_citations
from yunshu_gateway.responses_client_tools import as_function, call_item
from yunshu_gateway.server_tools.responses_loop import input_item_to_messages


def encode(data):
    return f"event: {data['type']}\ndata: {json.dumps(data)}\n\n"


@pytest.mark.parametrize(
    "actions",
    [
        [{"type": "screenshot"}],
        [
            {"type": "click", "button": "left", "x": 1, "y": 2},
            {"type": "type", "text": "penguin"},
        ],
        [
            {"type": "double_click", "x": 1, "y": 2, "keys": ["SHIFT"]},
            {"type": "drag", "path": [{"x": 1, "y": 2}]},
            {"type": "move", "x": 3, "y": 4},
            {"type": "scroll", "x": 1, "y": 2, "scroll_x": 0, "scroll_y": 10},
            {"type": "keypress", "keys": ["CTRL", "A"]},
            {"type": "wait"},
        ],
    ],
)
def test_computer_call_sdk_and_roundtrip(actions):
    from openai.types.responses import ResponseComputerToolCall

    assert as_function({"type": "computer"}).name == "computer"
    call = call_item(
        {
            "id": "item_1",
            "call_id": "call_2",
            "status": "completed",
            "arguments": json.dumps({"actions": actions}),
        },
        {"type": "computer"},
    )
    TypeAdapter(ResponseComputerToolCall).validate_python(call)
    message = input_item_to_messages(call)[0]
    assert message["tool_calls"][0]["id"] == "call_2"
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
        "actions": actions
    }


@pytest.mark.parametrize(
    "action",
    [
        {"type": "launch"},
        {"type": "click", "x": 1},
        {"type": "type", "text": 123},
        {"type": "move", "x": True, "y": 1},
    ],
)
def test_invalid_computer_actions_fail_closed(action):
    with pytest.raises(HTTPException, match="invalid computer"):
        call_item(
            {
                "id": "i",
                "call_id": "c",
                "status": "completed",
                "arguments": json.dumps({"actions": [action]}),
            },
            {"type": "computer"},
        )


@pytest.mark.parametrize("split", range(1, 18))
def test_citation_split_and_immediate_text(split):
    from anthropic.types import RawMessageStreamEvent

    marker = "[[cite:0:0:4]]"
    advanced = []

    async def source():
        yield encode(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
        )
        yield encode(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Answer. "},
            }
        )
        advanced.append("after_text")
        for text in (marker[:split], marker[split:], " Next."):
            yield encode(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text},
                }
            )
        yield encode({"type": "content_block_stop", "index": 0})

    async def run():
        iterator = stream_citations(
            StreamingResponse(source()),
            [Document(0, None, "BLUE", "text", [(0, 4)], True)],
        )
        await anext(iterator)  # start
        first = await anext(iterator)
        assert "Answer." in first and not advanced
        return [first] + [chunk async for chunk in iterator]

    events = [json.loads(chunk.split("data: ")[1]) for chunk in asyncio.run(run())]
    for event in events:
        TypeAdapter(RawMessageStreamEvent).validate_python(event)
    deltas = [e["delta"] for e in events if e["type"] == "content_block_delta"]
    assert "".join(d.get("text", "") for d in deltas) == "Answer.  Next."
    assert (
        next(d["citation"] for d in deltas if d["type"] == "citations_delta")[
            "cited_text"
        ]
        == "BLUE"
    )


def test_bad_citation_emits_error_without_success_stop():
    async def source():
        yield encode(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Bad[[cite:0:0:999]]"},
            }
        )
        yield encode({"type": "message_stop"})

    async def run():
        return "".join(
            [
                chunk
                async for chunk in stream_citations(
                    StreamingResponse(source()),
                    [Document(0, None, "BLUE", "text", [(0, 4)], True)],
                )
            ]
        )

    result = asyncio.run(run())
    assert "event: error" in result and "message_stop" not in result


def wav_bytes():
    import io
    import wave

    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\0\0" * 2400)
    return out.getvalue()


def test_voice_enrollment_sdk_and_speech_reference(monkeypatch, tmp_path):
    pytest.importorskip("soundfile")
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from openai import OpenAI

    from yunshu_gateway.routers import voices
    from yunshu_gateway.routers.audio import TTSRequest

    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "true")
    monkeypatch.setenv("YUNSHU_FILES_DIR", str(tmp_path))
    app = FastAPI()
    app.include_router(voices.router, prefix="/v1")
    with TestClient(app) as http:
        consent = http.post(
            "/v1/audio/voice_consents",
            data={"name": "Me", "language": "en"},
            files={"recording": ("consent.wav", wav_bytes(), "audio/wav")},
        )
        assert consent.status_code == 200, consent.text
        sdk = OpenAI(api_key="test", base_url="http://testserver/v1", http_client=http)
        voice = sdk.audio.voices.create(
            name="My voice",
            consent=consent.json()["id"],
            audio_sample=("sample.wav", wav_bytes(), "audio/wav"),
        )
        assert voice.object == "audio.voice" and voice.type == "audio_sample"
        assert voice.id.startswith("voice_")
        from openai.types.file_object import FileObject

        from yunshu_gateway.files_store import get_store
        from yunshu_gateway.routers.files import openai_file

        for metadata in get_store().list():
            TypeAdapter(FileObject).validate_python(openai_file(metadata))
        request = TTSRequest(model="local", input="Hello", voice={"id": voice.id})
        engine = SimpleNamespace(
            _model=SimpleNamespace(generate=lambda text, ref_audio: None)
        )
        resolved = voices.resolve_voice(request, engine)
        from pathlib import Path

        assert Path(resolved.ref_audio).read_bytes().startswith(b"RIFF")
        with pytest.raises(HTTPException, match="cannot clone"):
            voices.resolve_voice(
                request,
                SimpleNamespace(_model=SimpleNamespace(generate=lambda text: None)),
            )
        bad = http.post(
            "/v1/audio/voices",
            data={"name": "bad", "consent": "cons_missing"},
            files={"audio_sample": ("sample.wav", wav_bytes(), "audio/wav")},
        )
        assert bad.status_code == 404
        bad = http.post(
            "/v1/audio/voices",
            data={"name": "bad", "consent": consent.json()["id"]},
            files={"audio_sample": ("sample.wav", b"broken", "audio/wav")},
        )
        assert bad.status_code == 400


def test_webrtc_two_peers_sdp_and_channel_cpu(monkeypatch):
    pytest.importorskip("aiortc")
    from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient
    from openai import AsyncOpenAI

    from yunshu_gateway import realtime_webrtc as rtc

    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "true")
    app = FastAPI()
    app.include_router(rtc.router)

    async def run():
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        channel = peer.createDataChannel("oai-events")
        from aiortc import AudioStreamTrack

        peer.addTrack(AudioStreamTrack())
        media = asyncio.Queue()
        peer.on("track", media.put_nowait)
        received = asyncio.Queue()
        channel.on("message", received.put_nowait)
        await peer.setLocalDescription(await peer.createOffer())
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as http:
                sdk = AsyncOpenAI(
                    api_key="x", base_url="http://test/v1", http_client=http
                )
                answer = await sdk.realtime.calls.create(
                    sdp=peer.localDescription.sdp,
                    session={
                        "type": "realtime",
                        "model": "local",
                        "output_modalities": ["text"],
                        "audio": {"input": {"turn_detection": None}},
                    },
                )
                await peer.setRemoteDescription(
                    RTCSessionDescription(sdp=answer.text, type="answer")
                )
                event = json.loads(await asyncio.wait_for(received.get(), 10))
                assert event["type"] == "session.created"
                remote = await asyncio.wait_for(media.get(), 5)
                frame = await asyncio.wait_for(remote.recv(), 5)
                assert frame.samples > 0
                for _ in range(100):
                    if any(
                        sock.session._audio_buffer
                        for _, _, sockets in rtc._calls.values()
                        for sock in sockets
                    ):
                        break
                    await asyncio.sleep(0.02)
                assert any(
                    sock.session._audio_buffer
                    for _, _, sockets in rtc._calls.values()
                    for sock in sockets
                )
                channel.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "session": {"instructions": "CPU loopback"},
                        }
                    )
                )
                while True:
                    event = json.loads(await asyncio.wait_for(received.get(), 5))
                    if event["type"] == "session.updated":
                        assert event["session"]["instructions"] == "CPU loopback"
                        break
        finally:
            await peer.close()
            await rtc.close_all()
        assert not rtc._calls

    asyncio.run(run())


def test_webrtc_pcm_track_and_bounded_queue_cpu():
    pytest.importorskip("aiortc")
    from types import SimpleNamespace

    from yunshu_gateway.realtime_webrtc import ChannelSocket, output_track

    async def run():
        output = output_track()
        socket = ChannelSocket(
            SimpleNamespace(bufferedAmount=0, send=lambda data: None), output
        )
        await socket.send_json(
            {"type": "response.output_audio.delta", "delta": "AQABAA=="}
        )
        frame = await output.recv()
        assert (
            frame.sample_rate == 24000 and bytes(frame.planes[0])[:4] == b"\x01\0\x01\0"
        )
        for _ in range(257):
            socket.feed("{}")
        assert socket.closed and socket.incoming.qsize() == 1
        output.stop()

    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_computer_sdk_testclient_roundtrip(stream, monkeypatch):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient
    from openai import OpenAI

    from tests.unit.test_agent_client_compat import fake_body
    from yunshu_gateway.responses_client_tools import create_client_tools, replay
    from yunshu_gateway.routers import responses as r

    app = FastAPI()
    seen = []

    async def inner(q, request):
        seen.extend(r._convert_to_messages(q))
        item = {
            "type": "function_call",
            "name": "computer",
            "id": "fc",
            "call_id": "call_screen",
            "status": "completed",
            "arguments": '{"actions":[{"type":"screenshot"}]}',
        }
        body = fake_body([item])
        return StreamingResponse(replay(body)) if q.stream else JSONResponse(body)

    @app.post("/v1/responses")
    async def route(q: r.ResponsesRequest, request: Request):
        return await create_client_tools(q, request, inner)

    with TestClient(app) as http:
        sdk = OpenAI(api_key="x", base_url="http://testserver/v1", http_client=http)
        response = sdk.responses.create(
            model="local",
            input="Show screen",
            tools=[{"type": "computer"}],
            tool_choice={"type": "computer"},
            stream=stream,
            store=False,
        )
        if stream:
            ev = list(response)
            assert ev[-1].type == "response.completed"
            output = ev[-1].response.output
            assert (
                next(e.item for e in ev if e.type == "response.output_item.added").type
                == "computer_call"
            )
        else:
            output = response.output
        assert output[0].actions[0].type == "screenshot"
        sdk.responses.create(
            model="local",
            input=[
                output[0].model_dump(exclude_none=True),
                {
                    "type": "computer_call_output",
                    "call_id": "call_screen",
                    "output": {
                        "type": "computer_screenshot",
                        "image_url": "data:image/png;base64,aGVsbG8=",
                    },
                },
            ],
            tools=[{"type": "computer"}],
            store=False,
        )
        tool = next(m for m in seen if m["role"] == "tool")
        assert (
            tool["tool_call_id"] == "call_screen"
            and tool["content"][0]["type"] == "image_url"
        )


def test_respfeat_probe_dry_run_and_fail_closed(tmp_path):
    import subprocess

    from scripts.research import respfeat_routes as probe

    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    assert (
        probe.main(
            [
                "--model",
                "/fake/model",
                "--tree-sha",
                tree,
                "--device",
                "m5",
                "--out",
                str(tmp_path / "out.json"),
                "--dry-run",
            ]
        )
        == 0
    )
    assert not probe.judge({"complete": True, "pass": True, "checks": {}})[0]
    checks = {name: {"status": "pass"} for name in probe.CHECKS}
    assert probe.judge({"complete": True, "pass": True, "checks": checks})[0]
    checks[probe.CHECKS[0]] = {"status": "fail", "detail": "fixture"}
    assert not probe.judge({"complete": True, "pass": True, "checks": checks})[0]


def test_respfeat_yv_stage_is_one_pinned_short_cell(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts.verify import stages

    cells = []

    class Executor:
        def run_cells(self, selected):
            cells.extend(selected)
            return {
                selected[0].key: SimpleNamespace(
                    ok=False, reason="fixture", evidence=None, job="fixture-job"
                )
            }

    root = __import__("pathlib").Path(__file__).resolve().parents[2]
    monkeypatch.setattr(stages, "_finish", lambda ctx, result: result)
    ctx = SimpleNamespace(
        cand=SimpleNamespace(path=root, key="candidate-sha"),
        model="/fake/tiny",
        py="python",
        cand_env={},
        exe=Executor(),
    )
    result = stages.stage_respfeat(ctx)
    assert not result.passed and len(cells) == 1
    assert cells[0].timeout_min == 10 and cells[0].device == "m5" and not cells[0].quiet
    assert "--tree-sha" in cells[0].argv and cells[0].retries == 0


def test_qwen_clone_requires_base_and_transcript(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from yunshu_gateway.routers import voices
    from yunshu_gateway.routers.audio import TTSRequest

    monkeypatch.setenv("YUNSHU_FILES_DIR", str(tmp_path))
    consent = voices.put_record(wav_bytes(), {"object": "audio.voice_consent"})
    voice = voices.put_record(
        wav_bytes(), {"object": "audio.voice", "_consent": consent["id"]}
    )
    request = TTSRequest(model="local", input="Hello", voice={"id": voice["id"]})
    config = SimpleNamespace(model_type="qwen3_tts", tts_model_type="base")
    engine = SimpleNamespace(
        _model=SimpleNamespace(config=config, generate=lambda text, ref_audio: None)
    )
    with pytest.raises(HTTPException, match="requires ref_text"):
        voices.resolve_voice(request, engine)
    resolved = voices.resolve_voice(
        request.model_copy(update={"ref_text": "My sample."}), engine
    )
    assert resolved.ref_text == "My sample."
    config.tts_model_type = "custom_voice"
    with pytest.raises(HTTPException, match="ignore reference"):
        voices.resolve_voice(request, engine)


def test_computer_screenshot_file_and_missing_id(monkeypatch, tmp_path):
    from yunshu_gateway.files_store import get_store

    monkeypatch.setenv("YUNSHU_FILES_DIR", str(tmp_path))
    stored = get_store().put(b"\x89PNG\r\n\x1a\n", "screen.png", mime_type="image/png")
    item = {
        "type": "computer_call_output",
        "call_id": "call_screen",
        "output": {"type": "computer_screenshot", "file_id": stored["id"]},
    }
    message = input_item_to_messages(item)[0]
    assert message["content"][0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    item["output"]["file_id"] = "file_missing"
    with pytest.raises(HTTPException) as error:
        input_item_to_messages(item)
    assert error.value.status_code == 404


def test_respfeat_probe_stops_at_first_failed_check():
    from types import SimpleNamespace

    from scripts.research import respfeat_routes as probe

    seen = []

    def fail(ctx):
        seen.append("first")
        raise ValueError("fixture")

    registry = {name: SimpleNamespace(fn=fail, routes=[]) for name in probe.CHECKS}
    rows, failures = probe.run_checks(
        SimpleNamespace(),
        SimpleNamespace(proc=SimpleNamespace(poll=lambda: None)),
        registry,
    )
    assert seen == ["first"] and len(rows) == len(failures) == 1


def test_webrtc_preserves_transcription_secret_scope(monkeypatch):
    import sys
    from types import SimpleNamespace

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from yunshu_gateway import realtime_secrets
    from yunshu_gateway import realtime_webrtc as rtc
    from yunshu_gateway.routers.realtime import RealtimeSession

    seen = []

    class Peer:
        def __init__(self, config):
            self.callbacks = {}
            self.localDescription = SimpleNamespace(sdp="v=0\n")

        def on(self, name):
            def register(fn):
                self.callbacks[name] = fn
                return fn

            return register

        def addTrack(self, track):
            pass

        async def setRemoteDescription(self, offer):
            pass

        async def createAnswer(self):
            return None

        async def setLocalDescription(self, answer):
            channel = SimpleNamespace(label="oai-events", on=lambda *args: None)
            self.callbacks["datachannel"](channel)
            await asyncio.sleep(0)

        async def close(self):
            pass

    async def run(session):
        seen.append(session._secret_kind)

    monkeypatch.setattr(RealtimeSession, "run", run)
    monkeypatch.setattr(rtc, "output_track", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "aiortc",
        SimpleNamespace(
            RTCConfiguration=lambda **kwargs: None,
            RTCPeerConnection=Peer,
            RTCSessionDescription=lambda **kwargs: None,
        ),
    )
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(AudioResampler=None))
    secret = realtime_secrets.mint({}, "transcription", 60)
    app = FastAPI()
    app.include_router(rtc.router)
    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/realtime/calls",
                content="v=0\n",
                headers={
                    "content-type": "application/sdp",
                    "authorization": "Bearer " + secret.value,
                },
            )
            assert response.status_code == 201, response.text
        assert seen == ["transcription"] and not rtc._calls
    finally:
        realtime_secrets.reset()


def test_respfeat_waits_for_pool_without_retrying_model_failures():
    from scripts.research.respfeat_routes import wait_for_port

    clock, attempts = [0.0], []

    def factory():
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("no free port in 18990-18996")
        return "server"

    def sleep(seconds):
        clock[0] += seconds

    assert wait_for_port(factory, now=lambda: clock[0], sleep=sleep) == "server"
    assert len(attempts) == 2 and clock[0] == 2
    with pytest.raises(RuntimeError, match="model failed"):
        wait_for_port(lambda: (_ for _ in ()).throw(RuntimeError("model failed")))
