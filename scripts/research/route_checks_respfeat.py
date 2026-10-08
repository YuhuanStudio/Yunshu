"""Responses computer / citation / enrollment / WebRTC served-route probes."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import wave

from route_checks import Ctx, check, expect
from route_checks_agent_compat import _events


@check("respfeat-computer", "POST /v1/responses", served=True)
def computer(c: Ctx):
    events = _events(
        c,
        "/v1/responses",
        {
            "model": c.model,
            "input": 'Use the computer tool to take a screenshot. The actions must be exactly [{"type":"screenshot"}].',
            "tools": [{"type": "computer"}],
            "tool_choice": {"type": "computer"},
            "stream": True,
            "store": True,
            "max_output_tokens": 128,
            "temperature": 0,
            "enable_thinking": False,
        },
    )
    expect(events[-1]["type"] == "response.completed", str(events[-1]))
    item = next(
        i for i in events[-1]["response"]["output"] if i["type"] == "computer_call"
    )
    expect(item["actions"] == [{"type": "screenshot"}], str(item))
    expect(
        any(e.get("item", {}).get("type") == "computer_call" for e in events),
        str(events),
    )
    follow = c.oa.responses.create(
        model=c.model,
        previous_response_id=events[-1]["response"]["id"],
        input=[
            {
                "type": "computer_call_output",
                "call_id": item["call_id"],
                "output": {
                    "type": "computer_screenshot",
                    "image_url": "data:image/png;base64,"
                    + base64.b64encode(__import__("route_checks")._png()).decode(),
                },
            }
        ],
        instructions="Say BLUE in your answer.",
        max_output_tokens=64,
        temperature=0,
        extra_body={"enable_thinking": False},
    )
    expect(follow.status == "completed" and follow.output_text, str(follow))


@check("respfeat-citations", "POST /v1/messages", served=True)
def citations(c: Ctx):
    events = _events(
        c,
        "/v1/messages",
        {
            "model": c.model,
            "max_tokens": 96,
            "stream": True,
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": "BLUE",
                            },
                            "citations": {"enabled": True},
                        },
                        {
                            "type": "text",
                            "text": "Copy exactly, nothing else: BLUE[[cite:0:0:4]]",
                        },
                    ],
                }
            ],
        },
    )
    expect(events[-1]["type"] == "message_stop", str(events[-1]))
    cited = [
        e["delta"]["citation"]
        for e in events
        if e.get("delta", {}).get("type") == "citations_delta"
    ]
    expect(cited and cited[0]["cited_text"] == "BLUE", str(events))
    text = "".join(e.get("delta", {}).get("text", "") for e in events)
    expect("[[cite:" not in text, text)


def sample():
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"\0\0" * 2400)
    return out.getvalue()


@check(
    "respfeat-voices",
    "POST /v1/audio/voices",
    "POST /v1/audio/voice_consents",
    "GET /v1/audio/voices",
    served=True,
)
def voices(c: Ctx):
    # Enrollment is CPU-only; speech cloning needs a separate TTS checkpoint.
    consent = c.req(
        "POST",
        "/v1/audio/voice_consents",
        data={"name": "Route check", "language": "en"},
        files={"recording": ("consent.wav", sample(), "audio/wav")},
    )
    expect(consent.status_code == 200, consent.text[:500])
    voice = c.oa.audio.voices.create(
        name="Route voice",
        consent=consent.json()["id"],
        audio_sample=("sample.wav", sample(), "audio/wav"),
    )
    expect(voice.object == "audio.voice" and voice.type == "audio_sample", str(voice))
    listed = c.req("GET", "/v1/audio/voices")
    expect(any(v["id"] == voice.id for v in listed.json()["data"]), listed.text[:500])
    # Clean both metadata and recordings through the bounded Files store.
    for identifier in (voice.id, consent.json()["id"]):
        fid = "file_" + identifier.split("_", 1)[1]
        content = c.req("GET", f"/v1/files/{fid}/content")
        expect(content.status_code == 200, content.text[:100])
        recording = content.json()["_recording"]
        for file_id in (recording, fid):
            deleted = c.req("DELETE", f"/v1/files/{file_id}")
            expect(deleted.status_code == 200, deleted.text[:100])


@check("respfeat-webrtc", "POST /v1/realtime/calls", served=True)
def webrtc(c: Ctx):
    from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription

    async def run():
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        dc = pc.createDataChannel("oai-events")
        pc.addTransceiver("audio", direction="sendrecv")
        received = asyncio.Queue()
        dc.on("message", received.put_nowait)
        try:
            await pc.setLocalDescription(await pc.createOffer())
            answer = c.oa.realtime.calls.create(
                sdp=pc.localDescription.sdp,
                session={
                    "type": "realtime",
                    "model": c.model,
                    "output_modalities": ["text"],
                    "audio": {"input": {"turn_detection": None}},
                },
            )
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=answer.text, type="answer")
            )
            while True:
                event = json.loads(await asyncio.wait_for(received.get(), 15))
                if event["type"] == "session.created":
                    break
            dc.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Say BLUE."}],
                        },
                    }
                )
            )
            dc.send(
                json.dumps(
                    {
                        "type": "response.create",
                        "response": {
                            "output_modalities": ["text"],
                            "max_output_tokens": 64,
                        },
                    }
                )
            )
            text = ""
            while True:
                event = json.loads(await asyncio.wait_for(received.get(), 120))
                expect(event["type"] != "error", str(event))
                if event["type"] == "response.output_text.delta":
                    text += event["delta"]
                if event["type"] == "response.done":
                    expect(
                        event["response"]["status"] == "completed" and text, str(event)
                    )
                    break
        finally:
            await pc.close()

    asyncio.run(run())
