"""Optional aiortc transport for the shared Realtime session (no MLX here)."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import time
import uuid
from fractions import Fraction
from typing import Any

from fastapi import APIRouter, HTTPException, Request, WebSocketDisconnect
from fastapi.responses import Response

from . import realtime_secrets

logger = logging.getLogger(__name__)
router = APIRouter(tags=["realtime"])
_calls: dict[str, tuple[Any, list[asyncio.Task], list[ChannelSocket]]] = {}


class _AudioMessage(str):
    """An ``input_audio_buffer.append`` event: droppable under load (control events are not)."""


class _Inbox(asyncio.Queue):
    def evict_oldest_audio(self) -> bool:
        for index, message in enumerate(self._queue):
            if isinstance(message, _AudioMessage):
                del self._queue[index]
                return True
        return False


class ChannelSocket:
    """Bounded data-channel adapter consumed by RealtimeSession.

    When the session falls behind, the oldest queued audio frames are dropped (counted in
    ``dropped_audio``) so control events such as response.cancel still get through; the call is
    torn down only when a client floods the queue with control events alone.
    """

    def __init__(self, channel, output):
        self.channel, self.output = channel, output
        self.incoming = _Inbox(maxsize=256)
        self.closed = False
        self.session = None
        self.dropped_audio = 0

    def _note_dropped_audio(self):
        self.dropped_audio += 1
        if self.dropped_audio == 1 or self.dropped_audio % 1000 == 0:
            logger.warning(
                "realtime webrtc: session is behind, dropped %d audio frame(s)",
                self.dropped_audio,
            )

    def offer(self, message, *, audio=False):
        """Queue one inbound event without blocking. Returns False when it was dropped."""
        if self.closed:
            return False
        if audio:
            message = _AudioMessage(message)
        try:
            self.incoming.put_nowait(message)
            return True
        except asyncio.QueueFull:
            pass
        if audio:
            self._note_dropped_audio()  # the newest frame goes: keeps the queue order
            return False
        if self.incoming.evict_oldest_audio():
            self._note_dropped_audio()
            self.incoming.put_nowait(message)
            return True
        self.disconnect()  # 256 control events pending: a flood, not a slow session
        return False

    def feed(self, message):
        if self.closed:
            return
        if not isinstance(message, str) or len(message) > 1024 * 1024:
            self.disconnect()
            return
        audio = False
        try:
            event = json.loads(message)
            audio = event.get("type") == "input_audio_buffer.append"
            # RTP supplies/consumes audio; data-channel format overrides cannot change PCM.
            for config_key in ("session", "response"):
                config = event.get(config_key)
                if isinstance(config, dict):
                    for key in ("input_audio_format", "output_audio_format"):
                        if key in config:
                            config[key] = "pcm16"
                    audio = config.get("audio")
                    if isinstance(audio, dict):
                        for direction in ("input", "output"):
                            if (
                                isinstance(audio.get(direction), dict)
                                and "format" in audio[direction]
                            ):
                                audio[direction]["format"] = {
                                    "type": "audio/pcm",
                                    "rate": 24000,
                                }
            message = json.dumps(event)
        except (ValueError, AttributeError):
            pass  # The shared session emits its ordinary invalid-JSON error.
        self.offer(message, audio=audio)

    def disconnect(self):
        self.closed = True
        # Wake receive_text even if a client filled the queue before closing.
        while not self.incoming.empty():
            self.incoming.get_nowait()
        self.incoming.put_nowait(None)

    async def receive_text(self):
        message = await self.incoming.get()
        if message is None:
            raise WebSocketDisconnect()
        return message

    async def send_json(self, event):
        if self.closed:
            raise WebSocketDisconnect()
        if event["type"] == "response.output_audio.delta":
            await self.output.append(base64.b64decode(event["delta"], validate=True))
            return  # WebRTC audio travels on the media track.
        if event["type"] == "output_audio_buffer.cleared":
            self.output.clear()
        while self.channel.bufferedAmount > 1024 * 1024:
            if self.closed:
                raise WebSocketDisconnect()
            await asyncio.sleep(0.01)
        self.channel.send(json.dumps(event))


def output_track():
    from aiortc import AudioStreamTrack
    from av import AudioFrame

    class Output(AudioStreamTrack):
        def __init__(self):
            super().__init__()
            self.buffer = bytearray()
            self.pts = 0
            self.started = None
            self.space = asyncio.Event()
            self.space.set()

        async def append(self, pcm):
            # Bound output to 2 seconds; backpressure instead of dropping speech.
            for offset in range(0, len(pcm), 960):
                while len(self.buffer) >= 96000:
                    self.space.clear()
                    await self.space.wait()
                self.buffer.extend(pcm[offset : offset + 960])

        def clear(self):
            self.buffer.clear()
            self.space.set()

        async def recv(self):
            if self.started is None:
                self.started = time.monotonic()
            await asyncio.sleep(
                max(0, self.started + self.pts / 24000 - time.monotonic())
            )
            pcm = bytes(self.buffer[:960])
            del self.buffer[:960]
            self.space.set()
            frame = AudioFrame(format="s16", layout="mono", samples=480)
            frame.planes[0].update(pcm.ljust(960, b"\0"))
            frame.sample_rate, frame.pts, frame.time_base = (
                24000,
                self.pts,
                Fraction(1, 24000),
            )
            self.pts += 480
            return frame

        def stop(self):
            self.clear()
            super().stop()

    return Output()


async def close_call(identifier):
    call = _calls.pop(identifier, None)
    if call is None:
        return
    pc, tasks, sockets = call
    for socket in sockets:
        socket.disconnect()
    current = asyncio.current_task()
    for task in tasks:
        if task is not current:
            task.cancel()
    await pc.close()
    for task in tasks:
        if task is not current:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


async def close_all():
    await asyncio.gather(*(close_call(key) for key in list(_calls)))


@router.post("/v1/realtime/calls")
async def create_call(request: Request):
    from .routers.models import _check_permission

    token = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    secret = realtime_secrets.lookup(token)
    if secret is None:
        _check_permission(request, "can_infer")
    try:
        from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
        from av import AudioResampler
    except ImportError as exc:
        raise HTTPException(
            503,
            "WebRTC requires yunshu[webrtc]; use WS /v1/realtime with PCM audio events instead",
        ) from exc
    if len(_calls) >= 16:
        raise HTTPException(429, "Too many active WebRTC calls")
    content_type = request.headers.get("content-type", "").split(";")[0]
    config = dict(secret.config) if secret else {}
    if content_type == "application/sdp":
        raw = await request.body()
        if len(raw) > 128 * 1024:
            raise HTTPException(413, "SDP exceeds 128 KiB")
        try:
            sdp = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(400, "Invalid SDP encoding") from exc
    elif content_type == "multipart/form-data":
        async with request.form() as form:
            raw_sdp = form.get("sdp")
            if not isinstance(raw_sdp, str):
                raise HTTPException(400, "sdp must be a string")
            sdp = raw_sdp
            try:
                session_json = form.get("session", "{}")
                if not isinstance(session_json, str):
                    raise ValueError()
                body = json.loads(session_json)
                if not isinstance(body, dict):
                    raise ValueError()
                # An ephemeral key fixes its session settings; it cannot be overridden.
                if not secret:
                    config.update(realtime_secrets.internal_config(body))
            except (TypeError, ValueError) as exc:
                raise HTTPException(400, "session must be a JSON object") from exc
    else:
        raise HTTPException(415, "Use application/sdp or multipart sdp + session")
    if not isinstance(sdp, str) or len(sdp) > 128 * 1024 or not sdp.startswith("v=0"):
        raise HTTPException(400, "Invalid SDP offer")
    # The bridge's internal audio is PCM24k regardless of the negotiated RTP codec.
    config.update(input_audio_format="pcm16", output_audio_format="pcm16")
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    identifier = "rtc_" + uuid.uuid4().hex
    tasks: list[asyncio.Task] = []
    sockets: list[ChannelSocket] = []
    _calls[identifier] = (pc, tasks, sockets)
    output = output_track()
    pc.addTrack(output)
    opened = asyncio.Event()

    @pc.on("datachannel")
    def on_channel(channel):
        if channel.label != "oai-events" or sockets:
            channel.close()
            return
        from .routers.realtime import RealtimeSession

        socket = ChannelSocket(channel, output)
        sockets.append(socket)
        session = RealtimeSession(socket, dialect="ga")
        socket.session = session
        session.session.update(config)
        if secret:
            session._secret_kind = secret.kind
            session.session.id = secret.session_id
        channel.on("message", socket.feed)
        channel.on("close", socket.disconnect)

        async def run():
            opened.set()
            try:
                await session.run()
            finally:
                await close_call(identifier)

        tasks.append(asyncio.create_task(run()))

    @pc.on("track")
    def on_track(track):
        if track.kind != "audio":
            return

        async def receive():
            resampler = AudioResampler(format="s16", layout="mono", rate=24000)
            try:
                await opened.wait()
                while True:
                    frame = await track.recv()
                    for converted in resampler.resample(frame):
                        pcm = bytes(converted.planes[0])[: converted.samples * 2]
                        # Serial, bounded queue; a full queue drops this frame (counted), never the call.
                        sockets[0].offer(
                            json.dumps(
                                {
                                    "type": "input_audio_buffer.append",
                                    "audio": base64.b64encode(pcm).decode(),
                                }
                            ),
                            audio=True,
                        )
            except Exception:
                await close_call(identifier)

        tasks.append(asyncio.create_task(receive()))

    @pc.on("connectionstatechange")
    async def state_changed():
        if pc.connectionState in ("failed", "closed"):
            await close_call(identifier)

    async def expire():
        try:
            await asyncio.wait_for(opened.wait(), 60)
            await asyncio.sleep(3600)
        except TimeoutError:
            pass
        await close_call(identifier)

    tasks.append(asyncio.create_task(expire()))
    try:
        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
        await asyncio.wait_for(pc.setLocalDescription(await pc.createAnswer()), 15)
    except Exception as exc:
        await close_call(identifier)
        raise HTTPException(400, "Cannot negotiate SDP offer") from exc
    return Response(
        pc.localDescription.sdp,
        status_code=201,
        media_type="application/sdp",
        headers={"Location": f"/v1/realtime/calls/{identifier}"},
    )
