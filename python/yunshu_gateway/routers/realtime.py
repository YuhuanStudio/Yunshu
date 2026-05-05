"""Yunshu Realtime API router — WebSocket-based real-time inference.

Implements OpenAI Realtime API compatible WebSocket protocol for:
- Real-time text generation with streaming
- Real-time audio input/output (TTS/ASR)
- Function calling during conversation
- Conversation item management

Protocol: JSON events over WebSocket, following OpenAI's realtime API structure.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["realtime"])


# ── Event types ──


class RealtimeEvent:
    """Event type constants for the realtime protocol."""

    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    CONVERSATION_CREATED = "conversation.created"
    RESPONSE_CREATED = "response.created"
    RESPONSE_AUDIO_DELTA = "response.audio.delta"
    RESPONSE_AUDIO_DONE = "response.audio.done"
    RESPONSE_AUDIO_TRANSCRIPT_DELTA = "response.audio_transcript.delta"
    RESPONSE_AUDIO_TRANSCRIPT_DONE = "response.audio_transcript.done"
    RESPONSE_TEXT_DELTA = "response.text.delta"
    RESPONSE_TEXT_DONE = "response.text.done"
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA = "response.function_call_arguments.delta"
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"
    RESPONSE_DONE = "response.done"
    INPUT_AUDIO_BUFFER_COMMITTED = "input_audio_buffer.committed"
    INPUT_AUDIO_BUFFER_SPEECH_STARTED = "input_audio_buffer.speech_started"
    INPUT_AUDIO_BUFFER_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
    ERROR = "error"


def _event(event_type: str, **kwargs) -> dict:
    """Build a realtime event dict."""
    return {"type": event_type, "event_id": f"evt_{uuid.uuid4().hex[:16]}", **kwargs}


# ── Session model ──


class Voice(str):
    ALLOY = "alloy"
    ECHO = "echo"
    SHIMMER = "shimmer"


class SessionConfig:
    """Mutable session state for a realtime connection."""

    def __init__(self):
        self.model = "default"
        self.modalities = ["text"]
        self.voice = Voice.ALLOY
        self.input_audio_format = "pcm16"
        self.output_audio_format = "pcm16"
        self.turn_detection = {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 500,
        }
        self.max_response_output_tokens = 4096
        self.temperature = 0.7
        self.tools: list[dict] = []

    def update(self, data: dict) -> list[str]:
        """Apply partial updates, return list of changed fields."""
        changed = []
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
                changed.append(key)
        return changed

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "modalities": self.modalities,
            "voice": self.voice,
            "input_audio_format": self.input_audio_format,
            "output_audio_format": self.output_audio_format,
            "turn_detection": self.turn_detection,
            "max_response_output_tokens": self.max_response_output_tokens,
            "temperature": self.temperature,
            "tools": self.tools,
        }


# ── Conversation model ──


class ConversationItem:
    """A single item in a realtime conversation."""

    def __init__(
        self,
        item_id: str,
        item_type: str,
        role: str | None = None,
        content: list[dict] | None = None,
    ):
        self.item_id = item_id
        self.item_type = item_type
        self.role = role
        self.content = content or []
        self.status = "incomplete"

    def to_dict(self) -> dict:
        d = {"id": self.item_id, "type": self.item_type, "status": self.status}
        if self.role:
            d["role"] = self.role
        if self.content:
            d["content"] = self.content
        return d


class Conversation:
    """Tracks items in a single realtime conversation."""

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.items: list[ConversationItem] = []

    def add_item(self, item: ConversationItem) -> None:
        self.items.append(item)

    def get_item(self, item_id: str) -> ConversationItem | None:
        return next((i for i in self.items if i.item_id == item_id), None)


# ── Handler ──


class RealtimeSession:
    """Handles a single realtime WebSocket connection."""

    # Audio chunk duration for streaming: 20ms at 24kHz, 16-bit mono = 960 bytes
    _AUDIO_CHUNK_BYTES = 960

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.session = SessionConfig()
        self.conversation = Conversation(f"conv_{uuid.uuid4().hex[:16]}")
        self._active_response: Optional[asyncio.Task] = None
        self._audio_buffer = bytearray()
        # VAD state
        self._vad_speaking = False
        self._vad_silence_start: float | None = None
        self._vad_speech_start_offset: int = 0

    async def send_event(self, event: dict) -> None:
        try:
            await self.ws.send_json(event)
        except Exception as e:
            logger.warning(f"Failed to send realtime event: {e}")

    async def run(self) -> None:
        """Main event loop for the WebSocket session."""
        # Send session.created
        await self.send_event(_event(
            RealtimeEvent.SESSION_CREATED,
            session=self.session.to_dict(),
        ))

        # Send conversation.created
        await self.send_event(_event(
            RealtimeEvent.CONVERSATION_CREATED,
            conversation={"id": self.conversation.conversation_id, "items": []},
        ))

        try:
            while True:
                raw = await self.ws.receive_text()
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    await self.send_event(_event(
                        RealtimeEvent.ERROR,
                        error={"message": "Invalid JSON", "type": "invalid_request_error"},
                    ))
                    continue

                await self._handle_event(event)
        except WebSocketDisconnect:
            logger.info("Realtime client disconnected")
        except Exception as e:
            logger.error(f"Realtime session error: {e}")
        finally:
            if self._active_response and not self._active_response.done():
                self._active_response.cancel()

    async def _handle_event(self, event: dict) -> None:
        """Dispatch incoming events to handlers."""
        event_type = event.get("type", "")
        handler = _EVENT_HANDLERS.get(event_type)
        if handler is None:
            await self.send_event(_event(
                RealtimeEvent.ERROR,
                error={"message": f"Unknown event type: {event_type}", "type": "invalid_request_error"},
            ))
            return
        try:
            await handler(self, event)
        except Exception as e:
            logger.error(f"Handler error for {event_type}: {e}")
            await self.send_event(_event(
                RealtimeEvent.ERROR,
                error={"message": str(e), "type": "server_error"},
            ))

    async def _handle_session_update(self, event: dict) -> None:
        """Handle session.update — modify session configuration.

        Supports updating turn_detection configuration for VAD control.
        The turn_detection dict can contain:
          - type: "server_vad" or null (disable VAD)
          - threshold: float in [0.0, 1.0] — RMS energy threshold for speech detection
          - prefix_padding_ms: int — padding before speech start
          - silence_duration_ms: int — silence duration before speech_stopped
        """
        session_data = event.get("session", {})
        changed = self.session.update(session_data)
        await self.send_event(_event(
            RealtimeEvent.SESSION_UPDATED,
            session=self.session.to_dict(),
        ))

    async def _handle_conversation_item_create(self, event: dict) -> None:
        """Handle conversation.item.create — add a message to conversation."""
        item_data = event.get("item", {})
        item_id = item_data.get("id", f"item_{uuid.uuid4().hex[:12]}")
        item = ConversationItem(
            item_id=item_id,
            item_type=item_data.get("type", "message"),
            role=item_data.get("role"),
            content=item_data.get("content", []),
        )
        item.status = "completed"
        self.conversation.add_item(item)
        await self.send_event(_event(
            "conversation.item.created",
            item=item.to_dict(),
        ))

    async def _handle_conversation_item_delete(self, event: dict) -> None:
        """Handle conversation.item.delete — remove an item from conversation.

        Removes the item with the specified item_id from the conversation.
        Sends conversation.item.deleted confirmation event.
        """
        item_id = event.get("item_id", "")
        if not item_id:
            await self.send_event(_event(
                RealtimeEvent.ERROR,
                error={"message": "Missing item_id", "type": "invalid_request_error"},
            ))
            return

        # Find and remove the item
        removed = False
        for i, item in enumerate(self.conversation.items):
            if item.item_id == item_id:
                self.conversation.items.pop(i)
                removed = True
                break

        if removed:
            await self.send_event(_event(
                "conversation.item.deleted",
                item_id=item_id,
            ))
        else:
            await self.send_event(_event(
                RealtimeEvent.ERROR,
                error={"message": f"Item {item_id} not found", "type": "invalid_request_error"},
            ))

    async def _handle_response_create(self, event: dict) -> None:
        """Handle response.create — trigger model generation.

        Supports modalities filter in the response config:
        - ["text"]: text-only response (no audio synthesis)
        - ["audio"]: audio-only response (text used internally but not streamed)
        - ["text", "audio"]: both text deltas and audio chunks
        If modalities is not specified, uses session-level modalities.
        """
        if self._active_response and not self._active_response.done():
            await self.send_event(_event(
                RealtimeEvent.ERROR,
                error={"message": "Response already in progress", "type": "server_error"},
            ))
            return

        response_id = f"resp_{uuid.uuid4().hex[:16]}"
        item_id = f"item_{uuid.uuid4().hex[:12]}"

        response_config = event.get("response", {})
        modalities = response_config.get("modalities", self.session.modalities)

        # Validate modalities
        valid_modalities = {"text", "audio"}
        if not isinstance(modalities, list):
            modalities = self.session.modalities
        modalities = [m for m in modalities if m in valid_modalities]
        if not modalities:
            modalities = ["text"]

        await self.send_event(_event(
            RealtimeEvent.RESPONSE_CREATED,
            response={
                "id": response_id,
                "object": "realtime.response",
                "status": "in_progress",
                "modalities": modalities,
            },
        ))

        self._active_response = asyncio.create_task(
            self._generate_response(response_id, item_id, modalities, response_config)
        )

    async def _generate_response(
        self,
        response_id: str,
        item_id: str,
        modalities: list[str],
        config: dict,
    ) -> None:
        """Generate a response and stream deltas back."""
        try:
            messages = self._build_messages()
            if not messages:
                await self.send_event(_event(
                    RealtimeEvent.ERROR,
                    error={"message": "No messages in conversation", "type": "server_error"},
                ))
                return

            engine = self._resolve_engine()
            if engine is None:
                await self.send_event(_event(
                    RealtimeEvent.ERROR,
                    error={"message": "No engine available", "type": "server_error"},
                ))
                return

            max_tokens = config.get("max_output_tokens", self.session.max_response_output_tokens)
            temperature = config.get("temperature", self.session.temperature)

            from yunshu_engine.batched_engine import BatchedEngine
            is_batched = isinstance(engine, BatchedEngine)

            full_text = ""

            if is_batched:
                async for output in engine.stream_chat(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ):
                    if output.new_text:
                        full_text += output.new_text
                        if "text" in modalities:
                            await self.send_event(_event(
                                RealtimeEvent.RESPONSE_TEXT_DELTA,
                                response_id=response_id,
                                item_id=item_id,
                                output_index=0,
                                content_index=0,
                                delta=output.new_text,
                            ))
                    if output.finish_reason:
                        break
            else:
                async for output in engine.generate_stream(
                    prompt=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ):
                    if output.token_text:
                        full_text += output.token_text
                        if "text" in modalities:
                            await self.send_event(_event(
                                RealtimeEvent.RESPONSE_TEXT_DELTA,
                                response_id=response_id,
                                item_id=item_id,
                                output_index=0,
                                content_index=0,
                                delta=output.token_text,
                            ))
                    if output.finish_reason:
                        break

            # text done
            if "text" in modalities:
                await self.send_event(_event(
                    RealtimeEvent.RESPONSE_TEXT_DONE,
                    response_id=response_id,
                    item_id=item_id,
                    output_index=0,
                    content_index=0,
                    text=full_text,
                ))

            # Audio output: synthesize text to speech if audio modality is requested
            if "audio" in modalities and full_text:
                await self._synthesize_audio_response(full_text, response_id, item_id)

            # Add assistant item to conversation
            assistant_item = ConversationItem(
                item_id=item_id,
                item_type="message",
                role="assistant",
                content=[{"type": "text", "text": full_text}],
            )
            assistant_item.status = "completed"
            self.conversation.add_item(assistant_item)

            await self.send_event(_event(
                "conversation.item.created",
                item=assistant_item.to_dict(),
            ))

            # response.done
            await self.send_event(_event(
                RealtimeEvent.RESPONSE_DONE,
                response={
                    "id": response_id,
                    "object": "realtime.response",
                    "status": "completed",
                    "output": [assistant_item.to_dict()],
                },
            ))

            # Record metrics
            try:
                from ..middleware.metrics import get_metrics
                get_metrics().record_inference()
            except Exception:
                pass

        except asyncio.CancelledError:
            await self.send_event(_event(
                RealtimeEvent.RESPONSE_DONE,
                response={
                    "id": response_id,
                    "object": "realtime.response",
                    "status": "cancelled",
                },
            ))
        except Exception as e:
            logger.error(f"Realtime generation error: {e}")
            await self.send_event(_event(
                RealtimeEvent.ERROR,
                error={"message": str(e), "type": "server_error"},
            ))
            await self.send_event(_event(
                RealtimeEvent.RESPONSE_DONE,
                response={
                    "id": response_id,
                    "object": "realtime.response",
                    "status": "failed",
                    "error": str(e),
                },
            ))
        finally:
            self._active_response = None

    async def _handle_response_cancel(self, event: dict) -> None:
        """Handle response.cancel — abort current generation."""
        if self._active_response and not self._active_response.done():
            self._active_response.cancel()

    async def _synthesize_audio_response(
        self, text: str, response_id: str, item_id: str,
    ) -> None:
        """Synthesize text to audio and stream audio deltas in 20ms chunks.

        Instead of sending the entire audio buffer at once, splits the TTS
        output into 20ms PCM chunks (960 bytes at 24kHz 16-bit mono) and
        sends each as a response.audio.delta event with base64-encoded PCM.
        Sends response.audio.done when all chunks are sent.
        """
        try:
            from ..engine import get_model_manager
            manager = get_model_manager()
            if manager is None:
                return

            for entry in manager.list_entries():
                if entry.is_loaded and hasattr(entry.engine, 'synthesize'):
                    result = await entry.engine.synthesize(
                        text, voice=self.session.voice,
                    )
                    audio_data = getattr(result, 'audio', None)
                    if audio_data is None:
                        audio_data = result.get('audio') if isinstance(result, dict) else None
                    if audio_data is not None:
                        import base64
                        raw_audio = audio_data if isinstance(audio_data, bytes) else bytes(audio_data)

                        # Stream in 20ms chunks
                        chunk_size = self._AUDIO_CHUNK_BYTES
                        offset = 0
                        while offset < len(raw_audio):
                            chunk = raw_audio[offset:offset + chunk_size]
                            chunk_b64 = base64.b64encode(chunk).decode()
                            await self.send_event(_event(
                                RealtimeEvent.RESPONSE_AUDIO_DELTA,
                                response_id=response_id,
                                item_id=item_id,
                                output_index=0,
                                content_index=0,
                                delta=chunk_b64,
                            ))
                            offset += chunk_size

                        await self.send_event(_event(
                            RealtimeEvent.RESPONSE_AUDIO_DONE,
                            response_id=response_id,
                            item_id=item_id,
                            output_index=0,
                            content_index=0,
                        ))
                    break
        except Exception as e:
            logger.error(f"TTS error in realtime session: {e}")

    async def _handle_input_audio_buffer_append(self, event: dict) -> None:
        """Handle input_audio_buffer.append — receive audio chunk.

        Accumulates base64-encoded audio data in the session's audio buffer.
        If server-side VAD is enabled, runs energy-based VAD on each chunk
        to detect speech start/stop events.
        """
        audio_b64 = event.get("audio", "")
        if not audio_b64:
            return
        import base64
        chunk = base64.b64decode(audio_b64)
        self._audio_buffer.extend(chunk)

        # Server-side VAD: energy-based detection
        turn_detection = self.session.turn_detection
        if turn_detection and turn_detection.get("type") == "server_vad":
            await self._run_vad(chunk)

    async def _run_vad(self, audio_chunk: bytes) -> None:
        """Energy-based Voice Activity Detection on an audio chunk.

        Computes RMS of the PCM audio chunk and compares against the VAD
        threshold. Fires speech_started/speech_stopped events at transitions.

        Args:
            audio_chunk: Raw PCM bytes (16-bit signed, mono, 24kHz assumed).
        """
        turn_detection = self.session.turn_detection
        threshold = turn_detection.get("threshold", 0.5)
        silence_duration_ms = turn_detection.get("silence_duration_ms", 500)

        # Compute RMS of 16-bit PCM samples
        import struct
        num_samples = len(audio_chunk) // 2
        if num_samples == 0:
            return

        # Parse as signed 16-bit integers
        samples = struct.unpack(f"<{num_samples}h", audio_chunk[:num_samples * 2])
        sum_sq = sum(s * s for s in samples)
        rms = (sum_sq / num_samples) ** 0.5

        # Normalize to [0, 1] range (16-bit max = 32768)
        rms_normalized = min(1.0, rms / 32768.0)

        now = time.monotonic()

        if rms_normalized >= threshold:
            # Speech detected
            if not self._vad_speaking:
                self._vad_speaking = True
                self._vad_speech_start_offset = len(self._audio_buffer)
                self._vad_silence_start = None
                await self.send_event(_event(
                    RealtimeEvent.INPUT_AUDIO_BUFFER_SPEECH_STARTED,
                    audio_start_ms=len(self._audio_buffer) // 48,  # approximate: 24kHz*2 bytes per ms
                ))
        else:
            # Silence detected
            if self._vad_speaking:
                if self._vad_silence_start is None:
                    self._vad_silence_start = now
                elif (now - self._vad_silence_start) * 1000 >= silence_duration_ms:
                    # Silence duration exceeded threshold — speech ended
                    self._vad_speaking = False
                    self._vad_silence_start = None
                    await self.send_event(_event(
                        RealtimeEvent.INPUT_AUDIO_BUFFER_SPEECH_STOPPED,
                        audio_end_ms=len(self._audio_buffer) // 48,
                    ))
            else:
                self._vad_silence_start = None

    async def _handle_input_audio_buffer_commit(self, event: dict) -> None:
        """Handle input_audio_buffer.commit — finalize and transcribe audio.

        Sends the accumulated audio buffer through ASR, then adds the
        transcribed text as a user conversation item.
        """
        await self.send_event(_event(
            RealtimeEvent.INPUT_AUDIO_BUFFER_COMMITTED,
        ))

        audio_data = getattr(self, '_audio_buffer', bytearray())
        if not audio_data:
            return

        # Transcribe via ASR engine
        try:
            import tempfile, os, numpy as np
            from ..engine import get_model_manager

            # Convert raw PCM to WAV for ASR engine
            import wave
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            with wave.open(tmp.name, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(24000)  # OpenAI realtime default
                wf.writeframes(bytes(audio_data))

            manager = get_model_manager()
            if manager is not None:
                for entry in manager.list_entries():
                    if entry.is_loaded and hasattr(entry.engine, 'transcribe'):
                        transcript = await entry.engine.transcribe(tmp.name)
                        text = transcript.get("text", "") if isinstance(transcript, dict) else str(transcript)
                        if text:
                            item = ConversationItem(
                                item_id=f"item_{uuid.uuid4().hex[:8]}",
                                item_type="message",
                                role="user",
                                content=[{"type": "text", "text": text}],
                            )
                            self.conversation.add_item(item)
                            await self.send_event(_event(
                                "conversation.item.created",
                                item=item.to_dict(),
                            ))
                        break
        except Exception as e:
            logger.error(f"ASR error in realtime session: {e}")
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

        self._audio_buffer = bytearray()

    def _build_messages(self) -> list[dict]:
        """Build messages list from conversation items."""
        messages = []
        for item in self.conversation.items:
            if item.item_type != "message" or not item.role:
                continue
            text_parts = []
            for content in item.content:
                if isinstance(content, dict) and content.get("type") == "text":
                    text_parts.append(content.get("text", ""))
                elif isinstance(content, str):
                    text_parts.append(content)
            if text_parts:
                messages.append({
                    "role": item.role,
                    "content": "\n".join(text_parts),
                })
        return messages

    def _resolve_engine(self):
        """Resolve the inference engine for this session."""
        from ..engine import get_engine, get_model_manager

        # Try multi-model
        manager = get_model_manager()
        if manager is not None:
            for entry in manager.list_entries():
                if entry.is_loaded and entry.engine is not None:
                    return entry.engine

        # Single engine
        engine = get_engine()
        if engine and engine.is_loaded:
            return engine

        return None


# ── Event dispatch ──

_EVENT_HANDLERS = {
    "session.update": RealtimeSession._handle_session_update,
    "conversation.item.create": RealtimeSession._handle_conversation_item_create,
    "conversation.item.delete": RealtimeSession._handle_conversation_item_delete,
    "response.create": RealtimeSession._handle_response_create,
    "response.cancel": RealtimeSession._handle_response_cancel,
    "input_audio_buffer.append": RealtimeSession._handle_input_audio_buffer_append,
    "input_audio_buffer.commit": RealtimeSession._handle_input_audio_buffer_commit,
}


# ── WebSocket endpoint ──


@router.websocket("/realtime")
async def realtime_endpoint(ws: WebSocket):
    """OpenAI-compatible Realtime API WebSocket endpoint."""
    await ws.accept()
    session = RealtimeSession(ws)
    await session.run()
