from __future__ import annotations

"""Yunshu Realtime API router — WebSocket-based real-time inference.

Implements OpenAI Realtime API compatible WebSocket protocol for:
- Real-time text generation with streaming
- Real-time audio input/output (TTS/ASR)
- Function calling during conversation
- Conversation item management

Protocol: JSON events over WebSocket, following OpenAI's realtime API structure.
"""

import asyncio
import contextlib
import json
import logging
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(tags=["realtime"])


# bound the per-connection input audio buffer. It is only drained by
# commit/clear/auto-commit, so a client that streams input_audio_buffer.append forever
# WITHOUT committing (legal when turn_detection is null) — or speaks continuously so
# server-VAD never hits the silence window — grows it without limit (and g711 input is
# 2x-amplified on decode). Any can_infer key can reach the WS, making this a memory-
# growth/DoS vector (the 36GB-Mac SIGABRT class). Default ~10MB ≈ 3.5 min @ 24kHz/16-bit
# — generous for a real utterance, bounded against runaway. Env-tunable.
def _max_input_audio_bytes() -> int:
    import os

    try:
        v = int(os.environ.get("YUNSHU_REALTIME_MAX_INPUT_AUDIO_BYTES", "10485760"))
        return v if v > 0 else 10485760
    except (TypeError, ValueError):
        return 10485760


def _max_conversation_items() -> int:
    # cap conversation history. The input-audio buffer was bounded for exactly
    # this DoS class but left conversation.items unbounded — every turn appends user +
    # assistant (+ function_call) items and _build_messages replays the WHOLE history into
    # every prompt, so a client flooding conversation.item.create grows RSS without bound and
    # balloons per-prompt cost. Keep the most recent N (oldest evicted, see Conversation.add_item).
    import os

    try:
        v = int(os.environ.get("YUNSHU_REALTIME_MAX_CONVERSATION_ITEMS", "1000"))
        return v if v > 0 else 1000
    except (TypeError, ValueError):
        return 1000


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
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED = (
        "conversation.item.input_audio_transcription.completed"
    )
    ERROR = "error"


def _event(event_type: str, **kwargs) -> dict:
    """Build a realtime event dict."""
    return {"type": event_type, "event_id": f"evt_{uuid.uuid4().hex[:16]}", **kwargs}


def _omni_realtime_enabled() -> bool:
    """True iff the native-omni realtime path is opted in. Requires a configured
    omni model (YUNSHU_OMNI_MODEL) AND an explicit YUNSHU_REALTIME_OMNI flag, so
    the default realtime behaviour (ASR→LLM→TTS cascade) is never changed silently."""
    import os

    if not os.environ.get("YUNSHU_OMNI_MODEL"):
        return False
    return os.environ.get("YUNSHU_REALTIME_OMNI", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _messages_to_omni_prompt(messages: list[dict]) -> str:
    """Flatten the chat-message list into a single prompt for the omni model.

    OmniEngine.stream takes one user turn (+ optional media), so we prepend any
    system instruction (the being's persona) to the latest user text. Full
    multi-turn omni context is a follow-up — this preserves persona + last turn,
    which is what a voice reply needs most."""
    sys_parts = [
        m["content"] for m in messages if m.get("role") == "system" and m.get("content")
    ]
    user_parts = [
        m["content"] for m in messages if m.get("role") == "user" and m.get("content")
    ]
    if not user_parts:
        return ""
    last_user = user_parts[-1]
    if sys_parts:
        return f"{' '.join(sys_parts)}\n\n{last_user}".strip()
    return last_user


def _f32_to_pcm16_bytes(wav_f32) -> bytes:
    """float32 [-1,1] mono ndarray → int16 little-endian PCM bytes (24 kHz)."""
    import numpy as np

    arr = np.asarray(wav_f32, dtype=np.float32).reshape(-1)
    return (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def _omni_system_text(messages: list[dict]) -> str:
    """The system instruction (persona) only — used as the text turn when the
    user's RAW audio is the actual query (native speech-in)."""
    return " ".join(
        m["content"] for m in messages if m.get("role") == "system" and m.get("content")
    ).strip()


def _write_pcm16_wav(pcm: bytes, rate: int) -> str:
    """Write mono int16 PCM bytes to a temp WAV file; return its path."""
    import tempfile
    import wave

    fd, path = tempfile.mkstemp(suffix=".wav")
    import os as _os

    _os.close(fd)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return path


# ── Optional neural VAD (Silero via mlx-audio) ──────────────────────────────
# Default turn detection is the energy-RMS heuristic in _run_vad (zero deps). Set
# YUNSHU_REALTIME_VAD=silero to use Silero instead: far fewer false triggers, and
# it makes the OpenAI `threshold` config a real speech-probability (the RMS path
# has to fudge it as threshold*0.05). Lazy-loaded once, shared across sessions.
SILERO_VAD_RATE = 16000  # we resample input to 16 kHz for Silero
_silero_vad = None  # the loaded model (or False if load failed)
_SILERO_REPO = "mlx-community/silero-vad"


def _silero_vad_enabled() -> bool:
    import os

    return os.environ.get("YUNSHU_REALTIME_VAD", "").strip().lower() == "silero"


def _get_silero_vad():
    """Lazy-load the Silero VAD model. Returns the model, or None if disabled or
    the load fails (caller falls back to the energy heuristic)."""
    global _silero_vad
    if _silero_vad is None:
        if not _silero_vad_enabled():
            return None
        try:
            import os

            import mlx_audio.vad as _vad

            _silero_vad = _vad.load(
                os.environ.get("YUNSHU_REALTIME_VAD_MODEL", _SILERO_REPO)
            )
            logger.info("Realtime VAD: Silero loaded (%s).", _SILERO_REPO)
        except Exception:
            logger.warning(
                "Silero VAD load failed — falling back to energy VAD", exc_info=True
            )
            _silero_vad = False  # sentinel: don't retry every chunk
    return _silero_vad or None


def _strip_wav_header(data: bytes) -> bytes:
    """Return raw PCM, stripping a leading RIFF/WAVE header if present.

    synthesize_stream() prepends a 44-byte WAV header to its FIRST chunk and
    synthesize() returns a full WAV file, but the OpenAI Realtime protocol requires
    response.audio.delta payloads to be RAW PCM16. Forwarding the header made the
    client decode it as garbage samples AND — since 44 isn't a multiple of 2 — misalign
    every 16-bit sample boundary for the whole turn (white-noise playback).
    """
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        off = 12
        while off + 8 <= len(data):
            cid = data[off : off + 4]
            sz = int.from_bytes(data[off + 4 : off + 8], "little")
            if cid == b"data":
                return data[off + 8 :]
            off += 8 + sz + (sz & 1)
        return data[44:]  # fallback: standard 44-byte PCM header
    return data


# ── Session model ──


class Voice(str):
    ALLOY = "alloy"
    ECHO = "echo"
    SHIMMER = "shimmer"


def _coerce_max_response_tokens(value, fallback: int) -> int:
    """Coerce a Realtime `max_response_output_tokens` to an int the engine can use.

    OpenAI allows the literal string "inf" (and clients may send int/None); a raw str
    reaching stream_chat(max_tokens=) crashes mlx-lm (it compares the token counter
    against a str). "inf"/None → a large finite cap; other non-ints → int() or the
    fallback. Shared by SessionConfig.update and the per-response override path so they
    can't drift (the response.create path used to skip this entirely)."""
    if value in ("inf", None):
        return 1 << 20
    if isinstance(value, bool) or not isinstance(value, int):
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback
    return value


class SessionConfig:
    """Mutable session state for a realtime connection."""

    # Supported audio formats
    SUPPORTED_AUDIO_FORMATS = {"pcm16", "g711_ulaw", "g711_alaw"}

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
        # tool_choice — "auto" (model decides), "none" (forbid tool calls this
        # turn → plain text), "required", or a {"type":"function","function":{"name":…}} dict.
        # MUST be defined here so update()'s hasattr gate accepts a session.update for it.
        self.tool_choice: str | dict = "auto"
        self.instructions: str = ""  # System instructions for response.create
        # Sampling params read by _generate_response via getattr. They MUST be
        # defined here or update()'s `hasattr` gate silently drops a client's
        # session.update for them, locking them to the engine defaults forever.
        self.top_p = 1.0
        self.top_k = 0
        self.min_p = 0.0
        self.repetition_penalty = 1.0
        self.frequency_penalty = 0.0
        self.presence_penalty = 0.0
        self.logit_bias: dict | None = None
        self.enable_thinking = False
        self.seed: int | None = None

    def update(self, data: dict) -> list[str]:
        """Apply partial updates, return list of changed fields."""
        changed = []
        for key, value in data.items():
            if hasattr(self, key):
                if key in ("input_audio_format", "output_audio_format"):
                    if value not in self.SUPPORTED_AUDIO_FORMATS:
                        continue
                if key == "voice":
                    valid_voices = {"alloy", "echo", "shimmer"}
                    if value not in valid_voices:
                        continue
                if key == "tool_choice":
                    # a string mode or a named-function dict; reject other types
                    # so a garbage value can't be stored and later mis-compared.
                    if not (
                        value in ("auto", "none", "required") or isinstance(value, dict)
                    ):
                        continue
                if key == "turn_detection":
                    # turn_detection must be a dict (server_vad config) or
                    # null (disable VAD). The old guard validated ONLY when value was a
                    # dict, so a non-dict (e.g. the bare string "server_vad", or a list)
                    # skipped validation and was stored verbatim → every later
                    # input_audio_buffer.append ran turn_detection.get("type") on a str
                    # → AttributeError → a 500 on EVERY append for the rest of the
                    # session (VAD effectively dead + an opaque per-append error storm).
                    if value is not None and not isinstance(value, dict):
                        continue
                    if isinstance(value, dict):
                        td_type = value.get("type")
                        if td_type is not None and td_type not in ("server_vad", None):
                            continue
                if key == "modalities":
                    # validate modalities like voice/format/turn_detection — a
                    # client could otherwise set a non-list or ["video"] and to_dict() would
                    # echo the garbage back (response.create re-sanitizes, but the session
                    # state was still corrupt). Accept only a list drawn from {text,audio}.
                    if (
                        not isinstance(value, list)
                        or not value
                        or any(m not in ("text", "audio") for m in value)
                    ):
                        continue
                if key == "max_response_output_tokens":
                    # Coerce to an int (see _coerce_max_response_tokens). A non-int that
                    # can't convert → skip the update (keep the current value): use a
                    # sentinel fallback so we can tell "coercion failed" apart from a
                    # value that legitimately equals the current one.
                    _SENTINEL = object()
                    _coerced = _coerce_max_response_tokens(value, _SENTINEL)
                    if _coerced is _SENTINEL:
                        continue
                    value = _coerced
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
            "instructions": self.instructions,
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
        call_id: str | None = None,
        name: str | None = None,
        arguments: str | None = None,
        output: str | None = None,
    ):
        self.item_id = item_id
        self.item_type = item_type
        self.role = role
        self.content = content or []
        self.status = "incomplete"
        # function-call fields so tool calls + tool RESULTS survive in history.
        # function_call: name/arguments/call_id; function_call_output: call_id/output.
        self.call_id = call_id
        self.name = name
        self.arguments = arguments
        self.output = output

    def to_dict(self) -> dict:
        d = {"id": self.item_id, "type": self.item_type, "status": self.status}
        if self.role:
            d["role"] = self.role
        if self.content:
            d["content"] = self.content
        if self.call_id:
            d["call_id"] = self.call_id
        if self.name:
            d["name"] = self.name
        if self.arguments is not None:
            d["arguments"] = self.arguments
        if self.output is not None:
            d["output"] = self.output
        return d


class Conversation:
    """Tracks items in a single realtime conversation."""

    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.items: list[ConversationItem] = []

    def add_item(
        self, item: ConversationItem, previous_item_id: str | None = None
    ) -> None:
        # honor previous_item_id — insert AFTER the referenced item so
        # out-of-order client inserts land in the right chronological position
        # (the OpenAI Realtime spec semantics). Falls back to append when the ref
        # is missing/None.
        if previous_item_id:
            for idx, it in enumerate(self.items):
                if it.item_id == previous_item_id:
                    self.items.insert(idx + 1, item)
                    self._trim()
                    return
        self.items.append(item)
        self._trim()

    def _trim(self) -> None:
        # bound history to the most recent N items (FIFO eviction of the oldest)
        # so a conversation.item.create flood / long session can't grow memory or prompt
        # cost without limit. Mirrors the audio-buffer cap.
        _cap = _max_conversation_items()
        if len(self.items) > _cap:
            del self.items[: len(self.items) - _cap]

    def has_item(self, item_id: str) -> bool:
        return any(i.item_id == item_id for i in self.items)

    def get_item(self, item_id: str) -> ConversationItem | None:
        return next((i for i in self.items if i.item_id == item_id), None)


# ── Handler ──


class RealtimeSession:
    """Handles a single realtime WebSocket connection."""

    # Audio chunk duration for streaming: 20ms at 24kHz, 16-bit mono = 960 bytes
    _AUDIO_CHUNK_BYTES = 960

    # G.711 μ-law decoding table (8-bit → 16-bit linear PCM)
    _ULAW_TABLE: list[int] | None = None
    # G.711 A-law decoding table (8-bit → 16-bit linear PCM)
    _ALAW_TABLE: list[int] | None = None

    @classmethod
    def _get_ulaw_table(cls) -> list[int]:
        if cls._ULAW_TABLE is None:
            cls._ULAW_TABLE = cls._build_ulaw_table()
        return cls._ULAW_TABLE

    @classmethod
    def _get_alaw_table(cls) -> list[int]:
        if cls._ALAW_TABLE is None:
            cls._ALAW_TABLE = cls._build_alaw_table()
        return cls._ALAW_TABLE

    @staticmethod
    def _build_ulaw_table() -> list[int]:
        """Build μ-law to linear PCM decoding table (ITU-T G.711)."""
        table = [0] * 256
        for i in range(256):
            # Invert all bits (μ-law is transmitted complemented)
            val = ~i & 0xFF
            sign = (val & 0x80) >> 7
            exponent = (val >> 4) & 0x07
            mantissa = val & 0x0F
            linear = (mantissa << 3 | 0x84) << exponent
            linear -= 0x84
            # negate when the sign bit is SET (not clear). The old `if sign == 0`
            # inverted EVERY nonzero sample's polarity vs ITU-T G.711 — self-consistent
            # (encode+decode share the table, so loopback + the "37dB" self-test pass),
            # but garbled for any real telephony/SIP μ-law client (interop SNR ~-6 dB,
            # full-scale inversion). Verified 256/256 against ITU-T (audioop) after this flip.
            if sign:
                linear = -linear
            table[i] = max(-32768, min(32767, linear))
        return table

    @staticmethod
    def _build_alaw_table() -> list[int]:
        """Build A-law to linear PCM decoding table (ITU-T G.711)."""
        table = [0] * 256
        for i in range(256):
            val = i ^ 0x55  # A-law is XOR'd with 0x55
            sign = (val & 0x80) >> 7
            exponent = (val >> 4) & 0x07
            mantissa = val & 0x0F
            if exponent == 0:
                linear = (mantissa << 4) + 8
            else:
                # include the +8 half-step bias (0x108, not 0x100) so the decoded
                # value lands at the quantization-interval MIDPOINT per ITU-T G.711. The old
                # 0x100 left A-law ~6 dB below spec (error 8<<(exp-1), up to 512 at exp 7).
                # Verified 256/256 against ITU-T (audioop) after this fix.
                linear = (mantissa << 4 | 0x108) << (exponent - 1)
            if sign == 0:
                linear = -linear
            table[i] = max(-32768, min(32767, linear))
        return table

    def _decode_g711_ulaw(self, data: bytes) -> bytes:
        """Decode G.711 μ-law audio to 16-bit PCM."""
        import struct

        table = self._get_ulaw_table()
        return struct.pack(f"<{len(data)}h", *[table[b] for b in data])

    def _decode_g711_alaw(self, data: bytes) -> bytes:
        """Decode G.711 A-law audio to 16-bit PCM."""
        import struct

        table = self._get_alaw_table()
        return struct.pack(f"<{len(data)}h", *[table[b] for b in data])

    # ── G.711 ENCODE — output path was decode-only, so g711 clients ──
    # got raw 24 kHz PCM16 mislabeled as 8 kHz μ-law → garbage playback. Build the
    # encoder by INVERTING the decode table (nearest decoded value) so encode→decode
    # round-trips optimally against our own decoder (verifiable without audio playback).
    _ULAW_ENC: bytes | None = None
    _ALAW_ENC: bytes | None = None

    @classmethod
    def _build_encode_lut(cls, decode_table: list[int]) -> bytes:
        import bisect

        codes = sorted(range(256), key=lambda c: decode_table[c])
        vals = [decode_table[c] for c in codes]
        lut = bytearray(65536)
        for pcm in range(-32768, 32768):
            j = bisect.bisect_left(vals, pcm)
            if j <= 0:
                best = codes[0]
            elif j >= 256:
                best = codes[255]
            else:
                best = (
                    codes[j]
                    if abs(vals[j] - pcm) < abs(vals[j - 1] - pcm)
                    else codes[j - 1]
                )
            lut[pcm & 0xFFFF] = best
        return bytes(lut)

    @classmethod
    def _get_ulaw_encode_lut(cls) -> bytes:
        if cls._ULAW_ENC is None:
            cls._ULAW_ENC = cls._build_encode_lut(cls._get_ulaw_table())
        return cls._ULAW_ENC

    @classmethod
    def _get_alaw_encode_lut(cls) -> bytes:
        if cls._ALAW_ENC is None:
            cls._ALAW_ENC = cls._build_encode_lut(cls._get_alaw_table())
        return cls._ALAW_ENC

    @staticmethod
    def _resample_24k_to_8k(pcm16: bytes) -> tuple[bytes, bytes]:
        """24 kHz → 8 kHz PCM16: box-filter (average each group of 3) + decimate by 3.
        The averaging is a crude anti-alias so content >4 kHz doesn't fold back.

        Returns (resampled_pcm8k, leftover_pcm16) where leftover is the
        up-to-2 trailing samples that didn't fill a group of 3. The caller carries
        the leftover into the next chunk so resampling per streaming chunk no
        longer drops boundary samples or resets the decimation phase (which caused
        cumulative shortening + periodic clicks for g711 streaming clients)."""
        import struct

        n = len(pcm16) // 2
        if n < 3:
            return b"", pcm16[: n * 2]
        s = struct.unpack(f"<{n}h", pcm16[: n * 2])
        groups = n // 3
        consumed = groups * 3
        out = [(s[i] + s[i + 1] + s[i + 2]) // 3 for i in range(0, consumed, 3)]
        leftover = pcm16[consumed * 2 : n * 2]
        return struct.pack(f"<{len(out)}h", *out), leftover

    def _resample_pcm16_linear(
        self, pcm16: bytes, from_rate: int, to_rate: int, state_attr: str
    ) -> bytes:
        """Streaming linear-interpolation resampler with carried state.

        Used when the TTS engine's native rate ≠ the realtime output target. Carries
        (last_input_sample, fractional_position) across chunks so per-chunk resampling
        doesn't drop boundary samples or reset phase. Linear interp (no anti-alias) is
        fine for speech; the 24 kHz→8 kHz g711 path keeps its dedicated box filter."""
        if from_rate == to_rate or not pcm16:
            return pcm16
        import struct

        n = len(pcm16) // 2
        if n == 0:
            return b""
        samples = list(struct.unpack(f"<{n}h", pcm16[: n * 2]))
        st = getattr(self, state_attr, None)
        last, pos = st if st is not None else (samples[0], 0.0)
        buf = [last] + samples  # prepend carried sample so interp spans the boundary
        step = from_rate / to_rate  # input samples consumed per output sample
        out = []
        i = pos
        _last_idx = len(buf) - 1
        while i < _last_idx:
            i0 = int(i)
            frac = i - i0
            out.append(int(buf[i0] * (1.0 - frac) + buf[i0 + 1] * frac))
            i += step
        setattr(self, state_attr, (buf[-1], i - _last_idx))
        return struct.pack(f"<{len(out)}h", *out) if out else b""

    def _encode_output_audio(
        self, pcm16: bytes, in_rate: int = 24000, fmt: str | None = None
    ) -> tuple[bytes, int]:
        """Convert engine PCM16 (at ``in_rate``) to the negotiated output_audio_format.

        The engine PCM is at the TTS model's REAL sample rate (e.g. dia=44.1 kHz),
        which the caller now passes in. The old code assumed 24 kHz unconditionally, so a
        non-24 kHz model played at the wrong pitch/speed (pcm16 mislabeled; g711 decimated by
        the wrong factor). Resample to the format's target (24 kHz pcm16 / 8 kHz g711) from
        the real rate. The default 24 kHz path is unchanged (passthrough / box filter).

        ``fmt`` lets a per-response output_audio_format override win over the session
        default (snapshotted at response-create time). Reading self.session live here silently
        dropped that override (session pcm16 + response.create g711_ulaw → encoded as pcm16).

        Returns (encoded_bytes, chunk_size_for_20ms)."""
        fmt = fmt or self.session.output_audio_format
        if fmt == "g711_ulaw" or fmt == "g711_alaw":
            if in_rate == 24000:
                # prepend the carried remainder so the 3-sample decimation window
                # is continuous across streaming chunks (the dedicated anti-aliased path).
                pcm = getattr(self, "_g711_resample_remainder", b"") + pcm16
                pcm8k, self._g711_resample_remainder = self._resample_24k_to_8k(pcm)
            else:
                # Non-24k engine: general streaming resample real_rate → 8 kHz.
                pcm8k = self._resample_pcm16_linear(
                    pcm16, in_rate, 8000, "_g711_lin_state"
                )
            n = len(pcm8k) // 2
            import struct

            samples = struct.unpack(f"<{n}h", pcm8k[: n * 2]) if n else ()
            enc = (
                self._get_ulaw_encode_lut()
                if fmt == "g711_ulaw"
                else self._get_alaw_encode_lut()
            )
            out = bytes(enc[sample & 0xFFFF] for sample in samples)
            return out, 160  # 20 ms @ 8 kHz, 1 byte/sample
        # pcm16 output: the OpenAI Realtime contract is 24 kHz — resample if the engine isn't.
        pcm24 = (
            pcm16
            if in_rate == 24000
            else self._resample_pcm16_linear(pcm16, in_rate, 24000, "_pcm16_lin_state")
        )
        return pcm24, self._AUDIO_CHUNK_BYTES

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.session = SessionConfig()
        self.conversation = Conversation(f"conv_{uuid.uuid4().hex[:16]}")
        self._active_response: asyncio.Task | None = None
        self._cancel_event: asyncio.Event | None = None
        self._audio_buffer = bytearray()
        # Native-omni speech-in: (pcm16_bytes, sample_rate) of the last committed
        # user audio, stashed so the omni response can use it directly. None when
        # absent/consumed. Only populated when the omni realtime path is enabled.
        self._last_user_audio: tuple[bytes, int] | None = None
        # per-response leftover for the continuous 24k→8k g711 resampler.
        self._g711_resample_remainder = b""
        # VAD state
        self._vad_speaking = False
        self._vad_silence_bytes: int = (
            0  # silence window in AUDIO bytes, not wall-clock
        )
        self._vad_speech_start_offset: int = 0
        # Barge-in debounce: accumulated speech (audio bytes) since the utterance
        # started, and whether the barge-in cancel already fired for it. We interrupt
        # an active response only after SUSTAINED speech (barge_in_min_ms) so a single
        # noise blip / cough that crosses threshold for one window can't kill a reply.
        self._vad_speech_bytes: int = 0
        self._barge_in_fired: bool = False
        # Silero VAD streaming state (only used when YUNSHU_REALTIME_VAD=silero):
        # LSTM/context state + a leftover buffer of resampled-16k float32 samples
        # not yet forming a full Silero window. Reset on buffer clear/commit.
        self._silero_state = None
        self._silero_leftover = None  # np.ndarray | None
        # Track whether the active response includes audio modality
        self._active_modalities: list[str] = []
        # (self-audit): lifecycle bookkeeping for the active response.
        # _response_item_open — an output item was announced (output_item.added) but
        # not yet closed (output_item.done); cancel/error paths must close it.
        # _response_done_emitted — the generation task already sent a terminal
        # response.done (happy/OOM/error). The cancel handler reads this AFTER
        # awaiting the task to avoid emitting a SECOND response.done when the task
        # finished naturally in the same scheduling window as a cancel/barge-in.
        self._response_item_open: bool = False
        self._response_done_emitted: bool = False

    async def send_event(self, event: dict) -> None:
        try:
            await self.ws.send_json(event)
        except Exception as e:
            logger.warning(f"Failed to send realtime event: {e}")

    # Lock to prevent interleaving of cancel-done events and new response events
    _send_lock: asyncio.Lock | None = None

    async def _locked_send_event(self, event: dict) -> None:
        """Send an event under the send lock to preserve ordering."""
        if self._send_lock is None:
            self._send_lock = asyncio.Lock()
        async with self._send_lock:
            await self.send_event(event)

    async def run(self) -> None:
        """Main event loop for the WebSocket session."""
        # Send session.created
        await self.send_event(
            _event(
                RealtimeEvent.SESSION_CREATED,
                session=self.session.to_dict(),
            )
        )

        # Send conversation.created
        await self.send_event(
            _event(
                RealtimeEvent.CONVERSATION_CREATED,
                conversation={"id": self.conversation.conversation_id, "items": []},
            )
        )

        # Warn if no engine is available — prevents silent failures on response.create
        if self._resolve_engine() is None:
            await self.send_event(
                _event(
                    RealtimeEvent.ERROR,
                    error={
                        "message": "No inference engine is loaded. Requests will fail until a model is loaded.",
                        "type": "server_error",
                    },
                )
            )

        try:
            while True:
                raw = await self.ws.receive_text()
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    await self.send_event(
                        _event(
                            RealtimeEvent.ERROR,
                            error={
                                "message": "Invalid JSON",
                                "type": "invalid_request_error",
                            },
                        )
                    )
                    continue

                await self._handle_event(event)
        except WebSocketDisconnect:
            logger.info("Realtime client disconnected")
            # Signal cancellation immediately so GPU work stops promptly
            if self._cancel_event is not None:
                self._cancel_event.set()
        except Exception as e:
            logger.error(f"Realtime session error: {e}", exc_info=True)
        finally:
            # Cancel any active generation
            if self._cancel_event is not None:
                self._cancel_event.set()
            if self._active_response and not self._active_response.done():
                self._active_response.cancel()
                # Give the cancelled task a chance to run its finally block
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.shield(self._active_response)
            # Release all session state to free memory on disconnect
            self._audio_buffer = bytearray()
            self.conversation.items.clear()
            self.session.tools = []
            self._active_response = None
            self._active_modalities = []
            self._cancel_event = None
            self._send_lock = None
            # Explicitly close the WebSocket to prevent connection leaks
            with contextlib.suppress(Exception):
                await self.ws.close()

    async def _handle_event(self, event: dict) -> None:
        """Dispatch incoming events to handlers."""
        event_type = event.get("type", "")
        handler = _EVENT_HANDLERS.get(event_type)
        if handler is None:
            await self.send_event(
                _event(
                    RealtimeEvent.ERROR,
                    error={
                        "message": f"Unknown event type: {event_type}",
                        "type": "invalid_request_error",
                    },
                )
            )
            return
        try:
            await handler(self, event)
        except Exception as e:
            logger.error(f"Handler error for {event_type}: {e}", exc_info=True)
            await self.send_event(
                _event(
                    RealtimeEvent.ERROR,
                    error={"message": "Internal server error", "type": "server_error"},
                )
            )

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
        self.session.update(session_data)
        await self.send_event(
            _event(
                RealtimeEvent.SESSION_UPDATED,
                session=self.session.to_dict(),
            )
        )

    async def _handle_conversation_item_create(self, event: dict) -> None:
        """Handle conversation.item.create — add a message to conversation."""
        item_data = event.get("item", {})
        item_id = item_data.get("id", f"item_{uuid.uuid4().hex[:24]}")
        # reject a client-supplied id that already exists. Otherwise both
        # copies live in the list but get_item/delete/truncate all resolve to the
        # FIRST → the second is unreachable/un-deletable and silently duplicates
        # into the prompt history.
        if self.conversation.has_item(item_id):
            await self.send_event(
                _event(
                    RealtimeEvent.ERROR,
                    error={
                        "message": f"Item id '{item_id}' already exists",
                        "type": "invalid_request_error",
                    },
                )
            )
            return
        # bound a single item's content size. content is taken verbatim into the
        # prompt history, so an arbitrarily large client item is a memory / prompt-cost DoS
        # vector (sibling of the audio cap). Cap the serialized content bytes.
        _content = item_data.get("content", [])
        try:
            _csize = len(str(_content))
        except Exception:
            _csize = 0
        if _csize > _max_input_audio_bytes():
            await self.send_event(
                _event(
                    RealtimeEvent.ERROR,
                    error={
                        "message": "Item content exceeds the maximum allowed size",
                        "type": "invalid_request_error",
                    },
                )
            )
            return
        item = ConversationItem(
            item_id=item_id,
            item_type=item_data.get("type", "message"),
            role=item_data.get("role"),
            content=item_data.get("content", []),
            call_id=item_data.get("call_id"),
            name=item_data.get("name"),
            arguments=item_data.get("arguments"),
            output=item_data.get("output"),
        )
        item.status = "completed"
        # previous_item_id may be on the event or inside the item object.
        _prev_id = event.get("previous_item_id") or item_data.get("previous_item_id")
        self.conversation.add_item(item, previous_item_id=_prev_id)
        await self.send_event(
            _event(
                "conversation.item.created",
                previous_item_id=_prev_id,
                item=item.to_dict(),
            )
        )

    async def _handle_conversation_item_delete(self, event: dict) -> None:
        """Handle conversation.item.delete — remove an item from conversation.

        Removes the item with the specified item_id from the conversation.
        Sends conversation.item.deleted confirmation event.
        """
        item_id = event.get("item_id", "")
        if not item_id:
            await self.send_event(
                _event(
                    RealtimeEvent.ERROR,
                    error={
                        "message": "Missing item_id",
                        "type": "invalid_request_error",
                    },
                )
            )
            return

        # Find and remove the item
        removed = False
        for i, item in enumerate(self.conversation.items):
            if item.item_id == item_id:
                self.conversation.items.pop(i)
                removed = True
                break

        if removed:
            await self.send_event(
                _event(
                    "conversation.item.deleted",
                    item_id=item_id,
                )
            )
        else:
            await self.send_event(
                _event(
                    RealtimeEvent.ERROR,
                    error={
                        "message": f"Item {item_id} not found",
                        "type": "invalid_request_error",
                    },
                )
            )

    async def _handle_conversation_item_truncate(self, event: dict) -> None:
        """Handle conversation.item.truncate (formerly an 'unknown event' error).

        After a barge-in, the client tells the server how much of an assistant audio item the
        user actually heard. Truncate the item's audio content (and proportionally its
        transcript) at audio_end_ms so future turns are conditioned on what was really heard,
        then confirm with conversation.item.truncated.
        """
        item_id = event.get("item_id", "")
        content_index = event.get("content_index", 0)
        audio_end_ms = event.get("audio_end_ms", 0)
        if not item_id:
            await self.send_event(
                _event(
                    RealtimeEvent.ERROR,
                    error={
                        "message": "Missing item_id",
                        "type": "invalid_request_error",
                    },
                )
            )
            return
        target = next(
            (it for it in self.conversation.items if it.item_id == item_id), None
        )
        if target is None:
            await self.send_event(
                _event(
                    RealtimeEvent.ERROR,
                    error={
                        "message": f"Item {item_id} not found",
                        "type": "invalid_request_error",
                    },
                )
            )
            return
        # Best-effort transcript truncation proportional to audio_end_ms when we know the
        # item's audio duration; otherwise leave the text and just record the truncation.
        try:
            # Per the protocol, truncate ONLY the content part at content_index — not
            # every audio part. Walk audio/input_audio parts in order and act on the
            # content_index-th one.
            _audio_seen = -1
            for c in target.content or []:
                if isinstance(c, dict) and c.get("type") in ("audio", "input_audio"):
                    _audio_seen += 1
                    if _audio_seen != content_index:
                        continue
                    _dur = c.get("duration_ms")
                    _tr = c.get("transcript")
                    if (
                        _tr
                        and isinstance(_dur, (int, float))
                        and _dur > 0
                        and audio_end_ms < _dur
                    ):
                        _keep = max(0, int(len(_tr) * (audio_end_ms / _dur)))
                        c["transcript"] = _tr[:_keep]
                    c["audio_end_ms"] = audio_end_ms
                    break
        except Exception:
            logger.debug("item truncate transcript adjust failed", exc_info=True)
        await self.send_event(
            _event(
                "conversation.item.truncated",
                item_id=item_id,
                content_index=content_index,
                audio_end_ms=audio_end_ms,
            )
        )

    async def _handle_response_create(self, event: dict) -> None:
        """Handle response.create — trigger model generation.

        Supports modalities filter in the response config:
        - ["text"]: text-only response (no audio synthesis)
        - ["audio"]: audio-only response (text used internally but not streamed)
        - ["text", "audio"]: both text deltas and audio chunks
        If modalities is not specified, uses session-level modalities.
        """
        if self._active_response and not self._active_response.done():
            await self.send_event(
                _event(
                    RealtimeEvent.ERROR,
                    error={
                        "message": "Response already in progress",
                        "type": "server_error",
                    },
                )
            )
            return

        response_id = f"resp_{uuid.uuid4().hex[:16]}"
        item_id = f"item_{uuid.uuid4().hex[:24]}"

        response_config = event.get("response", {})
        modalities = response_config.get("modalities", self.session.modalities)

        # Validate modalities
        valid_modalities = {"text", "audio"}
        if not isinstance(modalities, list):
            modalities = self.session.modalities
        modalities = [m for m in modalities if m in valid_modalities]
        if not modalities:
            modalities = ["text"]

        await self.send_event(
            _event(
                RealtimeEvent.RESPONSE_CREATED,
                response={
                    "id": response_id,
                    "object": "realtime.response",
                    "status": "in_progress",
                    "modalities": modalities,
                },
            )
        )

        self._active_response = asyncio.create_task(
            self._generate_response(response_id, item_id, modalities, response_config)
        )
        self._active_response._response_id = response_id
        self._active_response._item_id = item_id
        self._active_modalities = modalities

    async def _generate_response(
        self,
        response_id: str,
        item_id: str,
        modalities: list[str],
        config: dict,
    ) -> None:
        """Generate a response and stream deltas back."""
        # Native-omni path: one unified Qwen3-Omni model produces the reply text
        # AND its speech in a single shared-context pass (Thinker+Talker), instead
        # of the LLM→(separate)TTS cascade below. Opt-in via YUNSHU_REALTIME_OMNI;
        # off by default so the cascade behaviour is untouched.
        if _omni_realtime_enabled():
            await self._generate_response_omni(response_id, item_id, modalities, config)
            return
        # Create cancel_event so engine can check for cancellation
        self._cancel_event = asyncio.Event()
        # (self-audit R2): fresh response → no terminal response.done yet.
        self._response_done_emitted = False
        try:
            # honor per-response config overrides (instructions/voice/
            # tools) from response.create — previously only session-level values
            # were used, so a turn-specific instruction/voice/tool set was ignored.
            _instr_override = config.get("instructions")
            messages = self._build_messages(instructions_override=_instr_override)
            if not messages:
                await self.send_event(
                    _event(
                        RealtimeEvent.ERROR,
                        error={
                            "message": "No messages in conversation",
                            "type": "server_error",
                        },
                    )
                )
                # an `error` event does NOT terminate a response in the
                # Realtime protocol — the response stays in_progress until
                # response.done. These two early returns emitted ONLY the error, so
                # the client's response future never resolved → per-turn deadlock
                # (the SDK also refuses to start the next response while one is
                # "active"). Emit the terminal response.done(failed), mirroring the
                # OOM/Exception paths. (_close_response_item is a no-op here — the
                # output item isn't open yet — but kept for symmetry/future-proofing.)
                await self._close_response_item(
                    response_id, item_id, status="incomplete"
                )
                await self.send_event(
                    _event(
                        RealtimeEvent.RESPONSE_DONE,
                        response={
                            "id": response_id,
                            "object": "realtime.response",
                            "status": "failed",
                            "error": "No messages in conversation",
                        },
                    )
                )
                self._response_done_emitted = True
                return

            engine = self._resolve_engine()
            if engine is None:
                await self.send_event(
                    _event(
                        RealtimeEvent.ERROR,
                        error={
                            "message": "No engine available",
                            "type": "server_error",
                        },
                    )
                )
                # see above — emit the terminal response.done so the client
                # doesn't hang when the (per-response) model is unavailable mid-session.
                await self._close_response_item(
                    response_id, item_id, status="incomplete"
                )
                await self.send_event(
                    _event(
                        RealtimeEvent.RESPONSE_DONE,
                        response={
                            "id": response_id,
                            "object": "realtime.response",
                            "status": "failed",
                            "error": "No engine available",
                        },
                    )
                )
                self._response_done_emitted = True
                return

            # the per-response cap is `max_response_output_tokens` (same name
            # as the session field), NOT `max_output_tokens` — reading the wrong key
            # silently ignored every client-supplied per-response cap. AND this path
            # skipped the string-coercion SessionConfig.update applies, so a documented
            # value like "inf" (or any string) flowed verbatim into stream_chat(max_tokens=)
            # → mlx-lm compares the counter against a str → every such turn failed.
            max_tokens = _coerce_max_response_tokens(
                config.get(
                    "max_response_output_tokens",
                    self.session.max_response_output_tokens,
                ),
                self.session.max_response_output_tokens,
            )
            # per-response temperature was likewise unvalidated — a non-numeric
            # override (e.g. "high") reached the sampler and failed the turn. Fall back.
            temperature = config.get("temperature", self.session.temperature)
            try:
                temperature = float(temperature)
            except (TypeError, ValueError):
                temperature = self.session.temperature
            # SNAPSHOT the config this response depends on at creation time. The
            # receive loop runs concurrently with this generation task and a mid-response
            # session.update mutates self.session live — reading tools/model/voice live could
            # parse tool calls against a different tool set or synthesize audio in a voice the
            # response was not created with. Use the snapshot throughout this response.
            # per-response overrides win over the session snapshot.
            # `tools` uses "in config" (an explicit empty list disables tools for
            # this turn); voice falls back to the session voice.
            _snap_tools = config.get("tools", self.session.tools)
            # per-response tool_choice override (falls back to the session default),
            # snapshotted like tools/voice so a mid-response session.update can't change it.
            _snap_tool_choice = config.get(
                "tool_choice", getattr(self.session, "tool_choice", "auto")
            )
            # response.create with conversation="none" is an OUT-OF-BAND response —
            # the generated output is delivered to the client but must NOT be appended to the
            # persistent conversation, else the next normal turn's prompt (_build_messages reads
            # self.conversation.items) wrongly includes this side-query (e.g. a guardrail /
            # classification call). The response output events still fire; only the history
            # writes + the conversation.item.created signal are suppressed.
            _oob = config.get("conversation") == "none"
            _snap_model = self.session.model
            _snap_voice = config.get("voice") or self.session.voice
            # per-response output_audio_format override (a documented OpenAI Realtime
            # per-response field), snapshotted like voice. _encode_output_audio used to read
            # self.session.output_audio_format LIVE, so a per-response override (e.g. session
            # pcm16 + response.create output_audio_format=g711_ulaw) was silently dropped and
            # the audio was encoded at the wrong format/rate (the class). An invalid value
            # falls back to the session format (mirrors SessionConfig.update validation).
            _snap_out_fmt = config.get("output_audio_format")
            if _snap_out_fmt not in self.session.SUPPORTED_AUDIO_FORMATS:
                _snap_out_fmt = self.session.output_audio_format

            from yunshu_engine.batched_engine import BatchedEngine

            is_batched = isinstance(engine, BatchedEngine)

            # emit the required OpenAI-Realtime lifecycle events that announce the
            # output item + content part. SDK clients register the output item from
            # response.output_item.added (carrying output_index + the item shell); without it
            # the subsequent text deltas reference an item/index the client never saw and are
            # dropped. Chain: output_item.added → content_part.added → deltas →
            # content_part.done → output_item.done → response.done.
            # (self-audit R1): mark the output item OPEN *before* emitting the open
            # events, not after. A cancel/barge-in landing during either send's await would
            # otherwise leave the flag False → _close_response_item early-returns → dangling
            # in-progress item (the very leak c7a7ad4 set out to fix, residual window). Setting
            # it first means the close path always balances; the worst case is an early
            # output_item.done for an item whose .added send was interrupted, which SDK clients
            # tolerate far better than a never-closed in-progress item.
            self._response_item_open = True
            await self.send_event(
                _event(
                    "response.output_item.added",
                    response_id=response_id,
                    output_index=0,
                    item={
                        "id": item_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "status": "in_progress",
                    },
                )
            )
            await self.send_event(
                _event(
                    "response.content_part.added",
                    response_id=response_id,
                    item_id=item_id,
                    output_index=0,
                    content_index=0,
                    part={"type": "text", "text": ""},
                )
            )

            full_text = ""
            _usage_pt = 0  # prompt tokens
            _usage_ct = 0  # completion tokens

            # Extract session-level generation parameters
            _stop = config.get("stop") or getattr(self.session, "stop", None)
            _stop_token_ids = config.get("stop_token_ids")
            _thinking_budget = config.get("thinking_budget")
            _priority = config.get("priority", 0)

            if is_batched:
                async for output in engine.stream_chat(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=getattr(self.session, "top_p", 1.0),
                    top_k=getattr(self.session, "top_k", 0),
                    min_p=getattr(self.session, "min_p", 0.0),
                    repetition_penalty=getattr(self.session, "repetition_penalty", 1.0),
                    frequency_penalty=getattr(self.session, "frequency_penalty", 0.0),
                    presence_penalty=getattr(self.session, "presence_penalty", 0.0),
                    logit_bias=getattr(self.session, "logit_bias", None),
                    enable_thinking=getattr(self.session, "enable_thinking", False),
                    thinking_budget=_thinking_budget,
                    stop=_stop,
                    stop_token_ids=_stop_token_ids,
                    seed=getattr(self.session, "seed", None),
                    priority=_priority,
                    cancel_event=self._cancel_event,
                ):
                    if output.new_text:
                        full_text += output.new_text
                        if "text" in modalities:
                            await self.send_event(
                                _event(
                                    RealtimeEvent.RESPONSE_TEXT_DELTA,
                                    response_id=response_id,
                                    item_id=item_id,
                                    output_index=0,
                                    content_index=0,
                                    delta=output.new_text,
                                )
                            )
                        # an audio response must stream its transcript too —
                        # OpenAI server_vad always emits response.audio_transcript.delta
                        # alongside the audio so the client can render what's being said.
                        # Without this an audio-only modality response carried NO transcript
                        # at all (text delta is gated off, transcript was never emitted).
                        if "audio" in modalities:
                            await self.send_event(
                                _event(
                                    RealtimeEvent.RESPONSE_AUDIO_TRANSCRIPT_DELTA,
                                    response_id=response_id,
                                    item_id=item_id,
                                    output_index=0,
                                    content_index=0,
                                    delta=output.new_text,
                                )
                            )
                    if getattr(output, "prompt_tokens", 0):
                        _usage_pt = output.prompt_tokens
                    if getattr(output, "completion_tokens", 0):
                        _usage_ct = max(_usage_ct, output.completion_tokens)
                    if output.finish_reason:
                        break
            else:
                async for output in engine.generate_stream(
                    prompt=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=getattr(self.session, "top_p", 1.0),
                    top_k=getattr(self.session, "top_k", 0),
                    min_p=getattr(self.session, "min_p", 0.0),
                    repetition_penalty=getattr(self.session, "repetition_penalty", 1.0),
                    frequency_penalty=getattr(self.session, "frequency_penalty", 0.0),
                    presence_penalty=getattr(self.session, "presence_penalty", 0.0),
                    logit_bias=getattr(self.session, "logit_bias", None),
                    enable_thinking=getattr(self.session, "enable_thinking", False),
                    thinking_budget=_thinking_budget,
                    stop=_stop,
                    stop_token_ids=_stop_token_ids,
                    seed=getattr(self.session, "seed", None),
                    cancel_event=self._cancel_event,
                ):
                    if output.token_text:
                        full_text += output.token_text
                        if "text" in modalities:
                            await self.send_event(
                                _event(
                                    RealtimeEvent.RESPONSE_TEXT_DELTA,
                                    response_id=response_id,
                                    item_id=item_id,
                                    output_index=0,
                                    content_index=0,
                                    delta=output.token_text,
                                )
                            )
                        # stream the transcript for audio responses (see above).
                        if "audio" in modalities:
                            await self.send_event(
                                _event(
                                    RealtimeEvent.RESPONSE_AUDIO_TRANSCRIPT_DELTA,
                                    response_id=response_id,
                                    item_id=item_id,
                                    output_index=0,
                                    content_index=0,
                                    delta=output.token_text,
                                )
                            )
                    if getattr(output, "prompt_tokens", 0):
                        _usage_pt = output.prompt_tokens
                    if getattr(output, "completion_tokens", 0):
                        _usage_ct = max(_usage_ct, output.completion_tokens)
                    if output.finish_reason:
                        break

            # text done
            if "text" in modalities:
                await self.send_event(
                    _event(
                        RealtimeEvent.RESPONSE_TEXT_DONE,
                        response_id=response_id,
                        item_id=item_id,
                        output_index=0,
                        content_index=0,
                        text=full_text,
                    )
                )
            # finalize the audio transcript stream for audio responses.
            if "audio" in modalities:
                await self.send_event(
                    _event(
                        RealtimeEvent.RESPONSE_AUDIO_TRANSCRIPT_DONE,
                        response_id=response_id,
                        item_id=item_id,
                        output_index=0,
                        content_index=0,
                        transcript=full_text,
                    )
                )
            # close the content part (required by the protocol; was missing).
            await self.send_event(
                _event(
                    "response.content_part.done",
                    response_id=response_id,
                    item_id=item_id,
                    output_index=0,
                    content_index=0,
                    part={"type": "text", "text": full_text},
                )
            )

            # Check for tool calls in the response (against the snapshotted tool set/model).
            tool_calls = None
            _visible_text = full_text
            # the realtime streaming path never separates reasoning, so an
            # enable_thinking session would replay the whole <think>…</think> chain-of-thought
            # into the next turn's prompt (and TTS it) — the leak class,
            # un-propagated here. Strip it from the stored transcript (no-op without markup).
            if "<think" in full_text:
                from ..streaming import extract_thinking

                try:
                    _, _visible_text = extract_thinking(full_text, _snap_model)
                except Exception:
                    _visible_text = full_text
            # honor tool_choice="none" — the client explicitly wants a plain-text
            # turn, so do NOT parse/emit tool calls even when tools are present. The realtime
            # path previously parsed unconditionally (the sibling of the anthropic / chat
            # tool_choice enforcement, which realtime lacked) → a function_call leaked out when
            # the client asked for none. "auto"/"required"/a named-function dict keep parsing
            # (post-gen "required"/named forcing isn't feasible here — it stays prompt-advised).
            if _snap_tools and _snap_tool_choice != "none":
                from yunshu_engine.tool_call_parser import parse_tool_calls

                tool_calls = parse_tool_calls(full_text, model_name=_snap_model)
                if tool_calls:
                    # full_text still holds the raw <tool_call>… markup. Storing it
                    # as the assistant message text leaked the markup back into the NEXT turn's
                    # prompt AND duplicated the call (once as the function_call item below, once
                    # as markup in the message). Strip it; when nothing visible remains,
                    # _build_messages skips the empty message (no text_parts) so the turn is
                    # represented ONCE by the function_call item. Also don't TTS the markup.
                    from ..streaming import clean_tool_call_markup

                    try:
                        _visible_text = clean_tool_call_markup(_visible_text).strip()
                    except Exception:
                        _visible_text = ""
            elif (
                _snap_tools
                and _snap_tool_choice == "none"
                and "<tool_call" in full_text
            ):
                # tool_choice="none" forbids EMITTING tool calls, but if the model
                # emitted <tool_call> markup anyway, still strip it from the visible transcript
                # so it doesn't leak into the next prompt / get TTS'd (class). No
                # function_call items are produced (tool_calls stays None).
                from ..streaming import clean_tool_call_markup

                with contextlib.suppress(Exception):
                    _visible_text = clean_tool_call_markup(_visible_text).strip()

            # persist each tool call as a function_call item so it's in history
            # WITH its call_id. The client returns a function_call_output carrying the same
            # call_id; _build_messages then pairs the assistant tool_call with the tool
            # result for the next turn. the streamed events (output_item.added +
            # arguments delta/done + output_item.done) are emitted AFTER the message item's
            # output_item.done below, each as its OWN output item at a distinct output_index
            # with its OWN item id — the old code streamed them under the assistant MESSAGE
            # item_id at output_index 0 (colliding with the text item) and never sent
            # output_item.added, so SDK clients keying on output_item.added missed the call.
            _fc_items = []
            if tool_calls:
                for tc in tool_calls:
                    _fc_item = ConversationItem(
                        item_id=f"item_{uuid.uuid4().hex[:24]}",
                        item_type="function_call",
                        call_id=f"call_{uuid.uuid4().hex[:8]}",
                        name=tc.name,
                        arguments=tc.arguments,
                    )
                    _fc_item.status = "completed"
                    # out-of-band (conversation="none") tool calls are still streamed
                    # to the client (via the output_item events below) but NOT persisted.
                    if not _oob:
                        self.conversation.add_item(_fc_item)
                    _fc_items.append(_fc_item)

            # Audio output: synthesize text to speech if audio modality is requested
            # .
            if "audio" in modalities and _visible_text:
                await self._synthesize_audio_response(
                    _visible_text,
                    response_id,
                    item_id,
                    voice=_snap_voice,
                    out_fmt=_snap_out_fmt,
                )
            elif "audio" in modalities:
                # audio was requested but this turn produced no visible text
                # (empty / tool-only) → no synthesis runs, but the client still expects a
                # terminal audio event for the turn. Emit a bare response.audio.done.
                await self.send_event(
                    _event(
                        RealtimeEvent.RESPONSE_AUDIO_DONE,
                        response_id=response_id,
                        item_id=item_id,
                        output_index=0,
                        content_index=0,
                    )
                )

            # Add assistant item to conversation. when audio was
            # synthesized, store the reply as an AUDIO content part carrying the
            # transcript + a duration_ms estimate. The old text-only part made
            # conversation.item.truncate a silent NO-OP (it only trims audio parts)
            # → after a barge-in the model re-saw the FULL reply. _build_messages
            # already reads the transcript from audio parts, and truncate trims it
            # proportionally to audio_end_ms — so this single change makes barge-in
            # truncation actually work. (duration_ms is a speech-rate estimate; only
            # a non-zero duration is needed for the proportional trim.)
            if "audio" in modalities and _visible_text:
                content_parts = [
                    {
                        "type": "audio",
                        "transcript": _visible_text,
                        "duration_ms": max(1, len(_visible_text) * 70),
                    }
                ]
            else:
                content_parts = [{"type": "text", "text": _visible_text}]
            if tool_calls:
                for tc in tool_calls:
                    content_parts.append(
                        {
                            "type": "function_call",
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    )
            assistant_item = ConversationItem(
                item_id=item_id,
                item_type="message",
                role="assistant",
                content=content_parts,
            )
            assistant_item.status = "completed"
            # skip persisting + signalling conversation.item.created for an
            # out-of-band response (conversation="none"); the output_item.done + response.done
            # events below still deliver the assistant turn to the client.
            if not _oob:
                self.conversation.add_item(assistant_item)
                await self.send_event(
                    _event(
                        "conversation.item.created",
                        item=assistant_item.to_dict(),
                    )
                )

            # close the output item before response.done (required by protocol).
            await self.send_event(
                _event(
                    "response.output_item.done",
                    response_id=response_id,
                    output_index=0,
                    item=assistant_item.to_dict(),
                )
            )
            self._response_item_open = False  # closed on the happy path

            # now stream each function_call as its own output item at output_index
            # 1+i, AFTER the message item's done so output order is monotonic. Each carries
            # its OWN item id (so clients correlate the deltas) and call_id.
            for _fc_i, _fc_item in enumerate(_fc_items):
                _fc_oidx = 1 + _fc_i
                await self.send_event(
                    _event(
                        "response.output_item.added",
                        response_id=response_id,
                        output_index=_fc_oidx,
                        item={
                            "id": _fc_item.item_id,
                            "object": "realtime.item",
                            "type": "function_call",
                            "status": "in_progress",
                            "name": _fc_item.name,
                            "call_id": _fc_item.call_id,
                            "arguments": "",
                        },
                    )
                )
                await self.send_event(
                    _event(
                        RealtimeEvent.RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA,
                        response_id=response_id,
                        item_id=_fc_item.item_id,
                        output_index=_fc_oidx,
                        call_id=_fc_item.call_id,
                        name=_fc_item.name,
                        delta=_fc_item.arguments,
                    )
                )
                await self.send_event(
                    _event(
                        RealtimeEvent.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE,
                        response_id=response_id,
                        item_id=_fc_item.item_id,
                        output_index=_fc_oidx,
                        call_id=_fc_item.call_id,
                        name=_fc_item.name,
                        arguments=_fc_item.arguments,
                    )
                )
                await self.send_event(
                    _event(
                        "response.output_item.done",
                        response_id=response_id,
                        output_index=_fc_oidx,
                        item=_fc_item.to_dict(),
                    )
                )

            # response.done — include usage so metering clients get token counts.
            await self.send_event(
                _event(
                    RealtimeEvent.RESPONSE_DONE,
                    response={
                        "id": response_id,
                        "object": "realtime.response",
                        "status": "completed",
                        "output": [assistant_item.to_dict()]
                        + [fi.to_dict() for fi in _fc_items],
                        "usage": {
                            "total_tokens": _usage_pt + _usage_ct,
                            "input_tokens": _usage_pt,
                            "output_tokens": _usage_ct,
                        },
                    },
                )
            )
            self._response_done_emitted = True  # (self-audit R2)

            # Record metrics
            try:
                from ..middleware.metrics import get_metrics

                get_metrics().record_inference()
            except Exception:
                logger.debug("metrics recording failed", exc_info=True)

        except asyncio.CancelledError:
            # Do NOT send response.done here. The caller (_handle_response_cancel)
            # is responsible for the full teardown sequence: audio.done THEN
            # response.done. Sending response.done here causes wrong protocol
            # ordering (response.done before audio.done) and duplicates the
            # response.done event.
            pass
        except MemoryError:
            logger.error("Realtime generation OOM", exc_info=True)
            await self.send_event(
                _event(
                    RealtimeEvent.ERROR,
                    error={"message": "Out of GPU memory", "type": "memory_error"},
                )
            )
            await self._close_response_item(response_id, item_id, status="incomplete")
            await self.send_event(
                _event(
                    RealtimeEvent.RESPONSE_DONE,
                    response={
                        "id": response_id,
                        "object": "realtime.response",
                        "status": "failed",
                        "error": "Out of GPU memory",
                    },
                )
            )
            self._response_done_emitted = True  # (self-audit R2)
        except Exception as e:
            logger.error(f"Realtime generation error: {e}", exc_info=True)
            await self.send_event(
                _event(
                    RealtimeEvent.ERROR,
                    error={"message": "Internal server error", "type": "server_error"},
                )
            )
            await self._close_response_item(response_id, item_id, status="incomplete")
            await self.send_event(
                _event(
                    RealtimeEvent.RESPONSE_DONE,
                    response={
                        "id": response_id,
                        "object": "realtime.response",
                        "status": "failed",
                        "error": "Generation failed",
                    },
                )
            )
            self._response_done_emitted = True  # (self-audit R2)
        finally:
            # Only clear if this task is still the active response.
            # Prevents race: cancel → new response.create → old finally wipes new task ref.
            if self._active_response is asyncio.current_task():
                self._active_response = None
                self._active_modalities = []
                self._cancel_event = None

    async def _generate_response_omni(
        self,
        response_id: str,
        item_id: str,
        modalities: list[str],
        config: dict,
    ) -> None:
        """Native-omni realtime response: ONE unified model (Qwen3-Omni
        Thinker+Talker) produces the reply text AND its speech in a single
        shared-context pass, replacing the LLM→(separate)TTS cascade. The audio
        carries the model's own Talker voice, not a downstream TTS voice.

        Emits the same OpenAI-Realtime event chain as the cascade path
        (output_item.added → content_part.added → text/transcript+audio deltas →
        transcript.done → audio.done → content_part.done → output_item.done →
        response.done) so SDK clients see no protocol difference. Mirrors the
        cascade's cancel contract: swallow CancelledError with `pass` and never
        double-emit response.done (the cancel handler owns terminal teardown)."""
        import base64

        self._cancel_event = asyncio.Event()
        self._response_done_emitted = False
        # Native speech-in: prefer the user's RAW committed audio over the ASR
        # transcript (consume it so it can't bleed into the next turn). When present,
        # the audio IS the query and the text turn carries only the persona.
        _audio_in = self._last_user_audio
        self._last_user_audio = None
        _tmp_audio_path: str | None = None
        try:
            _instr_override = config.get("instructions")
            _messages = self._build_messages(instructions_override=_instr_override)
            _audio_path: str | None = None
            if _audio_in and _audio_in[0]:
                pcm_bytes, in_rate = _audio_in
                _tmp_audio_path = _write_pcm16_wav(pcm_bytes, in_rate)
                _audio_path = _tmp_audio_path
                # persona-only text; the spoken audio is the user's actual turn
                prompt = _omni_system_text(_messages) or "Respond to the user's speech."
            else:
                prompt = _messages_to_omni_prompt(_messages)
            if not prompt:
                await self.send_event(
                    _event(
                        RealtimeEvent.ERROR,
                        error={
                            "message": "No messages in conversation",
                            "type": "server_error",
                        },
                    )
                )
                await self._close_response_item(
                    response_id, item_id, status="incomplete"
                )
                await self.send_event(
                    _event(
                        RealtimeEvent.RESPONSE_DONE,
                        response={
                            "id": response_id,
                            "object": "realtime.response",
                            "status": "failed",
                            "error": "No messages in conversation",
                        },
                    )
                )
                self._response_done_emitted = True
                return

            _snap_voice = config.get("voice") or self.session.voice
            _snap_out_fmt = config.get("output_audio_format")
            if _snap_out_fmt not in self.session.SUPPORTED_AUDIO_FORMATS:
                _snap_out_fmt = self.session.output_audio_format
            _oob = config.get("conversation") == "none"

            from .omni import _get_omni_engine

            eng = _get_omni_engine()

            # Lifecycle open (flag set BEFORE the awaits so a cancel landing mid-send
            # still balances via _close_response_item — mirrors the cascade's R1 fix).
            self._response_item_open = True
            await self.send_event(
                _event(
                    "response.output_item.added",
                    response_id=response_id,
                    output_index=0,
                    item={
                        "id": item_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "status": "in_progress",
                    },
                )
            )
            await self.send_event(
                _event(
                    "response.content_part.added",
                    response_id=response_id,
                    item_id=item_id,
                    output_index=0,
                    content_index=0,
                    part={"type": "text", "text": ""},
                )
            )

            full_text = ""
            self._pcm16_lin_state = None
            self._g711_lin_state = None
            self._g711_resample_remainder = b""
            async for ch in eng.stream(
                prompt, audio_path=_audio_path, speaker=_snap_voice or None
            ):
                if ch.kind == "text":
                    if not ch.data:
                        continue
                    full_text += ch.data
                    if "text" in modalities:
                        await self.send_event(
                            _event(
                                RealtimeEvent.RESPONSE_TEXT_DELTA,
                                response_id=response_id,
                                item_id=item_id,
                                output_index=0,
                                content_index=0,
                                delta=ch.data,
                            )
                        )
                    if "audio" in modalities:
                        await self.send_event(
                            _event(
                                RealtimeEvent.RESPONSE_AUDIO_TRANSCRIPT_DELTA,
                                response_id=response_id,
                                item_id=item_id,
                                output_index=0,
                                content_index=0,
                                delta=ch.data,
                            )
                        )
                elif ch.kind == "audio" and "audio" in modalities:
                    pcm = _f32_to_pcm16_bytes(ch.data)  # Talker f32 @ eng.sample_rate
                    _omni_sr = getattr(eng, "sample_rate", 24000)
                    out, _csz = self._encode_output_audio(
                        pcm, _omni_sr, fmt=_snap_out_fmt
                    )
                    offset = 0
                    while offset < len(out):
                        sub = out[offset : offset + _csz]
                        await self.send_event(
                            _event(
                                RealtimeEvent.RESPONSE_AUDIO_DELTA,
                                response_id=response_id,
                                item_id=item_id,
                                output_index=0,
                                content_index=0,
                                delta=base64.b64encode(sub).decode(),
                            )
                        )
                        offset += _csz

            if "audio" in modalities:
                await self.send_event(
                    _event(
                        RealtimeEvent.RESPONSE_AUDIO_TRANSCRIPT_DONE,
                        response_id=response_id,
                        item_id=item_id,
                        output_index=0,
                        content_index=0,
                        transcript=full_text,
                    )
                )
                await self.send_event(
                    _event(
                        RealtimeEvent.RESPONSE_AUDIO_DONE,
                        response_id=response_id,
                        item_id=item_id,
                        output_index=0,
                        content_index=0,
                    )
                )
            await self.send_event(
                _event(
                    "response.content_part.done",
                    response_id=response_id,
                    item_id=item_id,
                    output_index=0,
                    content_index=0,
                    part={"type": "text", "text": full_text},
                )
            )

            # Store the reply as an audio part (transcript + duration estimate) when
            # audio was produced, so conversation.item.truncate can trim it on barge-in
            # — same rationale as the cascade path.
            if "audio" in modalities and full_text:
                content_parts: list[dict] = [
                    {
                        "type": "audio",
                        "transcript": full_text,
                        "duration_ms": max(1, len(full_text) * 70),
                    }
                ]
            else:
                content_parts = [{"type": "text", "text": full_text}]
            assistant_item = ConversationItem(
                item_id=item_id,
                item_type="message",
                role="assistant",
                content=content_parts,
            )
            assistant_item.status = "completed"
            if not _oob:
                self.conversation.add_item(assistant_item)
                await self.send_event(
                    _event(
                        "conversation.item.created",
                        item=assistant_item.to_dict(),
                    )
                )
            await self.send_event(
                _event(
                    "response.output_item.done",
                    response_id=response_id,
                    output_index=0,
                    item=assistant_item.to_dict(),
                )
            )
            self._response_item_open = False
            await self.send_event(
                _event(
                    RealtimeEvent.RESPONSE_DONE,
                    response={
                        "id": response_id,
                        "object": "realtime.response",
                        "status": "completed",
                        "output": [assistant_item.to_dict()],
                        "usage": {
                            "total_tokens": 0,
                            "input_tokens": 0,
                            "output_tokens": 0,
                        },
                    },
                )
            )
            self._response_done_emitted = True
        except asyncio.CancelledError:
            # Cancel handler owns the terminal teardown (audio.done → response.done).
            pass
        except Exception as e:
            logger.error(f"Realtime omni generation error: {e}", exc_info=True)
            await self.send_event(
                _event(
                    RealtimeEvent.ERROR,
                    error={"message": "Internal server error", "type": "server_error"},
                )
            )
            await self._close_response_item(response_id, item_id, status="incomplete")
            await self.send_event(
                _event(
                    RealtimeEvent.RESPONSE_DONE,
                    response={
                        "id": response_id,
                        "object": "realtime.response",
                        "status": "failed",
                        "error": "Generation failed",
                    },
                )
            )
            self._response_done_emitted = True
        finally:
            if _tmp_audio_path:
                import os

                with contextlib.suppress(OSError):
                    os.unlink(_tmp_audio_path)
            if self._active_response is asyncio.current_task():
                self._active_response = None
                self._active_modalities = []
                self._cancel_event = None

    async def _handle_response_cancel(self, event: dict) -> None:
        """Handle response.cancel — abort current generation with audio truncation.

        Cancels the active response task. If audio was being streamed,
        sends response.audio.done to signal the client to truncate playback.
        Per OpenAI Realtime API protocol, audio.done must be sent BEFORE
        response.done.

        Terminal events (audio.done + response.done) are sent under the send
        lock to prevent interleaving with events from a subsequent response.create
        that arrives immediately after cancel.
        """
        task = self._active_response
        if not (task and not task.done()):
            # response.cancel with no active response was a SILENT no-op; OpenAI
            # returns an error so a client awaiting an acknowledgment doesn't hang.
            await self.send_event(
                _event(
                    RealtimeEvent.ERROR,
                    error={
                        "type": "invalid_request_error",
                        "code": "response_cancel_not_active",
                        "message": "Cancellation failed: no active response found.",
                    },
                )
            )
            return
        if task and not task.done():
            # Capture task attributes before cancelling (cancel triggers finally which
            # sets self._active_response = None).
            response_id = getattr(task, "_response_id", "")
            item_id = getattr(task, "_item_id", "")
            # Capture audio modality BEFORE awaiting the task, because the task's
            # finally block clears self._active_modalities to [].
            _had_audio = "audio" in self._active_modalities
            # Signal the cancel_event so the engine can stop mid-generation
            if self._cancel_event is not None:
                self._cancel_event.set()
            task.cancel()
            # Await the cancelled task to ensure its finally block runs before we
            # return, preventing a race with a subsequent response.create.
            with contextlib.suppress(asyncio.CancelledError):
                await task
            # (self-audit R2): if the generation task already finished its
            # happy/OOM/error path before task.cancel() took effect (it honored
            # _cancel_event and completed in the same scheduling window), it ALREADY
            # emitted a terminal response.done. Emitting our cancel terminals now would
            # send a SECOND response.done (one completed/failed + one cancelled). The
            # task swallows CancelledError with `pass` (no response.done), so
            # _response_done_emitted is the reliable discriminator (task.cancelled()
            # would be False either way). If a terminal was already sent, stop here.
            if self._response_done_emitted:
                return
            # Per OpenAI Realtime API protocol ordering:
            # 1. audio.done FIRST (if audio modality was active)
            # 2. response.done SECOND
            # Use locked send to prevent interleaving with a new response.create
            if _had_audio:
                await self._locked_send_event(
                    _event(
                        RealtimeEvent.RESPONSE_AUDIO_DONE,
                        response_id=response_id,
                        item_id=item_id,
                        output_index=0,
                        content_index=0,
                    )
                )
            # (self-audit): close the OPEN output item (added but not yet done) so a
            # cancel/barge-in doesn't leave a dangling in-progress item for SDK clients.
            await self._close_response_item(
                response_id, item_id, status="incomplete", locked=True
            )
            await self._locked_send_event(
                _event(
                    RealtimeEvent.RESPONSE_DONE,
                    response={
                        "id": response_id,
                        "object": "realtime.response",
                        "status": "cancelled",
                    },
                )
            )
            self._response_done_emitted = True  # (self-audit R2)

    async def _close_response_item(
        self,
        response_id: str,
        item_id: str,
        status: str = "incomplete",
        text: str = "",
        locked: bool = False,
    ) -> None:
        """Emit content_part.done + output_item.done for an OPEN output item exactly once.

        The lifecycle open events (output_item.added/content_part.added)
        were only closed on the happy path; cancel/barge-in/error left a dangling in-progress
        item. Idempotent via the _response_item_open flag. ``locked`` routes through
        _locked_send_event so it can't interleave with a subsequent response.create.
        """
        if not getattr(self, "_response_item_open", False):
            return
        self._response_item_open = False
        _send = self._locked_send_event if locked else self.send_event
        await _send(
            _event(
                "response.content_part.done",
                response_id=response_id,
                item_id=item_id,
                output_index=0,
                content_index=0,
                part={"type": "text", "text": text},
            )
        )
        await _send(
            _event(
                "response.output_item.done",
                response_id=response_id,
                output_index=0,
                item={
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                    "status": status,
                },
            )
        )

    async def _synthesize_audio_response(
        self,
        text: str,
        response_id: str,
        item_id: str,
        voice: str | None = None,
        out_fmt: str | None = None,
    ) -> None:
        """Synthesize text to audio and stream audio deltas as they're produced.

        Uses synthesize_stream() for token-level audio output when available,
        falling back to synthesize() with 20ms chunking. ``voice`` is the
        snapshot from response-create time so a mid-response session.update can't switch the
        voice halfway through; falls back to the live session voice for other callers.
        ``out_fmt`` is the per-response output_audio_format snapshot, threaded into
        the encoder so a per-response override wins over the session default; falls back to the
        live session format for other callers.
        """
        if voice is None:
            voice = self.session.voice
        if out_fmt is None:
            out_fmt = self.session.output_audio_format
        # fresh response → reset the g711 resampler carry so a leftover
        # from a previous (e.g. barged-in) response can't bleed into this one.
        self._g711_resample_remainder = b""
        # guarantee EXACTLY ONE terminal response.audio.done. The old code only
        # emitted it inside the synthesis loop (before break) and in the except handler — so
        # when manager is None or no entry has synthesize, the loop fell through with NO
        # audio.done and an OpenAI-SDK client tracking per-turn audio state waited forever
        # for the terminal event.
        _audio_done_sent = False

        async def _send_audio_done():
            nonlocal _audio_done_sent
            if _audio_done_sent:
                return
            _audio_done_sent = True
            try:
                await self.send_event(
                    _event(
                        RealtimeEvent.RESPONSE_AUDIO_DONE,
                        response_id=response_id,
                        item_id=item_id,
                        output_index=0,
                        content_index=0,
                    )
                )
            except Exception:
                logger.debug("Failed to send audio.done", exc_info=True)

        try:
            from ..engine import get_model_manager

            manager = get_model_manager()
            if manager is None:
                return

            import base64

            for entry in manager.list_entries():
                if entry.is_loaded and hasattr(entry.engine, "synthesize"):
                    engine = entry.engine

                    # Prefer streaming synthesis for token-level audio
                    if hasattr(engine, "synthesize_stream"):
                        _hdr_stripped = False
                        # the engine emits PCM at its REAL sample rate; capture it
                        # so _encode_output_audio resamples to the format target instead of
                        # assuming 24 kHz (a non-24 kHz TTS model otherwise plays at the wrong
                        # pitch/speed). Reset the per-response resampler state.
                        _raw_rate = getattr(engine, "sample_rate", 24000)
                        _in_rate = (
                            int(_raw_rate)
                            if isinstance(_raw_rate, (int, float)) and _raw_rate > 0
                            else 24000
                        )
                        self._pcm16_lin_state = None
                        self._g711_lin_state = None
                        async for chunk in engine.synthesize_stream(
                            text,
                            voice=voice,
                        ):
                            if chunk.get("is_final"):
                                continue
                            audio = chunk.get("audio", b"")
                            if not audio:
                                continue
                            # Strip the WAV header the first chunk carries → raw PCM16.
                            if not _hdr_stripped:
                                audio = _strip_wav_header(audio)
                                _hdr_stripped = True
                            # Convert to the negotiated output format (g711 → 8 kHz μ/A-law).
                            audio, _csz = self._encode_output_audio(
                                audio, _in_rate, fmt=out_fmt
                            )
                            # Stream in 20ms sub-chunks if the audio chunk is large
                            offset = 0
                            while offset < len(audio):
                                sub = audio[offset : offset + _csz]
                                chunk_b64 = base64.b64encode(sub).decode()
                                await self.send_event(
                                    _event(
                                        RealtimeEvent.RESPONSE_AUDIO_DELTA,
                                        response_id=response_id,
                                        item_id=item_id,
                                        output_index=0,
                                        content_index=0,
                                        delta=chunk_b64,
                                    )
                                )
                                offset += _csz
                    else:
                        _raw_rate = getattr(engine, "sample_rate", 24000)
                        _in_rate = (
                            int(_raw_rate)
                            if isinstance(_raw_rate, (int, float)) and _raw_rate > 0
                            else 24000
                        )
                        self._pcm16_lin_state = None
                        self._g711_lin_state = None
                        result = await engine.synthesize(
                            text,
                            voice=voice,
                        )
                        # AudioEngine.synthesize returns raw WAV bytes; other
                        # engine variants may return an object/dict carrying it.
                        if isinstance(result, bytes | bytearray):
                            audio_data = result
                        else:
                            audio_data = getattr(result, "audio", None)
                            if audio_data is None and isinstance(result, dict):
                                audio_data = result.get("audio")
                        if audio_data is not None:
                            raw_audio = (
                                audio_data
                                if isinstance(audio_data, bytes)
                                else bytes(audio_data)
                            )
                            raw_audio = _strip_wav_header(
                                raw_audio
                            )  # full WAV → raw PCM16
                            raw_audio, _csz = self._encode_output_audio(
                                raw_audio, _in_rate, fmt=out_fmt
                            )  # → output fmt
                            offset = 0
                            while offset < len(raw_audio):
                                chunk = raw_audio[offset : offset + _csz]
                                chunk_b64 = base64.b64encode(chunk).decode()
                                await self.send_event(
                                    _event(
                                        RealtimeEvent.RESPONSE_AUDIO_DELTA,
                                        response_id=response_id,
                                        item_id=item_id,
                                        output_index=0,
                                        content_index=0,
                                        delta=chunk_b64,
                                    )
                                )
                                offset += _csz

                    await _send_audio_done()
                    break
        except Exception as e:
            logger.error(f"TTS error in realtime session: {e}", exc_info=True)
            # Always send audio.done on error so the client is not stuck waiting
            # for a terminal audio event that will never arrive.
            await _send_audio_done()
        finally:
            # covers the no-engine / manager-None fall-through paths that
            # previously emitted no terminal event.
            await _send_audio_done()

    async def _handle_input_audio_buffer_append(self, event: dict) -> None:
        """Handle input_audio_buffer.append — receive audio chunk.

        Accumulates base64-encoded audio data in the session's audio buffer.
        If the negotiated format is g711_ulaw or g711_alaw, decodes to PCM16 first.
        If server-side VAD is enabled, runs energy-based VAD on each chunk
        to detect speech start/stop events.
        """
        audio_b64 = event.get("audio", "")
        if not audio_b64:
            return
        import base64

        chunk = base64.b64decode(audio_b64)

        # Decode g711 to PCM16 if needed (VAD and ASR expect PCM)
        fmt = self.session.input_audio_format
        if fmt == "g711_ulaw":
            chunk = self._decode_g711_ulaw(chunk)
        elif fmt == "g711_alaw":
            chunk = self._decode_g711_alaw(chunk)

        self._audio_buffer.extend(chunk)

        # cap the buffer. Over the limit → emit a protocol-visible error and
        # discard the buffer + reset VAD state (mirrors input_audio_buffer.clear), rather
        # than letting an un-committing / continuously-speaking client grow RSS unbounded.
        if len(self._audio_buffer) > _max_input_audio_bytes():
            await self.send_event(
                _event(
                    RealtimeEvent.ERROR,
                    error={
                        "message": (
                            "input_audio_buffer exceeded the maximum size; the buffer was "
                            "cleared. Commit more frequently or enable server VAD."
                        ),
                        "type": "invalid_request_error",
                        "code": "input_audio_buffer_overflow",
                    },
                )
            )
            self._audio_buffer = bytearray()
            self._vad_speaking = False
            self._vad_silence_bytes = 0
            self._reset_silero()
            return

        # Server-side VAD: energy-based detection
        turn_detection = self.session.turn_detection
        if turn_detection and turn_detection.get("type") == "server_vad":
            await self._run_vad(chunk)

    def _reset_silero(self) -> None:
        """Drop Silero streaming state so the next utterance starts fresh."""
        self._silero_state = None
        self._silero_leftover = None
        self._silero_lin_state = None

    def _silero_speech_prob(self, audio_chunk: bytes) -> float | None:
        """Speech probability [0,1] for this PCM16 chunk via Silero, or None when
        the backend is disabled/unavailable or there isn't yet a full 32 ms window
        to score. Maintains streaming state (resampler + LSTM + leftover) so windows
        are continuous across the arbitrarily-sized append chunks."""
        model = _get_silero_vad()
        if model is None:
            return None
        try:
            import numpy as np

            _in_fmt = str(
                getattr(self.session, "input_audio_format", "pcm16") or "pcm16"
            ).lower()
            in_rate = 8000 if "g711" in _in_fmt else 24000
            pcm16_16k = self._resample_pcm16_linear(
                audio_chunk, in_rate, SILERO_VAD_RATE, "_silero_lin_state"
            )
            if not pcm16_16k:
                return None
            samples = np.frombuffer(pcm16_16k, "<i2").astype(np.float32) / 32767.0
            if self._silero_leftover is not None and len(self._silero_leftover):
                samples = np.concatenate([self._silero_leftover, samples])
            cs = model._branch(SILERO_VAD_RATE).config.chunk_size
            max_prob: float | None = None
            i = 0
            while i + cs <= len(samples):
                p, self._silero_state = model.feed(
                    samples[i : i + cs], self._silero_state, SILERO_VAD_RATE
                )
                prob = float(np.array(p).reshape(-1)[0])
                max_prob = prob if max_prob is None else max(max_prob, prob)
                i += cs
            self._silero_leftover = samples[i:]  # carry the unscored remainder
            return max_prob
        except Exception:
            logger.debug(
                "Silero VAD scoring failed; falling back to energy", exc_info=True
            )
            return None

    async def _run_vad(self, audio_chunk: bytes) -> None:
        """Voice Activity Detection on an audio chunk: Silero (neural) when
        YUNSHU_REALTIME_VAD=silero, else an RMS-energy heuristic. Fires
        speech_started/speech_stopped events at transitions.

        Args:
            audio_chunk: Raw PCM bytes (16-bit signed, mono, at the input rate).
        """
        turn_detection = self.session.turn_detection
        threshold = turn_detection.get("threshold", 0.5)
        silence_duration_ms = turn_detection.get("silence_duration_ms", 500)

        num_samples = len(audio_chunk) // 2
        if num_samples == 0:
            return

        # Decide speech vs silence. Silero gives a real speech PROBABILITY, so the
        # OpenAI `threshold` (default 0.5) is used directly. The energy fallback has
        # to map it onto normalized RMS (speech ~0.03-0.1, silence below) — a 0.5
        # direct compare against RMS would never fire, so gate at threshold*0.05.
        _silero_prob = self._silero_speech_prob(audio_chunk)
        if _silero_prob is not None:
            speech = _silero_prob >= threshold
        else:
            import struct

            samples = struct.unpack(f"<{num_samples}h", audio_chunk[: num_samples * 2])
            rms = (sum(s * s for s in samples) / num_samples) ** 0.5
            speech = min(1.0, rms / 32768.0) >= threshold * 0.05

        # Offset before this chunk was appended (buffer was extended in caller)
        _pre_offset = len(self._audio_buffer) - len(audio_chunk)

        # Bytes per millisecond of the BUFFERED PCM16. g711 is decoded to 16-bit PCM
        # but stays 8 kHz (16 B/ms); native pcm16 is 24 kHz (48 B/ms). Hardcoding //48
        # made g711 speech_started/stopped report ms 3× too small (wrong client trims).
        _in_fmt = str(
            getattr(self.session, "input_audio_format", "pcm16") or "pcm16"
        ).lower()
        _bytes_per_ms = 16 if "g711" in _in_fmt else 48

        if speech:
            # Speech detected
            if not self._vad_speaking:
                self._vad_speaking = True
                self._vad_speech_start_offset = _pre_offset
                self._vad_silence_bytes = 0
                self._vad_speech_bytes = 0
                self._barge_in_fired = False
                await self.send_event(
                    _event(
                        RealtimeEvent.INPUT_AUDIO_BUFFER_SPEECH_STARTED,
                        audio_start_ms=_pre_offset // _bytes_per_ms,
                    )
                )
            # Barge-in. If the assistant is mid-response when the user speaks, INTERRUPT
            # it (cancel + response.done status=cancelled) so the user isn't talked over,
            # AND _auto_commit_and_respond (fired on the following speech_stopped) then
            # actually generates a reply — it bails out while a response is still active,
            # which previously dropped the barged-in turn entirely. Debounced on SUSTAINED
            # speech (barge_in_min_ms, default 120 ms) so a one-window noise blip / cough
            # can't kill a reply; fired at most once per utterance.
            self._vad_speech_bytes += len(audio_chunk)
            _barge_in_min_ms = turn_detection.get("barge_in_min_ms", 120)
            if (
                not self._barge_in_fired
                and self._vad_speech_bytes / _bytes_per_ms >= _barge_in_min_ms
                and self._active_response is not None
                and not self._active_response.done()
            ):
                self._barge_in_fired = True
                await self._handle_response_cancel({})
        else:
            # Silence detected
            if self._vad_speaking:
                # measure the silence window in AUDIO time (accumulated silent
                # bytes / _bytes_per_ms), NOT wall-clock. The old time.monotonic() basis made
                # turn-end detection depend on network DELIVERY rather than audio content: a
                # faster-than-realtime bulk/catch-up upload fired speech_stopped very late or
                # never, and jittery/sparse delivery fired it early, truncating the user
                # mid-pause. OpenAI server_vad measures the silence window in audio time.
                self._vad_silence_bytes += len(audio_chunk)
                if self._vad_silence_bytes / _bytes_per_ms >= silence_duration_ms:
                    # Silence duration (of actual audio) exceeded threshold — speech ended
                    self._vad_speaking = False
                    # audio_end_ms marks the END of speech (= the start of the
                    # trailing silence), which clients use to trim the committed audio.
                    # _pre_offset is the CURRENT position (start of the final silent
                    # chunk) — ~silence_duration_ms LATER than speech actually ended, so
                    # reporting it over-trims by the whole silence window. Back up by the
                    # accumulated silence (which includes this chunk, added just above) to
                    # the true speech-end offset. Capture BEFORE the reset below.
                    _speech_end = max(
                        0, _pre_offset + len(audio_chunk) - self._vad_silence_bytes
                    )
                    self._vad_silence_bytes = 0
                    await self.send_event(
                        _event(
                            RealtimeEvent.INPUT_AUDIO_BUFFER_SPEECH_STOPPED,
                            audio_end_ms=_speech_end // _bytes_per_ms,
                        )
                    )
                    # Auto-commit and trigger response (OpenAI behavior with server_vad)
                    await self._auto_commit_and_respond()
            else:
                self._vad_silence_bytes = 0

    async def _handle_input_audio_buffer_commit(
        self, event: dict, *, vad_trim: bool = False
    ) -> bool:
        """Handle input_audio_buffer.commit — finalize and transcribe audio.

                Sends the accumulated audio buffer through ASR, then adds the
                transcribed text as a user conversation item. ``vad_trim`` (server_vad
                auto-commit only) trims leading pre-speech silence to ~prefix_padding_ms
                before the VAD-detected speech start.

                Returns True iff a user conversation item was actually created (non-empty
                transcript). The server_vad auto-commit path gates the auto-response on this
                so a VAD false-trigger (noise/breath → empty transcript) or a missing ASR
                engine does NOT fire a phantom response that re-answers the previous turn
        .
        """
        # Snapshot the buffer as bytes immediately to prevent race conditions
        # where audio appended during await yields is incorrectly included.
        audio_data = bytes(self._audio_buffer)
        if not audio_data:
            # a MANUAL commit of an empty buffer must return the OpenAI
            # protocol error so the client isn't left hanging with no terminal
            # event. A server_vad auto-commit (vad_trim) on empty stays silent.
            if not vad_trim:
                await self.send_event(
                    _event(
                        RealtimeEvent.ERROR,
                        error={
                            "type": "invalid_request_error",
                            "code": "input_audio_buffer_commit_empty",
                            "message": "Cannot commit an empty input audio buffer.",
                        },
                    )
                )
            return False
        # (R6): on a VAD auto-commit, drop the leading idle/pre-speech audio
        # (keeping prefix_padding_ms before speech start) — prefix_padding_ms was
        # recorded but never applied, so seconds of accumulated silence were sent to
        # ASR every turn, degrading transcription. Manual commits keep the full buffer.
        _ss = getattr(self, "_vad_speech_start_offset", 0)
        if vad_trim and _ss > 0:
            _td = self.session.turn_detection or {}
            _pad_ms = int(_td.get("prefix_padding_ms", 300))
            _in_fmt = str(
                getattr(self.session, "input_audio_format", "pcm16") or "pcm16"
            ).lower()
            _sr = 8000 if "g711" in _in_fmt else 24000
            _pad_bytes = int(_pad_ms / 1000 * _sr) * 2  # 16-bit PCM
            _start = max(0, _ss - _pad_bytes)
            if 0 < _start < len(audio_data):
                audio_data = audio_data[_start:]

        await self.send_event(
            _event(
                RealtimeEvent.INPUT_AUDIO_BUFFER_COMMITTED,
            )
        )

        # Native-omni realtime feeds the user's RAW PCM straight to the omni model
        # (true speech-in, no ASR cascade). Stash the committed audio (post VAD-trim)
        # + its sample rate; _generate_response_omni consumes it. ASR below still runs
        # for the displayed transcript / history. Only stash when omni is enabled so the
        # cascade path never holds extra audio refs.
        if _omni_realtime_enabled():
            _omni_in_fmt = str(
                getattr(self.session, "input_audio_format", "pcm16") or "pcm16"
            ).lower()
            self._last_user_audio = (
                audio_data,
                8000 if "g711" in _omni_in_fmt else 24000,
            )

        # Transcribe via ASR engine
        tmp_path = None
        _item_created = False
        try:
            import os
            import tempfile

            # Convert raw PCM to WAV for ASR engine
            import wave

            from ..engine import get_model_manager

            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)  # Close raw fd so wave.open can write to the path
            # the sample RATE depends on the negotiated input format. g711
            # (μ-law/A-law) is 8 kHz by definition; pcm16 is 24 kHz. audio_data was
            # already decoded to PCM16, but writing 8 kHz g711 samples as 24 kHz made
            # ASR transcribe 3×-sped/garbled audio (and skewed the VAD ms math).
            _in_fmt = str(
                getattr(self.session, "input_audio_format", "pcm16") or "pcm16"
            ).lower()
            _in_sr = 8000 if "g711" in _in_fmt else 24000
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(_in_sr)
                wf.writeframes(bytes(audio_data))

            manager = get_model_manager()
            asr_found = False
            if manager is not None:
                for entry in manager.list_entries():
                    if entry.is_loaded and hasattr(entry.engine, "transcribe"):
                        asr_found = True
                        transcript = await entry.engine.transcribe(tmp_path)
                        text = (
                            transcript.get("text", "")
                            if isinstance(transcript, dict)
                            else str(transcript)
                        )
                        if text:
                            item = ConversationItem(
                                item_id=f"item_{uuid.uuid4().hex[:8]}",
                                item_type="message",
                                role="user",
                                content=[{"type": "text", "text": text}],
                            )
                            self.conversation.add_item(item)
                            _item_created = True
                            await self.send_event(
                                _event(
                                    "conversation.item.created",
                                    item=item.to_dict(),
                                )
                            )
                        break
            if not asr_found and not _omni_realtime_enabled():
                # No transcribe-capable engine loaded — emit an error event so
                # the client knows the audio buffer was committed but cannot be
                # transcribed.
                await self.send_event(
                    _event(
                        "error",
                        error={
                            "type": "invalid_request_error",
                            "code": "no_asr_engine",
                            "message": "No ASR engine available — load an ASR-capable model first",
                        },
                    )
                )
            elif not asr_found:
                # Native-omni realtime: the unified model consumes the RAW user
                # audio (already stashed above) and produces its own transcript in
                # the response. A separate ASR engine is OPTIONAL here — its absence
                # is normal, not an error. The user turn just won't carry a
                # pre-transcribed text item (the omni reply still answers the speech).
                logger.debug(
                    "No ASR engine, but native-omni realtime is on — using raw "
                    "audio as the speech-in (no input transcript)."
                )
        except Exception as e:
            logger.error(f"ASR error in realtime session: {e}", exc_info=True)
        finally:
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)

        # reset VAD state too (like the clear handler). A manual commit that
        # left _vad_speaking/_vad_silence_bytes stale could let the next quiet append
        # immediately satisfy the aged silence counter → a phantom speech_stopped +
        # auto-response on a near-empty buffer. Reset the speech-start offset too so
        # the next turn's prefix-padding trim (R6) doesn't use a stale offset.
        self._audio_buffer = bytearray()
        self._vad_speaking = False
        self._vad_silence_bytes = 0
        self._vad_speech_start_offset = 0
        self._reset_silero()
        return _item_created

    async def _handle_input_audio_buffer_clear(self, event: dict) -> None:
        """Handle input_audio_buffer.clear — discard audio buffer without processing."""
        self._audio_buffer = bytearray()
        self._vad_speaking = False
        self._vad_silence_bytes = 0
        self._reset_silero()

    async def _auto_commit_and_respond(self) -> None:
        """Auto-commit audio buffer and trigger response (server_vad mode).

        Called when VAD detects speech has ended. Commits the accumulated audio
        buffer (transcribes via ASR), then creates a response automatically.
        Matches OpenAI Realtime API behavior when turn_detection type is server_vad.

        The response availability check happens BEFORE committing the buffer so
        that audio data is not lost if a response is already in progress.
        """
        # Check response availability first — if a response is already running,
        # do NOT commit the buffer (audio would be lost with no response generated).
        if self._active_response is not None and not self._active_response.done():
            return
        # only fire the auto-response when the commit actually produced a user
        # item. A VAD false-trigger (noise/breath → empty transcript) or a missing ASR engine
        # creates no item; responding anyway would re-answer the PREVIOUS user turn (the new
        # "turn" is invisible to the model) or burn a full generation on a misconfigured
        # server. OpenAI emits response.created only after a real conversation.item.created.
        created = await self._handle_input_audio_buffer_commit(
            {"type": "input_audio_buffer.commit"}, vad_trim=True
        )
        if created:
            await self._handle_response_create(
                {"type": "response.create", "response": {}}
            )

    def _build_messages(self, instructions_override: str | None = None) -> list[dict]:
        """Build messages list from conversation items.

        ``instructions_override`` lets a per-response
        ``response.create`` carry its own ``instructions`` (a documented OpenAI
        Realtime feature, e.g. "answer in one word" for just this turn) instead
        of always using the session-level instructions.
        """
        messages = []
        # Prepend instructions as system message — per-response override wins.
        session_cfg = getattr(self, "session", None)
        _instr = instructions_override
        if _instr is None and session_cfg:
            _instr = getattr(session_cfg, "instructions", "")
        if _instr:
            messages.append({"role": "system", "content": _instr})
        for item in self.conversation.items:
            # feed function calls + tool RESULTS back to the model (they were
            # dropped, so tool calling over realtime never saw its results). Map the
            # OpenAI-Realtime item types to chat-template messages.
            if item.item_type == "function_call":
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": item.call_id or item.item_id,
                                "type": "function",
                                "function": {
                                    "name": item.name or "",
                                    "arguments": item.arguments or "{}",
                                },
                            }
                        ],
                    }
                )
                continue
            if item.item_type == "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.call_id or "",
                        "content": item.output if item.output is not None else "",
                    }
                )
                continue
            if item.item_type != "message" or not item.role:
                continue
            text_parts = []
            for content in item.content:
                if isinstance(content, dict):
                    # OpenAI Realtime API uses "input_text" for user content
                    # and "text" for assistant content. Accept both.
                    if content.get("type") in ("text", "input_text"):
                        text_parts.append(content.get("text", ""))
                    # Audio transcript fallback (when transcription available)
                    elif content.get("type") in ("input_audio", "audio"):
                        if content.get("transcript"):
                            text_parts.append(content["transcript"])
                elif isinstance(content, str):
                    text_parts.append(content)
            if text_parts:
                messages.append(
                    {
                        "role": item.role,
                        "content": "\n".join(text_parts),
                    }
                )
        return messages

    def _resolve_engine(self):
        """Resolve the inference engine for this session."""
        from ..engine import get_engine, get_model_manager

        # Try multi-model
        manager = get_model_manager()
        if manager is not None:
            # If session has a specific model set, prefer matching engine
            if self.session.model and self.session.model != "default":
                for entry in manager.list_entries():
                    if entry.is_loaded and entry.engine is not None:
                        if getattr(entry, "model_id", None) == self.session.model:
                            return entry.engine
            # Fall back to first loaded engine
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
    "conversation.item.truncate": RealtimeSession._handle_conversation_item_truncate,
    "response.create": RealtimeSession._handle_response_create,
    "response.cancel": RealtimeSession._handle_response_cancel,
    "input_audio_buffer.append": RealtimeSession._handle_input_audio_buffer_append,
    "input_audio_buffer.commit": RealtimeSession._handle_input_audio_buffer_commit,
    "input_audio_buffer.clear": RealtimeSession._handle_input_audio_buffer_clear,
}


# ── WebSocket endpoint ──


@router.websocket("/v1/realtime")
@router.websocket("/realtime")
async def realtime_endpoint(ws: WebSocket):
    """OpenAI-compatible Realtime API WebSocket endpoint.

    Exposed at both `/realtime` (legacy) and `/v1/realtime` (matches
    the OpenAI SDK URL `wss://api.openai.com/v1/realtime`). Clients
    using the official SDK will hit the /v1/ path; existing yunshu
    callers keep working on the bare /realtime path.
    """
    import os

    auth_token = os.environ.get("YUNSHU_AUTH_TOKEN")

    # Accept the WebSocket first — Starlette requires accept() before close().
    await ws.accept()

    # Origin validation: reject cross-origin WebSocket connections unless CORS is wildcard
    origin = ws.headers.get("origin", "")
    if origin:
        cors_origins_str = os.environ.get("YUNSHU_CORS_ORIGINS", "*")
        if cors_origins_str != "*":
            allowed = {
                o.strip().rstrip("/") for o in cors_origins_str.split(",") if o.strip()
            }
            origin_stripped = origin.rstrip("/")
            if origin_stripped not in allowed:
                await ws.close(code=4003, reason="Origin not allowed")
                return

    # Determine if auth is required (honor YUNSHU_AUTH_DISABLED, matching
    # the REST middleware policy in main.py). Single-consumer model: only the
    # static YUNSHU_AUTH_TOKEN gate is honored; the RBAC WebSocket path has
    # been removed.
    auth_disabled = os.environ.get("YUNSHU_AUTH_DISABLED", "").lower() in (
        "true",
        "1",
        "yes",
    )
    auth_required = bool(auth_token) and not auth_disabled

    if auth_required:
        import hmac

        token = ws.headers.get("authorization", "").removeprefix("Bearer ")
        if not token:
            token = ws.query_params.get("token")
        if not token:
            await ws.close(code=4001, reason="Authentication required")
            return

        if not (auth_token and hmac.compare_digest(token, auth_token)):
            await ws.close(code=4001, reason="Invalid token")
            return
        # Static-token holders are the single owner; no per-key model scoping.
        session = RealtimeSession(ws)
        await session.run()
        return
    session = RealtimeSession(ws)
    await session.run()
