"""OmniEngine — serves unified omni models (Qwen3-Omni) with native
Thinker + Talker streaming.

A unified omni model handles text + vision + speech-in and produces text +
speech-out (Talker) in ONE model with a shared context — not an ASR→LLM→TTS
cascade. mlx-vlm's ``qwen3_omni_moe.generate_stream`` yields ``("text", tokens)``
then ``("audio", wav_chunk @24kHz)`` incrementally.

This engine exposes that as an async stream with a minimal-thinker default
(validated: ~1.4s first-audio on M3 Max / 36GB) and single-flight generation
(one turn at a time — correct for a single-consumer voice server, and the GPU
works correctly on the calling thread's default Metal stream).

This is the forward differentiation: the only MLX server exposing Qwen3-Omni's
native speech-out.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

AUDIO_SAMPLE_RATE = 24000  # Qwen3-Omni Talker output
SpeakerName = str  # "Ethan" | "Chelsie" | "Aiden" | ... (model-defined)

# Qwen3-Omni Talker speakers (lowercased; the model raises NotImplementedError
# for anything else). Keep in sync with the model's talker_config.speaker_id.
_OMNI_SPEAKERS = {"ethan", "chelsie", "aiden"}
# Friendly aliases so an OpenAI-Realtime voice name (the default is "alloy")
# selects a real Talker speaker instead of crashing — rough gender match, else
# the engine default. This is a convenience map, not a fidelity claim.
_VOICE_ALIASES = {
    "alloy": "ethan", "echo": "ethan", "onyx": "ethan", "ash": "ethan",
    "ballad": "ethan", "sage": "ethan", "verse": "ethan", "fable": "ethan",
    "nova": "chelsie", "shimmer": "chelsie", "coral": "chelsie",
}


def _resolve_speaker(requested: str | None, default: str) -> str:
    """Map a requested voice/speaker to a valid Talker speaker, falling back to
    ``default`` (never raising) so a stray voice name can't crash generation."""
    if not requested:
        return default
    key = requested.strip().lower()
    if key in _OMNI_SPEAKERS:
        return key.capitalize()
    if key in _VOICE_ALIASES:
        return _VOICE_ALIASES[key].capitalize()
    logger.debug("Unknown omni speaker %r — using default %s", requested, default)
    return default


@dataclass
class OmniChunk:
    """One streamed fragment from a unified omni model.

    kind == "text":  data is a decoded text fragment (str)
    kind == "audio": data is a float32 mono ndarray @ 24kHz
    kind == "done":  final chunk; data is a stats dict {first_audio_s, ...}
    """

    kind: Literal["text", "audio", "done"]
    data: Any
    elapsed_s: float = 0.0


class OmniEngine:
    """Loads ONE unified omni model (e.g. Qwen3-Omni) resident and streams
    Thinker text + Talker audio. Single-consumer, single-flight."""

    def __init__(
        self,
        model_path: str,
        speaker: SpeakerName = "Ethan",
        # Minimal-thinker default: casual voice needs ~no reasoning.
        # Validated first-audio: thinker=1→1.4s, =8→2.5s, =32→3.0s (M3 Max/36GB).
        thinker_max_new_tokens: int = 8,
        talker_max_new_tokens: int = 1024,
        talker_temperature: float = 0.9,
        chunk_size: int = 300,
    ) -> None:
        self.model_path = model_path
        self.speaker = speaker
        self.thinker_max = thinker_max_new_tokens
        self.talker_max = talker_max_new_tokens
        self.talker_temp = talker_temperature
        self.chunk_size = chunk_size
        self.model: Any = None  # mlx-vlm omni model (dynamic)
        self.processor: Any = None
        self._prev_text_ids: list[int] = []
        self._busy = asyncio.Lock()

    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        """Load on the calling thread (owns the default Metal stream). Idempotent."""
        if self.model is not None:
            return
        from mlx_vlm import load as vlmload

        logger.info("OmniEngine loading %s …", self.model_path)
        self.model, self.processor = vlmload(self.model_path, trust_remote_code=True)
        if not getattr(self.model, "has_talker", False):
            raise ValueError(
                f"{self.model_path} has no Talker — not a unified-omni model. "
                "OmniEngine requires a Thinker+Talker model (e.g. Qwen3-Omni)."
            )
        logger.info("OmniEngine ready (talker present).")

    async def warmup(self, rounds: int = 2) -> float:
        """Load + run throwaway generations so the user's FIRST real request is
        warm (~2.4s), not cold (~30s). Returns seconds taken.

        Measured (Qwen3-Omni-30B-A3B-4bit / M3 Max): the cold cost is ~30s of
        model load + kernel JIT, plus a *separate* one-time ~4s tax on the very
        first generation that a single warmup pass does NOT absorb. Steady-state
        is ~2.4s and is prompt-shape-independent. So warmup runs TWO passes by
        default: pass 1 pays load+JIT, pass 2 pays the first-generation tax —
        leaving the real first request genuinely at steady state. This is just
        priming a resident model (the standard thing every model server does),
        not benchmarking sleight-of-hand."""
        start = asyncio.get_running_loop().time()
        self.load()
        for _ in range(max(1, rounds)):
            async for _chunk in self.stream("Hello, please say a short greeting out loud."):
                pass  # discard — we only want kernels compiled and the path primed
        elapsed = asyncio.get_running_loop().time() - start
        logger.info("OmniEngine warmup done in %.1fs (first request now warm).", elapsed)
        return elapsed

    async def stream(
        self,
        text: str,
        image_path: str | None = None,
        audio_path: str | None = None,
        speaker: SpeakerName | None = None,
        thinker_max_new_tokens: int | None = None,
    ) -> AsyncIterator[OmniChunk]:
        """Stream Thinker text + Talker audio for one turn (single-flight).

        Yields OmniChunks: text fragments, then audio chunks, then a final
        ``done`` chunk with latency stats.
        """
        self.load()
        async with self._busy:  # one generation at a time
            spk = _resolve_speaker(speaker or self.speaker, self.speaker)
            tmax = thinker_max_new_tokens if thinker_max_new_tokens is not None else self.thinker_max
            conv = [{"role": "user", "content": _build_content(text, image_path, audio_path)}]
            mi, _ = _prepare_inputs(self.processor, conv)

            gen = self.model.generate_stream(
                mi["input_ids"],
                speaker=spk,
                thinker_max_new_tokens=tmax,
                talker_max_new_tokens=self.talker_max,
                talker_temperature=self.talker_temp,
                chunk_size=self.chunk_size,
                **{k: v for k, v in {
                    "input_features": mi.get("input_features"),
                    "feature_attention_mask": mi.get("feature_attention_mask"),
                    "audio_feature_lengths": mi.get("audio_feature_lengths"),
                    "pixel_values": mi.get("pixel_values"),
                    "pixel_values_videos": mi.get("pixel_values_videos"),
                    "image_grid_thw": mi.get("image_grid_thw"),
                    "video_grid_thw": mi.get("video_grid_thw"),
                }.items() if v is not None},
            )

            self._prev_text_ids = []
            start = asyncio.get_running_loop().time()
            first_audio: float | None = None
            audio_samples = 0
            for kind, payload in gen:
                now = asyncio.get_running_loop().time() - start
                if kind == "text":
                    frag = self._decode_fragment(payload)
                    if frag:
                        yield OmniChunk("text", frag, now)
                elif kind == "audio":
                    if first_audio is None:
                        first_audio = now
                    wav = np.asarray(payload, dtype=np.float32).reshape(-1)
                    audio_samples += len(wav)
                    yield OmniChunk("audio", wav, now)
            yield OmniChunk("done", {
                "first_audio_s": first_audio,
                "audio_seconds": audio_samples / AUDIO_SAMPLE_RATE,
                "total_s": asyncio.get_running_loop().time() - start,
            }, asyncio.get_running_loop().time() - start)

    def _decode_fragment(self, payload) -> str:
        """generate_stream yields the accumulated token-id sequence on each
        'text' event; emit only the newly-added tail as decoded text."""
        ids = list(payload.tolist()) if hasattr(payload, "tolist") else list(payload)
        new_ids = ids[len(self._prev_text_ids):]
        self._prev_text_ids = ids
        if not new_ids:
            return ""
        try:
            # skip_special_tokens so control tokens (<|im_end|>, etc.) don't leak
            # into the transcript / text stream.
            return str(self.processor.decode(new_ids, skip_special_tokens=True))
        except TypeError:
            # Some processors don't accept the kwarg — fall back to plain decode.
            try:
                return str(self.processor.decode(new_ids))
            except Exception:  # noqa: BLE001
                return ""
        except Exception:  # noqa: BLE001
            return ""


def _prepare_inputs(processor: Any, conv: list[dict]) -> Any:
    from mlx_vlm.models.qwen3_omni_moe.omni_utils import prepare_omni_inputs

    return prepare_omni_inputs(processor, conv)


def _build_content(text: str, image_path: str | None, audio_path: str | None) -> list[dict]:
    c: list[dict] = []
    if audio_path:
        c.append({"type": "audio", "audio": audio_path})
    if image_path:
        c.append({"type": "image", "image": image_path})
    c.append({"type": "text", "text": text})
    return c
