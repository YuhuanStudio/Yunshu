"""OmniEngine — serves unified omni models (Qwen3-Omni) with native
Thinker + Talker streaming.

A unified omni model handles text + vision + speech-in and produces text +
speech-out (Talker) in ONE model with a shared context — not an ASR→LLM→TTS
cascade. mlx-vlm's ``qwen3_omni_moe.generate_stream`` yields ``("text", tokens)``
then ``("audio", wav_chunk @24kHz)`` incrementally.

This engine exposes that as an async stream with a minimal-thinker default
(validated: ~1.4s first-audio on M3 Max / 36GB) and single-flight generation
(one turn at a time — correct for a single-consumer voice server). A self-loaded
model runs on the calling thread's default Metal stream; a model *reused* from the
gateway (one omni model, two endpoints, no second copy) runs on the shared MLX
executor thread that owns it, inside generation_stream.

This is the forward differentiation: the only MLX server exposing Qwen3-Omni's
native speech-out.
"""

from __future__ import annotations

import asyncio
import logging
import os
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
    "alloy": "ethan",
    "echo": "ethan",
    "onyx": "ethan",
    "ash": "ethan",
    "ballad": "ethan",
    "sage": "ethan",
    "verse": "ethan",
    "fable": "ethan",
    "nova": "chelsie",
    "shimmer": "chelsie",
    "coral": "chelsie",
}


def _resolve_speaker(
    requested: str | None,
    default: str,
    valid_speakers: set[str] = _OMNI_SPEAKERS,
) -> str:
    """Map a requested voice/speaker to a valid Talker speaker, falling back to
    ``default`` (never raising) so a stray voice name can't crash generation.

    ``valid_speakers`` is the model's own speaker set (lowercased); it defaults
    to the Qwen3-Omni set so the module-level helper stays usable without a
    loaded model, but ``OmniEngine`` passes the set it derived at load time so
    other Talker models with different speakers work too."""
    if not requested:
        return default
    key = requested.strip().lower()
    if key in valid_speakers:
        return key.capitalize()
    if key in _VOICE_ALIASES and _VOICE_ALIASES[key] in valid_speakers:
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
        model_path: str | None = None,
        speaker: SpeakerName = "Ethan",
        # Thinker budget = the spoken reply's max length (the Talker speaks what the
        # Thinker writes). 8 was a latency micro-opt but truncated real answers mid-
        # sentence ("讲个故事" → it never tells the story) and pushed the model into
        # generic canned replies. Default to a conversational budget; a short reply
        # still stops at its natural EOS, so this only *allows* longer answers. Tune
        # with YUNSHU_OMNI_THINKER_MAX (lower = snappier/shorter, higher = longer).
        thinker_max_new_tokens: int | None = None,
        talker_max_new_tokens: int = 1024,
        talker_temperature: float = 0.9,
        chunk_size: int = 10,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        *,
        model: Any = None,
        processor: Any = None,
    ) -> None:
        # Either load from ``model_path`` OR adopt an already-loaded
        # ``(model, processor)`` — the latter lets the voice path reuse the model
        # the gateway already serves (one omni model, two endpoints, no 2nd copy).
        self.model_path = model_path
        self.speaker = speaker
        self.thinker_max = (
            thinker_max_new_tokens
            if thinker_max_new_tokens is not None
            else int(os.environ.get("YUNSHU_OMNI_THINKER_MAX", "256"))
        )
        self.talker_max = talker_max_new_tokens
        self.talker_temp = talker_temperature
        self.chunk_size = chunk_size
        # Talker output rate — single source of truth consumers (the SSE endpoint
        # and the realtime resampler) read instead of hardcoding 24000. Derived
        # from the model at load() when possible; 24 kHz is the Qwen3-Omni rate.
        self.sample_rate = sample_rate
        self.model: Any = model  # mlx-vlm omni model (dynamic; may be injected)
        self.processor: Any = processor
        self._shared = model is not None  # reusing a model we don't own
        self._setup_done = False  # talker check + speakers + kernel prime ran
        self._prev_text_ids: list[int] = []
        # When reusing the gateway's model, its weights live on the shared MLX
        # executor thread (which owns generation_stream / Stream(gpu,1)); all GPU
        # work on it MUST run there. Lazily resolved (kept None for self-loaded).
        self._executor = None
        # Valid Talker speakers — defaults to the Qwen3-Omni set, replaced at
        # load() with the loaded model's own speaker map (model-agnostic).
        self._valid_speakers: set[str] = set(_OMNI_SPEAKERS)
        self._busy = asyncio.Lock()

    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def valid_speakers(self) -> set[str]:
        """The loaded model's speaker set (lowercased). Until load(), the
        Qwen3-Omni default set."""
        return set(self._valid_speakers)

    def resolve_speaker(self, requested: str | None) -> str | None:
        """Resolve a requested voice to a canonical Talker speaker.

        Returns the default speaker when ``requested`` is empty, the canonical
        capitalized name when it's a valid speaker or a known alias, and ``None``
        when a non-empty ``requested`` is unrecognized — letting the API layer
        return a 400 (vs _normalize_speaker, which silently falls back so
        generation can't crash on a stray name)."""
        if not requested:
            return self.speaker
        key = requested.strip().lower()
        if key in self._valid_speakers:
            return key.capitalize()
        if key in _VOICE_ALIASES and _VOICE_ALIASES[key] in self._valid_speakers:
            return _VOICE_ALIASES[key].capitalize()
        return None

    def load(self) -> None:
        """Load on the calling thread (owns the default Metal stream). Idempotent.

        When a model was injected (reuse of the already-served model) we skip the
        load and just run the one-time setup (talker check, speaker map, kernel
        prime) on it — no second copy of the weights."""
        if self._setup_done:
            return
        if self.model is None:
            if not self.model_path:
                raise ValueError(
                    "OmniEngine needs a model_path or an injected model+processor."
                )
            from mlx_vlm import load as vlmload

            logger.info("OmniEngine loading %s …", self.model_path)
            self.model, self.processor = vlmload(
                self.model_path, trust_remote_code=True
            )
        else:
            logger.info("OmniEngine reusing the already-loaded served model (no copy).")
        if not getattr(self.model, "has_talker", False):
            raise ValueError(
                f"{self.model_path} has no Talker — not a unified-omni model. "
                "OmniEngine requires a Thinker+Talker model (e.g. Qwen3-Omni)."
            )
        # Derive the model's own speaker set (model-agnostic, not the hardcoded
        # Qwen names) from config.talker_config.speaker_id; keep the default set
        # as a fallback for models that don't publish one.
        talker_cfg = getattr(getattr(self.model, "config", None), "talker_config", None)
        speaker_map = getattr(talker_cfg, "speaker_id", None) if talker_cfg else None
        if speaker_map:
            self._valid_speakers = {str(s).lower() for s in speaker_map}
            logger.info("OmniEngine speakers: %s", sorted(self._valid_speakers))
        # Self-loaded: prime the Thinker MoE kernels here (cold-start hang guard).
        # Shared: the gateway already warmed the Thinker for text, and this would
        # touch the GPU off the executor thread that owns the model — so skip it;
        # the Talker primes on the first stream (which runs on the executor).
        if not self._shared:
            self._compile_kernels()
        self._setup_done = True
        logger.info(
            "OmniEngine ready (talker present%s).", ", reused" if self._shared else ""
        )

    def _compile_kernels(self) -> None:
        """Pre-compile the Thinker's MoE kernels with a throwaway forward BEFORE
        any ``generate_stream``.

        On the 30B-A3B MoE, ``generate_step``'s *first* command buffer otherwise
        JITs every kernel inline and overruns Metal's GPU watchdog
        (``kIOGPUCommandBufferCallbackErrorHang``) — the cold generation simply
        hangs. A plain parallel forward (one bounded command buffer) compiles
        those kernels safely; subsequent generation buffers then stay under the
        watchdog. Verified on M3 Max: cold ``generate_stream`` hangs; a single
        forward first → full Thinker+Talker audio. Best-effort; ``load`` only
        reaches here once."""
        import mlx.core as mx

        try:
            conv = [{"role": "user", "content": _build_content("Hi.", None, None)}]
            mi, _ = _prepare_inputs(self.processor, conv)
            for _ in range(2):
                out = self.model.thinker.language_model(mi["input_ids"])
                mx.eval(out.logits)
            logger.info("OmniEngine kernels pre-compiled (cold-start hang avoided).")
        except Exception as exc:  # noqa: BLE001 — priming is best-effort
            logger.warning(
                "OmniEngine kernel pre-compile failed (%s); the first generation "
                "may hang on large MoE models (Metal GPU watchdog).",
                exc,
            )

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
        not benchmarking sleight-of-hand.

        The text passes do NOT compile the audio-ENCODER kernels (those only run
        when speech is fed in). For a speech-to-speech server that leaves the very
        first audio-in turn paying a ~0.5s JIT tax (measured: 1.8s vs 1.3s steady
        on M3 Max). So warmup also runs ONE audio-in pass with a short throwaway
        clip, priming the encoder path too."""
        # Kernels compile on FIRST use regardless of reply length, so warmup caps
        # the Thinker hard: priming a full ~256-token reply just to JIT the kernels
        # made boot needlessly slow (a 256-token reply is ~1000 Talker steps).
        warmup_thinker_max = 8
        start = asyncio.get_running_loop().time()
        self.load()
        for _ in range(max(1, rounds)):
            async for _chunk in self.stream(
                "Hello, please say a short greeting out loud.",
                thinker_max_new_tokens=warmup_thinker_max,
            ):
                pass  # discard — we only want kernels compiled and the path primed
        # One audio-in pass to compile the speech-encoder kernels (best-effort).
        audio_path = self._make_warmup_audio()
        if audio_path is not None:
            try:
                async for _chunk in self.stream(
                    "Respond to the user.",
                    audio_path=audio_path,
                    thinker_max_new_tokens=warmup_thinker_max,
                ):
                    pass
            except Exception:  # noqa: BLE001 — encoder priming is best-effort
                logger.warning(
                    "OmniEngine audio-in warmup failed; first speech turn may be cold",
                    exc_info=True,
                )
            finally:
                import contextlib
                import os

                with contextlib.suppress(OSError):
                    os.unlink(audio_path)
        elapsed = asyncio.get_running_loop().time() - start
        logger.info(
            "OmniEngine warmup done in %.1fs (first request now warm).", elapsed
        )
        return elapsed

    def _make_warmup_audio(self) -> str | None:
        """Write a short, near-silent throwaway WAV for audio-encoder priming.
        Content is irrelevant — only the encoder kernels (shape-dependent) need
        to compile. Returns the temp path, or None if writing fails."""
        import tempfile
        import wave

        try:
            rate = 16000
            samples = np.zeros(rate // 2, dtype="<i2")  # 0.5s silence
            fd, path = tempfile.mkstemp(suffix=".wav")
            import os

            os.close(fd)
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(rate)
                wf.writeframes(samples.tobytes())
            return path
        except Exception:  # noqa: BLE001
            logger.debug("warmup audio synthesis failed", exc_info=True)
            return None

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
        # Load on the thread that owns the model: the shared executor for a reused
        # model, the calling thread for a self-loaded one.
        loop = asyncio.get_running_loop()
        if self._shared:
            await loop.run_in_executor(self._get_executor(), self.load)
        else:
            self.load()
        async with self._busy:  # one generation at a time
            spk = _resolve_speaker(
                speaker or self.speaker, self.speaker, self._valid_speakers
            )
            tmax = (
                thinker_max_new_tokens
                if thinker_max_new_tokens is not None
                else self.thinker_max
            )
            conv = [
                {
                    "role": "user",
                    "content": _build_content(text, image_path, audio_path),
                }
            ]
            self._prev_text_ids = []
            start = loop.time()
            first_audio: float | None = None
            audio_samples = 0
            async for kind, data in self._iter_materialized(conv, spk, tmax):
                now = loop.time() - start
                if kind == "text":
                    if data:
                        yield OmniChunk("text", data, now)
                elif kind == "audio":
                    if first_audio is None:
                        first_audio = now
                    audio_samples += len(data)
                    yield OmniChunk("audio", data, now)
            yield OmniChunk(
                "done",
                {
                    "first_audio_s": first_audio,
                    "audio_seconds": audio_samples / self.sample_rate,
                    "total_s": loop.time() - start,
                },
                loop.time() - start,
            )

    def _get_executor(self):
        """The shared single-thread MLX executor (owns generation_stream). Lazily
        resolved so OmniEngine construction stays side-effect-free."""
        if self._executor is None:
            from .mlx_executor import get_mlx_executor

            self._executor = get_mlx_executor()
        return self._executor

    def _make_gen(self, conv: list[dict], spk: str, tmax: int):
        """Build the mlx_vlm Thinker→Talker generator for one turn."""
        mi, _ = _prepare_inputs(self.processor, conv)
        return self.model.generate_stream(
            mi["input_ids"],
            speaker=spk,
            thinker_max_new_tokens=tmax,
            talker_max_new_tokens=self.talker_max,
            talker_temperature=self.talker_temp,
            chunk_size=self.chunk_size,
            **{
                k: v
                for k, v in {
                    "input_features": mi.get("input_features"),
                    "feature_attention_mask": mi.get("feature_attention_mask"),
                    "audio_feature_lengths": mi.get("audio_feature_lengths"),
                    "pixel_values": mi.get("pixel_values"),
                    "pixel_values_videos": mi.get("pixel_values_videos"),
                    "image_grid_thw": mi.get("image_grid_thw"),
                    "video_grid_thw": mi.get("video_grid_thw"),
                }.items()
                if v is not None
            },
        )

    async def _iter_materialized(self, conv: list[dict], spk: str, tmax: int):
        """Yield (kind, cpu_data) with mx arrays materialized to CPU (str for text,
        numpy for audio). A reused model lives on the shared executor thread (which
        owns generation_stream / Stream(gpu,1)), so its generation AND its
        materialization run there; a self-loaded model runs inline on this thread."""
        if not self._shared:
            for kind, payload in self._make_gen(conv, spk, tmax):
                if kind == "text":
                    yield ("text", self._decode_fragment(payload))
                elif kind == "audio":
                    yield ("audio", np.asarray(payload, dtype=np.float32).reshape(-1))
            return

        import mlx.core as mx
        from mlx_lm.generate import generation_stream

        loop = asyncio.get_running_loop()
        ex = self._get_executor()

        def _start():
            with mx.stream(generation_stream):
                return self._make_gen(conv, spk, tmax)

        gen = await loop.run_in_executor(ex, _start)
        _DONE = object()

        def _step():
            with mx.stream(generation_stream):
                try:
                    kind, payload = next(gen)
                except StopIteration:
                    return _DONE
                if kind == "text":
                    return ("text", self._decode_fragment(payload))
                if kind == "audio":
                    return ("audio", np.asarray(payload, dtype=np.float32).reshape(-1))
                return (kind, None)

        while True:
            item = await loop.run_in_executor(ex, _step)
            if item is _DONE:
                return
            yield item

    def _decode_fragment(self, payload) -> str:
        """generate_stream yields the accumulated token-id sequence on each
        'text' event; emit only the newly-added tail as decoded text."""
        ids = list(payload.tolist()) if hasattr(payload, "tolist") else list(payload)
        new_ids = ids[len(self._prev_text_ids) :]
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


def _build_content(
    text: str, image_path: str | None, audio_path: str | None
) -> list[dict]:
    c: list[dict] = []
    if audio_path:
        c.append({"type": "audio", "audio": audio_path})
    if image_path:
        c.append({"type": "image", "image": image_path})
    c.append({"type": "text", "text": text})
    return c
