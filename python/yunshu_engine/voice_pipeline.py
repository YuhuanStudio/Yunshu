from __future__ import annotations

"""VoicePipeline — end-to-end STT → LLM → TTS pipeline.

Chains ASR (speech-to-text), LLM (text generation), and TTS (text-to-speech)
into a single pipeline that accepts audio input and returns audio output.

Supports:
- Full pipeline: audio → text → LLM response → speech
- Streaming: yields intermediate results (transcription, LLM tokens, audio chunks)
- Configurable: each stage can be independently configured
"""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VoicePipelineConfig:
    """Configuration for the VoicePipeline."""

    llm_model: str = ""
    tts_voice: str | None = None
    tts_speed: float = 1.0
    tts_temperature: float | None = None
    llm_temperature: float = 0.7
    llm_max_tokens: int = 256
    asr_language: str | None = None
    system_prompt: str = "You are a helpful voice assistant. Keep responses concise and natural for speech."


@dataclass
class VoicePipelineEvent:
    """Event emitted during pipeline execution."""

    stage: str  # "transcription", "llm_token", "llm_complete", "audio_chunk", "done"
    data: Any = None


class VoicePipeline:
    """STT → LLM → TTS end-to-end voice pipeline.

    Usage:
        pipeline = VoicePipeline(config)
        async for event in pipeline.process(audio_bytes):
            if event.stage == "transcription":
                print(f"User said: {event.data}")
            elif event.stage == "audio_chunk":
                # stream audio to client
                yield event.data
    """

    def __init__(self, config: VoicePipelineConfig | None = None):
        self.config = config or VoicePipelineConfig()
        self._asr_engine = None
        self._llm_engine = None
        self._tts_engine = None

    def _find_asr_engine(self):
        from yunshu_engine.audio_engine import _find_asr_engine

        return _find_asr_engine()

    def _find_tts_engine(self):
        from yunshu_engine.audio_engine import _find_tts_engine

        return _find_tts_engine()

    def _find_llm_engine(self, model_id: str | None = None):
        # Resolve the LIVE gateway.engine module (dual-package import safe —
        # see audio_engine._resolve_gateway_engine_module). A plain
        # `from yunshu_gateway.engine import ...` may bind a duplicate, empty
        # singleton when the app runs under the `python.` package prefix.
        from yunshu_engine.audio_engine import _resolve_gateway_engine_module

        _mod = _resolve_gateway_engine_module()
        get_engine = (
            getattr(_mod, "get_engine", lambda: None) if _mod else (lambda: None)
        )
        get_model_manager = (
            getattr(_mod, "get_model_manager", lambda: None) if _mod else (lambda: None)
        )
        engine = get_engine()
        if engine and engine.is_loaded:
            return engine
        # Multi-model fallback: prefer the explicit model_id, else any loaded
        # engine that supports chat/generate (LLM / BatchedEngine / VLMEngine).
        manager = get_model_manager()
        if manager is not None:
            if model_id:
                try:
                    entry = manager.get_entry(model_id)
                    if entry and entry.is_loaded and entry.engine is not None:
                        return entry.engine
                except Exception:
                    pass
            # Any loaded LLM-style engine
            for entry in manager.list_entries():
                if entry.is_loaded and entry.engine is not None:
                    if hasattr(entry.engine, "chat") or hasattr(
                        entry.engine, "generate"
                    ):
                        return entry.engine
        return None

    @staticmethod
    def _extract_text(result) -> str:
        """Pull the response text out of an LLM result of unknown shape.

        Handles a GenerationOutput-like object (.text/.generated_text/.content),
        a plain dict, or a raw string — robust to engine-type variation.
        """
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return (
                result.get("text")
                or result.get("generated_text")
                or result.get("content")
                or ""
            )
        for attr in ("text", "generated_text", "content"):
            val = getattr(result, attr, None)
            if val:
                return val
        return ""

    @staticmethod
    def _messages_to_prompt(engine, messages: list[dict]) -> str:
        """Convert messages to a string prompt for non-batched engines.

        Tries apply_chat_template first, falls back to simple concatenation.
        """
        tokenizer = getattr(engine, "_tokenizer", None)
        if tokenizer and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # Fallback: concatenate message content
        return "\n".join(m.get("content", str(m)) for m in messages)

    async def process(
        self,
        audio_path: str,
        config: VoicePipelineConfig | None = None,
    ) -> dict:
        """Run the full STT → LLM → TTS pipeline, return final result."""
        cfg = config or self.config

        # Stage 1: STT
        asr = self._find_asr_engine()
        if asr is None:
            raise RuntimeError("No ASR engine available")
        transcription = await asr.transcribe(audio_path, language=cfg.asr_language)
        user_text = transcription.get("text", "")
        if not user_text.strip():
            return {"text": "", "audio": b"", "transcription": transcription}

        # Stage 2: LLM
        llm = self._find_llm_engine(model_id=cfg.llm_model)
        if llm is None:
            raise RuntimeError("No LLM engine available")

        messages = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": user_text},
        ]

        # Duck-type the engine: isinstance(llm, BatchedEngine) is unreliable
        # because dual-package imports (python.yunshu_engine vs yunshu_engine)
        # can yield two distinct BatchedEngine class objects. Prefer chat()
        # (applies the template internally); fall back to generate() with a
        # manually templated prompt. Extract text robustly from the result,
        # which may be a GenerationOutput object, a dict, or a string.
        if hasattr(llm, "chat"):
            result = await llm.chat(
                messages=messages,
                max_tokens=cfg.llm_max_tokens,
                temperature=cfg.llm_temperature,
            )
            response_text = self._extract_text(result)
        else:
            prompt_text = self._messages_to_prompt(llm, messages)
            state = await llm.generate(
                prompt=prompt_text,
                max_tokens=cfg.llm_max_tokens,
                temperature=cfg.llm_temperature,
            )
            response_text = self._extract_text(state)

        # Stage 3: TTS
        tts = self._find_tts_engine()
        audio_bytes = b""
        if tts and response_text.strip():
            audio_bytes = await tts.synthesize(
                text=response_text,
                voice=cfg.tts_voice,
                speed=cfg.tts_speed,
                temperature=cfg.tts_temperature,
            )

        return {
            "text": response_text,
            "audio": audio_bytes,
            "transcription": transcription,
        }

    async def process_stream(
        self,
        audio_path: str,
        config: VoicePipelineConfig | None = None,
    ) -> AsyncIterator[VoicePipelineEvent]:
        """Stream the full STT → LLM → TTS pipeline with intermediate events."""
        cfg = config or self.config

        # Stage 1: STT
        asr = self._find_asr_engine()
        if asr is None:
            raise RuntimeError("No ASR engine available")

        transcription = await asr.transcribe(audio_path, language=cfg.asr_language)
        user_text = transcription.get("text", "")
        yield VoicePipelineEvent(stage="transcription", data=user_text)

        if not user_text.strip():
            yield VoicePipelineEvent(stage="done")
            return

        # Stage 2: LLM streaming
        llm = self._find_llm_engine(model_id=cfg.llm_model)
        if llm is None:
            raise RuntimeError("No LLM engine available")

        messages = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": user_text},
        ]

        # Duck-type (dual-package imports break isinstance — see process()).
        full_response = []
        if hasattr(llm, "stream_chat"):
            async for output in llm.stream_chat(
                messages=messages,
                max_tokens=cfg.llm_max_tokens,
                temperature=cfg.llm_temperature,
            ):
                if output.new_text:
                    full_response.append(output.new_text)
                    yield VoicePipelineEvent(stage="llm_token", data=output.new_text)
        else:
            # Non-batched engines require a string prompt
            prompt_text = self._messages_to_prompt(llm, messages)
            async for output in llm.generate_stream(
                prompt=prompt_text,
                max_tokens=cfg.llm_max_tokens,
                temperature=cfg.llm_temperature,
            ):
                token_text = getattr(output, "token_text", "")
                if token_text:
                    full_response.append(token_text)
                    yield VoicePipelineEvent(stage="llm_token", data=token_text)

        response_text = "".join(full_response)
        yield VoicePipelineEvent(stage="llm_complete", data=response_text)

        # Stage 3: TTS
        tts = self._find_tts_engine()
        if tts and response_text.strip():
            audio_bytes = await tts.synthesize(
                text=response_text,
                voice=cfg.tts_voice,
                speed=cfg.tts_speed,
                temperature=cfg.tts_temperature,
            )
            yield VoicePipelineEvent(stage="audio_chunk", data=audio_bytes)

        yield VoicePipelineEvent(stage="done")
