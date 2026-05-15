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
from dataclasses import dataclass
from typing import Any, AsyncIterator

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

    def _find_llm_engine(self):
        from yunshu_gateway.engine import get_engine
        engine = get_engine()
        if engine and engine.is_loaded:
            return engine
        return None

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
        llm = self._find_llm_engine()
        if llm is None:
            raise RuntimeError("No LLM engine available")

        from yunshu_engine.batched_engine import BatchedEngine
        is_batched = isinstance(llm, BatchedEngine)
        messages = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": user_text},
        ]

        if is_batched:
            result = await llm.chat(
                messages=messages,
                max_tokens=cfg.llm_max_tokens,
                temperature=cfg.llm_temperature,
            )
            response_text = result.text
        else:
            state = await llm.generate(
                prompt=messages,
                max_tokens=cfg.llm_max_tokens,
                temperature=cfg.llm_temperature,
            )
            response_text = state.generated_text

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
        llm = self._find_llm_engine()
        if llm is None:
            raise RuntimeError("No LLM engine available")

        from yunshu_engine.batched_engine import BatchedEngine
        is_batched = isinstance(llm, BatchedEngine)
        messages = [
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": user_text},
        ]

        full_response = []
        if is_batched:
            async for output in llm.stream_chat(
                messages=messages,
                max_tokens=cfg.llm_max_tokens,
                temperature=cfg.llm_temperature,
            ):
                if output.new_text:
                    full_response.append(output.new_text)
                    yield VoicePipelineEvent(stage="llm_token", data=output.new_text)
        else:
            async for output in llm.generate_stream(
                prompt=messages,
                max_tokens=cfg.llm_max_tokens,
                temperature=cfg.llm_temperature,
            ):
                token_text = getattr(output, 'token_text', '')
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
