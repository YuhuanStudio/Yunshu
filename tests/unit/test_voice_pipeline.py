"""Tests for VoicePipeline (STT → LLM → TTS)."""
import pytest


class TestVoicePipelineConfig:
    def test_defaults(self):
        from yunshu_engine.voice_pipeline import VoicePipelineConfig
        cfg = VoicePipelineConfig()
        assert cfg.llm_temperature == 0.7
        assert cfg.llm_max_tokens == 256
        assert cfg.tts_speed == 1.0
        assert cfg.asr_language is None

    def test_custom_config(self):
        from yunshu_engine.voice_pipeline import VoicePipelineConfig
        cfg = VoicePipelineConfig(
            llm_model="test-model",
            tts_voice="alloy",
            llm_temperature=0.3,
            llm_max_tokens=512,
            asr_language="en",
        )
        assert cfg.llm_model == "test-model"
        assert cfg.tts_voice == "alloy"
        assert cfg.llm_temperature == 0.3
        assert cfg.asr_language == "en"


class TestVoicePipelineEvent:
    def test_transcription_event(self):
        from yunshu_engine.voice_pipeline import VoicePipelineEvent
        event = VoicePipelineEvent(stage="transcription", data="Hello world")
        assert event.stage == "transcription"
        assert event.data == "Hello world"

    def test_llm_token_event(self):
        from yunshu_engine.voice_pipeline import VoicePipelineEvent
        event = VoicePipelineEvent(stage="llm_token", data="Hi")
        assert event.stage == "llm_token"

    def test_done_event(self):
        from yunshu_engine.voice_pipeline import VoicePipelineEvent
        event = VoicePipelineEvent(stage="done")
        assert event.stage == "done"
        assert event.data is None


class TestVoicePipelineInit:
    def test_default_config(self):
        from yunshu_engine.voice_pipeline import VoicePipeline, VoicePipelineConfig
        pipeline = VoicePipeline()
        assert isinstance(pipeline.config, VoicePipelineConfig)

    def test_custom_config(self):
        from yunshu_engine.voice_pipeline import VoicePipeline, VoicePipelineConfig
        cfg = VoicePipelineConfig(llm_temperature=0.5)
        pipeline = VoicePipeline(cfg)
        assert pipeline.config.llm_temperature == 0.5

    def test_no_engines_by_default(self):
        from yunshu_engine.voice_pipeline import VoicePipeline
        pipeline = VoicePipeline()
        assert pipeline._asr_engine is None
        assert pipeline._llm_engine is None
        assert pipeline._tts_engine is None
