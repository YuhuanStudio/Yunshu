"""Tests for MultimodalPipelineCoordinator — staged multimodal pipeline."""

import time

import pytest

from yunshu_engine.staged_pipeline import (
    ModelPreprocessorRegistry,
    MultimodalPipelineCoordinator,
    PipelineRequest,
    PipelineStage,
    StageCache,
    StageConfig,
    StageStats,
    _cosyvoice_preprocessor,
    _generic_preprocessor,
    _llava_vlm_preprocessor,
    _qwen_omni_preprocessor,
    _qwen_vlm_preprocessor,
    create_default_registry,
)

# ── PipelineStage enum tests ──


class TestPipelineStage:
    def test_all_stages_defined(self):
        assert len(PipelineStage) == 7

    def test_stage_values(self):
        assert PipelineStage.TEXT_PREPROCESS.value == "text_preprocess"
        assert PipelineStage.IMAGE_PREPROCESS.value == "image_preprocess"
        assert PipelineStage.AUDIO_PREPROCESS.value == "audio_preprocess"
        assert PipelineStage.EMBEDDING_FUSION.value == "embedding_fusion"
        assert PipelineStage.PREFILL.value == "prefill"
        assert PipelineStage.DECODE.value == "decode"
        assert PipelineStage.POSTPROCESS.value == "postprocess"

    def test_stage_from_string(self):
        assert PipelineStage("text_preprocess") == PipelineStage.TEXT_PREPROCESS
        assert PipelineStage("decode") == PipelineStage.DECODE


# ── StageConfig tests ──


class TestStageConfig:
    def test_defaults(self):
        cfg = StageConfig(stage_type=PipelineStage.PREFILL, modality="text")
        assert cfg.model_id is None
        assert cfg.batch_size == 1
        assert cfg.timeout_ms == 30_000

    def test_custom_values(self):
        cfg = StageConfig(
            stage_type=PipelineStage.IMAGE_PREPROCESS,
            modality="image",
            model_id="qwen2.5-vl-7b",
            batch_size=4,
            timeout_ms=10_000,
        )
        assert cfg.model_id == "qwen2.5-vl-7b"
        assert cfg.batch_size == 4
        assert cfg.timeout_ms == 10_000

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError, match="batch_size"):
            StageConfig(stage_type=PipelineStage.DECODE, modality="text", batch_size=0)

    def test_invalid_timeout(self):
        with pytest.raises(ValueError, match="timeout_ms"):
            StageConfig(stage_type=PipelineStage.DECODE, modality="text", timeout_ms=-1)


# ── StageCache tests ──


class TestStageCache:
    def test_put_and_get(self):
        cache = StageCache()
        cache.put(PipelineStage.TEXT_PREPROCESS, "hello", {"tokens": [1, 2, 3]})
        result = cache.get(PipelineStage.TEXT_PREPROCESS, "hello")
        assert result == {"tokens": [1, 2, 3]}

    def test_cache_miss(self):
        cache = StageCache()
        assert cache.get(PipelineStage.TEXT_PREPROCESS, "missing") is None

    def test_lru_eviction(self):
        cache = StageCache(max_entries=2)
        cache.put(PipelineStage.TEXT_PREPROCESS, "a", 1)
        cache.put(PipelineStage.IMAGE_PREPROCESS, "b", 2)
        cache.put(PipelineStage.AUDIO_PREPROCESS, "c", 3)  # evicts "a"
        assert cache.get(PipelineStage.TEXT_PREPROCESS, "a") is None
        assert cache.get(PipelineStage.IMAGE_PREPROCESS, "b") == 2
        assert cache.size == 2

    def test_cache_hit_miss_tracking(self):
        cache = StageCache()
        cache.put(PipelineStage.DECODE, "x", "result")
        cache.get(PipelineStage.DECODE, "x")  # hit
        cache.get(PipelineStage.DECODE, "y")  # miss
        assert cache.hits == 1
        assert cache.misses == 1

    def test_clear(self):
        cache = StageCache()
        cache.put(PipelineStage.PREFILL, "a", 1)
        cache.get(PipelineStage.PREFILL, "a")  # hit
        cache.clear()
        assert cache.size == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_different_stage_same_inputs(self):
        cache = StageCache()
        cache.put(PipelineStage.TEXT_PREPROCESS, "hello", "text_result")
        cache.put(PipelineStage.IMAGE_PREPROCESS, "hello", "image_result")
        assert cache.get(PipelineStage.TEXT_PREPROCESS, "hello") == "text_result"
        assert cache.get(PipelineStage.IMAGE_PREPROCESS, "hello") == "image_result"


# ── MultimodalPipelineCoordinator tests ──


class TestMultimodalPipelineCoordinator:
    def _make_coordinator(self, **kwargs):
        return MultimodalPipelineCoordinator(**kwargs)

    def _make_text_request(self):
        return PipelineRequest(
            request_id="req-1",
            model_id="qwen2.5-7b",
            text="Hello world",
        )

    def _make_multimodal_request(self):
        return PipelineRequest(
            request_id="req-mm-1",
            model_id="qwen2.5-vl-7b",
            text="Describe these",
            images=["img1.png"],
            audio=["audio1.wav"],
        )

    def test_register_processor(self):
        coord = self._make_coordinator(enable_cache=False)
        called = []
        def processor(inputs, config):
            called.append(config.stage_type)
            return inputs

        coord.register_processor(PipelineStage.TEXT_PREPROCESS, "text", processor)
        results = coord.process(self._make_text_request())
        assert PipelineStage.TEXT_PREPROCESS in [r.stage for r in results]
        assert PipelineStage.TEXT_PREPROCESS in called

    def test_text_only_pipeline(self):
        coord = self._make_coordinator(enable_cache=False)
        def text_proc(inputs, config):
            return {"tokens": [1, 2, 3]}

        coord.register_processor(PipelineStage.TEXT_PREPROCESS, "text", text_proc)
        results = coord.process(self._make_text_request())
        # Should have TEXT_PREPROCESS + EMBEDDING_FUSION + PREFILL + DECODE + POSTPROCESS
        stages = [r.stage for r in results]
        assert PipelineStage.TEXT_PREPROCESS in stages

    def test_multimodal_parallel_execution(self):
        coord = self._make_coordinator(enable_cache=False)
        order = []

        def img_proc(inputs, config):
            order.append("image")
            return {"image_features": True}

        def audio_proc(inputs, config):
            order.append("audio")
            return {"audio_features": True}

        coord.register_processor(PipelineStage.IMAGE_PREPROCESS, "image", img_proc)
        coord.register_processor(PipelineStage.AUDIO_PREPROCESS, "audio", audio_proc)

        results = coord.process(self._make_multimodal_request())
        stages = [r.stage for r in results]
        assert PipelineStage.IMAGE_PREPROCESS in stages
        assert PipelineStage.AUDIO_PREPROCESS in stages

    def test_no_processor_pass_through(self):
        """Stages without registered processors pass through."""
        coord = self._make_coordinator(enable_cache=False)
        results = coord.process(self._make_text_request())
        assert len(results) > 0
        # All results should have no error and data should be the accumulated dict
        for r in results:
            assert r.error is None

    def test_stage_caching(self):
        coord = self._make_coordinator(enable_cache=True, cache_max=64)
        call_count = 0

        def processor(inputs, config):
            nonlocal call_count
            call_count += 1
            return "cached_result"

        coord.register_processor(PipelineStage.TEXT_PREPROCESS, "text", processor)

        # First request — cache miss
        r1 = coord.process(self._make_text_request())
        text_results_1 = [r for r in r1 if r.stage == PipelineStage.TEXT_PREPROCESS]
        assert any(not r.cached for r in text_results_1)

        # Second identical request — should hit cache
        r2 = coord.process(self._make_text_request())
        text_results_2 = [r for r in r2 if r.stage == PipelineStage.TEXT_PREPROCESS]
        assert any(r.cached for r in text_results_2)

    def test_clear_cache(self):
        coord = self._make_coordinator(enable_cache=True)
        def processor(inputs, config):
            return "result"
        coord.register_processor(PipelineStage.TEXT_PREPROCESS, "text", processor)
        coord.process(self._make_text_request())
        coord.clear_cache()
        stats = coord.get_stats()
        # Cache size should be 0
        assert stats[PipelineStage.TEXT_PREPROCESS.value]["cache_size"] == 0

    def test_process_stage_single(self):
        coord = self._make_coordinator(enable_cache=False)
        def processor(inputs, config):
            return {"processed": True}
        coord.register_processor(PipelineStage.PREFILL, "text", processor)

        result = coord.process_stage(PipelineStage.PREFILL, {"some": "input"}, "text")
        assert result.stage == PipelineStage.PREFILL
        assert result.data == {"processed": True}
        assert result.error is None

    def test_process_stage_error_handling(self):
        coord = self._make_coordinator(enable_cache=False)
        def failing_processor(inputs, config):
            raise RuntimeError("boom")

        coord.register_processor(PipelineStage.DECODE, "text", failing_processor)
        result = coord.process_stage(PipelineStage.DECODE, {}, "text")
        assert result.error == "boom"
        assert result.stage == PipelineStage.DECODE

    def test_process_stage_no_processor(self):
        coord = self._make_coordinator(enable_cache=False)
        result = coord.process_stage(PipelineStage.EMBEDDING_FUSION, {}, "video")
        assert result.error is not None
        assert "No processor" in result.error


# ── Stats tests ──


class TestStats:
    def test_get_stats_structure(self):
        coord = MultimodalPipelineCoordinator(enable_cache=False)
        stats = coord.get_stats()
        assert len(stats) == 7  # One per stage
        for _stage_val, stage_stats in stats.items():
            assert "call_count" in stage_stats
            assert "cache_hits" in stage_stats
            assert "avg_duration_ms" in stage_stats
            assert "error_count" in stage_stats

    def test_stats_tracking(self):
        coord = MultimodalPipelineCoordinator(enable_cache=False)
        call_count = 0

        def slow_proc(inputs, config):
            nonlocal call_count
            call_count += 1
            time.sleep(0.001)
            return "done"

        coord.register_processor(PipelineStage.TEXT_PREPROCESS, "text", slow_proc)
        coord.process(PipelineRequest(request_id="r1", model_id="m", text="hi"))
        coord.process(PipelineRequest(request_id="r2", model_id="m", text="hi2"))

        stats = coord.get_stats()
        text_stats = stats[PipelineStage.TEXT_PREPROCESS.value]
        assert text_stats["call_count"] >= 2
        assert text_stats["avg_duration_ms"] > 0

    def test_stage_stats_properties(self):
        stats = StageStats(call_count=10, cache_hits=7, cache_misses=3,
                           total_duration_ms=50.0)
        assert stats.avg_duration_ms == 5.0
        assert stats.cache_hit_rate == pytest.approx(0.7)


# ── ModelPreprocessorRegistry tests ──


class TestModelPreprocessorRegistry:
    def test_register_and_get(self):
        """Register a custom preprocessor for a known family, then retrieve it."""
        reg = ModelPreprocessorRegistry()
        def proc(inputs, config):
            return {"result": True}
        # Register with "qwen_vlm" family (has built-in detection pattern)
        reg.register("qwen_vlm", "image", proc)
        # Use a model_id that auto-detects as "qwen_vlm"
        result = reg.get_preprocessor("Qwen2.5-VL-7B-Instruct", "image")
        assert result is proc

    def test_register_custom_family_no_detection(self):
        """A custom family registered without a detection pattern falls to fallback."""
        reg = ModelPreprocessorRegistry()
        def custom_proc(inputs, config):
            return {"custom": True}
        def fallback_proc(inputs, config):
            return {"fallback": True}
        reg.register("my_custom_family", "text", custom_proc)
        reg.set_fallback(fallback_proc)
        # "random-model" won't match "my_custom_family" detection pattern
        result = reg.get_preprocessor("random-model", "text")
        assert result is fallback_proc

    def test_fallback(self):
        reg = ModelPreprocessorRegistry()
        def fallback(inputs, config):
            return {"fallback": True}
        reg.set_fallback(fallback)
        result = reg.get_preprocessor("unknown-model", "text")
        assert result is fallback

    def test_no_match_no_fallback(self):
        reg = ModelPreprocessorRegistry()
        result = reg.get_preprocessor("unknown-model", "text")
        assert result is None

    def test_detect_family_qwen_vl(self):
        assert ModelPreprocessorRegistry.detect_family("Qwen2.5-VL-7B-Instruct") == "qwen_vlm"
        assert ModelPreprocessorRegistry.detect_family("qwen2-vl-2b") == "qwen_vlm"

    def test_detect_family_llava(self):
        assert ModelPreprocessorRegistry.detect_family("llava-1.5-7b") == "llava_vlm"
        assert ModelPreprocessorRegistry.detect_family("llava-next-72b") == "llava_vlm"

    def test_detect_family_cosyvoice(self):
        assert ModelPreprocessorRegistry.detect_family("CosyVoice-300M") == "cosyvoice"

    def test_detect_family_qwen_omni(self):
        assert ModelPreprocessorRegistry.detect_family("qwen3_omni_moe") == "qwen_omni"
        assert ModelPreprocessorRegistry.detect_family("Qwen2-Audio-7B") == "qwen_omni"

    def test_detect_family_generic(self):
        assert ModelPreprocessorRegistry.detect_family("random-model-v2") == "generic"

    def test_list_registered(self):
        reg = ModelPreprocessorRegistry()
        reg.register("family_a", "text", lambda i, c: None)
        reg.register("family_b", "image", lambda i, c: None)
        listed = reg.list_registered()
        assert ("family_a", "text") in listed
        assert ("family_b", "image") in listed


# ── Built-in preprocessor tests ──


class TestBuiltinPreprocessors:
    def test_qwen_vlm_preprocessor(self):
        inputs = {"request": PipelineRequest(
            request_id="r1", model_id="qwen2.5-vl",
            text="hi", images=["img.png"])}
        result = _qwen_vlm_preprocessor(inputs, StageConfig(
            stage_type=PipelineStage.IMAGE_PREPROCESS, modality="image"))
        assert result["model_family"] == "qwen_vlm"
        assert result["image_count"] == 1

    def test_llava_vlm_preprocessor(self):
        inputs = {"request": PipelineRequest(
            request_id="r1", model_id="llava-7b",
            text="hi", images=["img1", "img2"])}
        result = _llava_vlm_preprocessor(inputs, StageConfig(
            stage_type=PipelineStage.IMAGE_PREPROCESS, modality="image"))
        assert result["model_family"] == "llava_vlm"
        assert result["image_count"] == 2

    def test_cosyvoice_preprocessor(self):
        inputs = {"request": PipelineRequest(
            request_id="r1", model_id="cosyvoice",
            text="hi", audio=["a1.wav"])}
        result = _cosyvoice_preprocessor(inputs, StageConfig(
            stage_type=PipelineStage.AUDIO_PREPROCESS, modality="audio"))
        assert result["model_family"] == "cosyvoice"
        assert result["audio_count"] == 1

    def test_qwen_omni_preprocessor(self):
        inputs = {"request": PipelineRequest(
            request_id="r1", model_id="qwen3_omni",
            text="hi", images=["img.png"], audio=["a.wav"])}
        result = _qwen_omni_preprocessor(inputs, StageConfig(
            stage_type=PipelineStage.IMAGE_PREPROCESS, modality="image"))
        assert result["model_family"] == "qwen_omni"
        assert result["image_count"] == 1
        assert result["audio_count"] == 1

    def test_generic_preprocessor(self):
        result = _generic_preprocessor(
            {}, StageConfig(stage_type=PipelineStage.DECODE, modality="video"))
        assert result["model_family"] == "generic"
        assert result["modality"] == "video"

    def test_no_images_no_audio(self):
        inputs = {"request": PipelineRequest(
            request_id="r1", model_id="qwen2.5-vl", text="hi")}
        result = _qwen_vlm_preprocessor(inputs, StageConfig(
            stage_type=PipelineStage.IMAGE_PREPROCESS, modality="image"))
        assert result["image_count"] == 0


# ── create_default_registry tests ──


class TestDefaultRegistry:
    def test_default_registry_has_builtins(self):
        reg = create_default_registry()
        listed = reg.list_registered()
        assert len(listed) >= 5  # qwen_vlm, llava_vlm, cosyvoice, qwen_omni x2

    def test_default_registry_fallback(self):
        reg = create_default_registry()
        proc = reg.get_preprocessor("unknown-model", "text")
        assert proc is not None
        result = proc({}, StageConfig(stage_type=PipelineStage.DECODE, modality="text"))
        assert result["model_family"] == "generic"

    def test_default_registry_qwen_vlm(self):
        reg = create_default_registry()
        proc = reg.get_preprocessor("Qwen2.5-VL-7B-Instruct", "image")
        assert proc is not None
        assert proc is _qwen_vlm_preprocessor

    def test_default_registry_llava(self):
        reg = create_default_registry()
        proc = reg.get_preprocessor("llava-next-72b", "image")
        assert proc is _llava_vlm_preprocessor


# ── PipelineRequest tests ──


class TestPipelineRequest:
    def test_request_creation(self):
        req = PipelineRequest(
            request_id="r1",
            model_id="test-model",
            text="Hello",
            images=["img.png"],
        )
        assert req.request_id == "r1"
        assert req.text == "Hello"
        assert len(req.images) == 1
        assert req.audio is None

    def test_request_defaults(self):
        req = PipelineRequest(request_id="r1", model_id="m")
        assert req.text is None
        assert req.images is None
        assert req.params == {}
