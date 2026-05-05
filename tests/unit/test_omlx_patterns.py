"""Tests for oMLX pattern replication modules."""
import json
import tempfile
import threading
from pathlib import Path

import pytest


class TestHardwareDetection:
    def test_detect_hardware(self):
        from yunshu_engine.utils.hardware import detect_hardware
        hw = detect_hardware()
        assert hw.chip_name
        assert hw.total_memory_gb > 0
        assert hw.max_working_set_bytes > 0

    def test_format_bytes(self):
        from yunshu_engine.utils.hardware import format_bytes
        assert format_bytes(0) == "0 B"
        assert "GB" in format_bytes(16 * 1024 ** 3)
        assert "MB" in format_bytes(512 * 1024 ** 2)
        assert "KB" in format_bytes(1024)

    def test_parse_chip_info(self):
        from yunshu_engine.utils.hardware import parse_chip_info
        assert parse_chip_info("Apple M4 Pro") == ("M4", "Pro")
        assert parse_chip_info("Apple M3 Max") == ("M3", "Max")
        assert parse_chip_info("Apple M2") == ("M2", "")

    def test_is_apple_silicon(self):
        from yunshu_engine.utils.hardware import is_apple_silicon
        # Should return True on ARM macOS
        import platform, sys
        if sys.platform == "darwin" and platform.machine() == "arm64":
            assert is_apple_silicon()

    def test_get_system_memory_gb(self):
        from yunshu_engine.utils.hardware import get_system_memory_gb
        gb = get_system_memory_gb()
        assert gb > 0


class TestOptimizations:
    def test_get_optimization_status(self):
        from yunshu_engine.optimizations import get_optimization_status
        status = get_optimization_status()
        assert "hardware" in status
        assert "chip" in status["hardware"]
        assert "total_memory_gb" in status["hardware"]
        assert "mlx_lm_features" in status


class TestModelRegistry:
    def test_singleton(self):
        from yunshu_engine.model_registry import ModelRegistry
        r1 = ModelRegistry()
        r2 = ModelRegistry()
        assert r1 is r2

    def test_acquire_release(self):
        from yunshu_engine.model_registry import get_registry
        registry = get_registry()

        class FakeModel:
            pass

        class FakeEngine:
            pass

        model = FakeModel()
        engine = FakeEngine()
        assert registry.acquire(model, engine, "engine-1")
        owned, owner = registry.is_owned(model)
        assert owned
        assert owner == "engine-1"

        assert registry.release(model, "engine-1")
        owned, _ = registry.is_owned(model)
        assert not owned

    def test_ownership_error(self):
        from yunshu_engine.model_registry import get_registry, ModelOwnershipError
        registry = get_registry()

        class FakeModel:
            pass

        class FakeEngine:
            pass

        model = FakeModel()
        engine1 = FakeEngine()
        engine2 = FakeEngine()
        registry.acquire(model, engine1, "e1")
        with pytest.raises(ModelOwnershipError):
            registry.acquire(model, engine2, "e2")
        # Force should work
        assert registry.acquire(model, engine2, "e2", force=True)
        registry.release(model, "e2")

    def test_stats(self):
        from yunshu_engine.model_registry import get_registry
        stats = get_registry().get_stats()
        assert "total_entries" in stats
        assert "active_owners" in stats


class TestModelDiscovery:
    def test_detect_llm(self):
        from yunshu_engine.model_discovery import detect_model_type
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "test-llm"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(json.dumps({
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
            }))
            assert detect_model_type(model_dir) == "llm"

    def test_detect_vlm(self):
        from yunshu_engine.model_discovery import detect_model_type
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "test-vlm"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(json.dumps({
                "model_type": "qwen3_omni",
                "architectures": ["Qwen3OmniForConditionalGeneration"],
                "vision_config": {"hidden_size": 1152},
            }))
            assert detect_model_type(model_dir) == "vlm"

    def test_detect_tts(self):
        from yunshu_engine.model_discovery import detect_model_type
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "test-tts"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(json.dumps({
                "model_type": "qwen3_tts",
                "architectures": ["Qwen3TTSForConditionalGeneration"],
            }))
            assert detect_model_type(model_dir) == "audio_tts"

    def test_detect_asr(self):
        from yunshu_engine.model_discovery import detect_model_type
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "test-asr"
            model_dir.mkdir()
            (model_dir / "config.json").write_text(json.dumps({
                "model_type": "qwen3_asr",
                "architectures": ["Qwen3ASRForConditionalGeneration"],
            }))
            assert detect_model_type(model_dir) == "audio_stt"

    def test_detect_from_name(self):
        # Dynamic detection delegates to model_manager._detect_model_type
        # For dirs without config.json, it falls back to directory name heuristics
        from yunshu_engine.model_manager import _detect_model_type, ModelType
        with tempfile.TemporaryDirectory() as tmp:
            # TTS by name heuristic
            tts_dir = Path(tmp) / "Qwen3-TTS-1.7B"
            tts_dir.mkdir()
            assert _detect_model_type(str(tts_dir)) == ModelType.TTS

            # ASR by name heuristic
            asr_dir = Path(tmp) / "Qwen3-ASR-1.7B"
            asr_dir.mkdir()
            assert _detect_model_type(str(asr_dir)) == ModelType.ASR

            # LLM default
            llm_dir = Path(tmp) / "SomeLLM-7B"
            llm_dir.mkdir()
            assert _detect_model_type(str(llm_dir)) == ModelType.LLM

    def test_discover_models(self):
        from yunshu_engine.model_discovery import discover_models
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)

            # Create fake model dirs
            llm = base / "llama-3b"
            llm.mkdir()
            (llm / "config.json").write_text(json.dumps({"model_type": "llama"}))
            (llm / "weights.safetensors").write_bytes(b"\x00" * 1000)

            models = discover_models(base)
            assert "llama-3b" in models
            assert models["llama-3b"].model_type == "llm"
            assert models["llama-3b"].estimated_size > 0


class TestRequest:
    def test_timing_properties(self):
        from yunshu_engine.request import Request, RequestStatus
        import time

        req = Request(request_id="test", prompt="hello")
        req.prefill_start = 100.0
        req.prefill_end = 100.5
        req.generation_start = 100.5
        req.generation_end = 101.5

        assert req.prefill_duration == 0.5
        assert req.generation_duration == 1.0

    def test_priority_comparison(self):
        from yunshu_engine.request import Request
        import time

        r1 = Request(request_id="high", prompt="a", priority=0)
        r2 = Request(request_id="low", prompt="b", priority=10)
        assert r1 < r2

    def test_vlm_fields(self):
        from yunshu_engine.request import Request
        req = Request(
            request_id="vlm-test",
            prompt="describe this",
            vlm_image_hash="abc123",
            cached_tokens=50,
        )
        assert req.vlm_image_hash == "abc123"
        assert req.cached_tokens == 50

    def test_append_token_increments_computed(self):
        from yunshu_engine.request import Request
        req = Request(request_id="test", prompt="hello")
        assert req.num_computed_tokens == 0
        req.append_token(42)
        assert req.num_computed_tokens == 1
        assert req.output_token_ids == [42]


class TestPrefillProgress:
    def test_update_and_get(self):
        from yunshu_engine.prefill_progress import PrefillProgressTracker
        tracker = PrefillProgressTracker()

        tracker.update("req-1", 500, 1000, "test-model")
        progress = tracker.get_model_progress("test-model")
        assert len(progress) == 1
        assert progress[0]["request_id"] == "req-1"
        assert progress[0]["progress_pct"] == 50.0

    def test_auto_remove_on_complete(self):
        from yunshu_engine.prefill_progress import PrefillProgressTracker
        tracker = PrefillProgressTracker()

        tracker.update("req-1", 500, 1000, "test-model")
        assert tracker.active_count == 1

        tracker.update("req-1", 1000, 1000, "test-model")
        assert tracker.active_count == 0

    def test_remove(self):
        from yunshu_engine.prefill_progress import PrefillProgressTracker
        tracker = PrefillProgressTracker()

        tracker.update("req-1", 100, 1000, "test-model")
        tracker.remove("req-1")
        assert tracker.active_count == 0


class TestServerMetrics:
    def test_record_and_snapshot(self):
        from yunshu_engine.server_metrics import ServerMetrics
        m = ServerMetrics()
        m.record_request_complete(
            prompt_tokens=100,
            completion_tokens=50,
            cached_tokens=20,
            prefill_duration=0.1,
            generation_duration=0.5,
            model_id="test-model",
        )

        snap = m.get_snapshot()
        assert snap["total_requests"] == 1
        assert snap["total_prompt_tokens"] == 100
        assert snap["total_completion_tokens"] == 50
        assert snap["avg_generation_tps"] == 100.0

    def test_per_model_snapshot(self):
        from yunshu_engine.server_metrics import ServerMetrics
        m = ServerMetrics()
        m.record_request_complete(100, 50, model_id="model-a")
        m.record_request_complete(200, 100, model_id="model-b")

        snap_a = m.get_snapshot(model_id="model-a")
        assert snap_a["total_requests"] == 1

        snap_b = m.get_snapshot(model_id="model-b")
        assert snap_b["total_requests"] == 1

    def test_clear_session(self):
        from yunshu_engine.server_metrics import ServerMetrics
        m = ServerMetrics()
        m.record_request_complete(100, 50)
        m.clear_session()
        snap = m.get_snapshot()
        assert snap["total_requests"] == 0

    def test_alltime_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            from yunshu_engine.server_metrics import ServerMetrics
            path = Path(tmp) / "stats.json"
            m1 = ServerMetrics(stats_path=path)
            m1.record_request_complete(100, 50)
            m1.save_alltime()

            m2 = ServerMetrics(stats_path=path)
            snap = m2.get_snapshot(scope="alltime")
            assert snap["total_requests"] == 1


class TestOutputCollector:
    def test_put_and_get_nowait(self):
        from yunshu_engine.output_collector import RequestOutputCollector
        from yunshu_engine.request import RequestOutput

        c = RequestOutputCollector()
        output = RequestOutput(request_id="r1", new_text="hello")
        c.put(output)
        result = c.get_nowait()
        assert result is not None
        assert result.new_text == "hello"

    def test_aggregation(self):
        from yunshu_engine.output_collector import RequestOutputCollector
        from yunshu_engine.request import RequestOutput

        c = RequestOutputCollector(aggregate=True)
        c.put(RequestOutput(request_id="r1", new_text="hel", new_token_ids=[1]))
        c.put(RequestOutput(request_id="r1", new_text="lo", new_token_ids=[2]))
        result = c.get_nowait()
        assert result.new_text == "hello"
        assert result.new_token_ids == [1, 2]

    def test_sentinel(self):
        from yunshu_engine.output_collector import RequestOutputCollector
        c = RequestOutputCollector()
        c.put(None)
        assert c.get_nowait() is None

    def test_stream_state(self):
        from yunshu_engine.output_collector import RequestStreamState
        ss = RequestStreamState(stream_interval=5)
        assert ss.should_send(1, finished=False)  # first token
        ss.mark_sent(1)
        assert not ss.should_send(3, finished=False)
        assert ss.should_send(6, finished=False)
        assert ss.should_send(3, finished=True)
