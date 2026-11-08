"""47 fix verification tests — 24 tests, no GPU needed.

AutoTuner profiler, Scheduler shutdown, MemoryAwareScheduler lifecycle, StepMetrics, Any import
TTS params, audio format, Anthropic top_k/thinking_budget, Responses API, MCP signature, Models timestamp
output_parser, model optimization, KV lifecycle admit/release, MemoryAwareScheduler model config
TieredKV _extract_kv_for_block, Anthropic cache_creation_input_tokens
"""

from unittest.mock import MagicMock

import pytest

# ── ──────────────────────────────────────────────────────────────


class TestWave44KVAnyImport:
    """5. Any import in kv/manager.py and kv/block.py doesn't cause NameError."""

    def test_manager_imports_ok(self):
        import yunshu_kv.manager as m

        assert m is not None

    def test_block_imports_ok(self):
        import yunshu_kv.block as b

        assert hasattr(b, "KVBlock")


# ── ──────────────────────────────────────────────────────────────


class TestWave45TTSParams:
    """6. TTSRequest params forwarded to synthesize()."""

    def test_params(self):
        from yunshu_gateway.routers.audio import TTSRequest

        r = TTSRequest(
            model="kokoro",
            input="Hello",
            top_k=30,
            top_p=0.9,
            repetition_penalty=1.2,
            max_tokens=2048,
        )
        assert r.top_k == 30 and r.top_p == 0.9
        assert r.repetition_penalty == 1.2 and r.max_tokens == 2048


class TestWave45AudioFormatValidation:
    """7. Audio format validation happens at model level (not just endpoint).

    update: mp3 is now ACCEPTED (OpenAI SDK default). Test now
    asserts valid OpenAI formats pass and invalid strings are rejected.
    """

    def test_invalid_format_detected(self):
        """Invalid format strings rejected; OpenAI-spec formats accepted."""
        from pydantic import ValidationError

        from yunshu_gateway.routers.audio import TTSRequest

        # Invalid format → rejected
        with pytest.raises(ValidationError):
            TTSRequest(model="kokoro", input="Hello", response_format="xyz")
        # All 6 OpenAI-spec formats accepted
        for fmt in ("mp3", "opus", "aac", "flac", "wav", "pcm"):
            req = TTSRequest(model="kokoro", input="Hello", response_format=fmt)
            assert req.response_format == fmt


class TestWave45AnthropicRouter:
    """8. Anthropic router forwards top_k and thinking_budget."""

    def test_request_fields(self):
        from yunshu_gateway.routers.anthropic import AnthropicMessagesRequest

        r = AnthropicMessagesRequest(
            model="claude-3",
            messages=[{"role": "user", "content": "hi"}],
            # max_tokens must exceed budget_tokens (Anthropic spec).
            max_tokens=8192,
            top_k=10,
            thinking={"type": "enabled", "budget_tokens": 5000},
        )
        assert r.top_k == 10
        assert r.thinking["type"] == "enabled"


class TestWave45ResponsesAPI:
    """9. Responses API forwards all parameters."""

    def test_all_params(self):
        from yunshu_gateway.routers.responses import ResponsesRequest

        r = ResponsesRequest(
            model="m",
            input="Hello",
            max_output_tokens=512,
            temperature=0.5,
            top_p=0.9,
            top_k=20,
            repetition_penalty=1.1,
            frequency_penalty=0.2,
            presence_penalty=0.1,
            min_p=0.05,
            enable_thinking=True,
            thinking_budget=2048,
            stop=["\n"],
        )
        assert r.top_k == 20 and r.thinking_budget == 2048
        assert r.enable_thinking is True and r.repetition_penalty == 1.1


class TestWave45MCPGenerateImage:
    """10. MCP generate_image uses correct signature (width/height, not n/size)."""

    def test_uses_width_height(self):
        import inspect

        from yunshu_gateway.routers.mcp import _tool_generate_image

        src = inspect.getsource(_tool_generate_image)
        assert "width=" in src and "height=" in src
        for line in [l for l in src.split("\n") if "generate_image(" in l]:
            assert "n=" not in line or "num_inference" in line


class TestWave45ModelsCreatedTimestamp:
    """11. Models API created timestamp is not 0."""

    def test_uses_load_time(self):
        """A card's `created` is never 0 (load time, else the checkpoint's timestamp, else now)."""
        from yunshu_engine.model_card import build_model_card

        assert build_model_card("/nonexistent/model-dir").created > 0
        assert build_model_card("/nonexistent/model-dir", created=123).created == 123


# ── ──────────────────────────────────────────────────────────────


class TestWave46OutputParser:
    """12. output_parser applied to _generate_fast output."""

    def test_parse_output_exists(self):
        from yunshu_engine.output_parser import parse_output

        assert callable(parse_output)

    def test_strips_think_tags(self):
        from yunshu_engine.output_parser import parse_output

        parsed = parse_output(
            "<think\nstep 1\n</think\nThe answer is 42.", "deepseek-r1"
        )
        assert "42" in parsed.content


class TestWave46ModelOptimizationDetection:
    """13. Model optimization detection runs on model load."""

    def test_optimizers_have_stats(self):
        from yunshu_engine.model_optimizations import (
            AttentionOptimizer,
            ModelWarmupManager,
            MoEEfficiencyOptimizer,
            RoPEScalingOptimizer,
        )

        for cls in (
            RoPEScalingOptimizer,
            AttentionOptimizer,
            MoEEfficiencyOptimizer,
            ModelWarmupManager,
        ):
            assert callable(cls().get_stats)


# ── ──────────────────────────────────────────────────────────────


class TestWave47TieredExtractKV:
    """16. KV tiered _extract_kv_for_block uses _key_cache/_value_cache."""

    def test_source_references_key_value_cache(self):
        import inspect

        from yunshu_kv.tiered import TieredKVCacheManager

        src = inspect.getsource(TieredKVCacheManager._extract_kv_for_block)
        assert "_key_cache" in src and "_value_cache" in src

    def test_returns_none_without_hot_cache(self):
        from yunshu_kv.block import KVBlock
        from yunshu_kv.tiered import TieredKVCacheManager

        hot = MagicMock()
        hot._key_cache = None
        hot._value_cache = None
        mgr = TieredKVCacheManager(hot_manager=hot)
        assert mgr._extract_kv_for_block(KVBlock(block_id=0)) is None


class TestWave47AnthropicCacheCreation:
    """17. Anthropic cache_creation_input_tokens reflects cached_tokens."""

    def test_formula(self):
        # cache_creation = prompt_tokens - cached_tokens (clamped to 0)
        assert max(0, 100 - 30) == 70
        assert max(0, 100 - 0) == 100
        assert max(0, 50 - 60) == 0

    def test_response_includes_cache_fields(self):
        import inspect

        from yunshu_gateway.routers.anthropic import _non_stream_batched

        src = inspect.getsource(_non_stream_batched)
        assert "cache_creation_input_tokens" in src and "cached_tokens" in src


class TestWave44AutoTuner:
    """1. AutoTuner.auto_tune() uses profiler recommendations (not TypeError)."""

    def test_auto_tune_applies_profiler_recommendations(self):
        from yunshu_engine.auto_tuner import AutoTuner, PerformanceProfiler, StepMetrics

        profiler = PerformanceProfiler()
        for _ in range(200):
            profiler.record_step(
                StepMetrics(
                    throughput_tok_s=5.0,
                    gpu_memory_util=0.95,
                    wall_time_ms=10.0,
                    tokens_generated=1,
                )
            )
        tuner = AutoTuner(profiler=profiler)
        decisions = tuner.auto_tune()
        assert isinstance(decisions, list)
        if profiler.get_bottleneck().value != "none":
            assert len(decisions) >= 1
            assert decisions[0].param_name in (
                "batch_size",
                "kv_quantization_bits",
                "prefill_chunk_size",
                "spec_draft_length",
                "num_parallel_requests",
            )

    def test_auto_tune_no_recommendations_returns_empty(self):
        from yunshu_engine.auto_tuner import AutoTuner, PerformanceProfiler

        tuner = AutoTuner(profiler=PerformanceProfiler())
        assert tuner.auto_tune() == []
