"""Tests for prompt_tokens accuracy across all generation paths.

Verifies that prompt_tokens is set correctly and consistently in:
- BatchedEngine fast path (non-streaming)
- BatchedEngine engine loop path (non-streaming)
- VLMEngine non-streaming
- RequestOutput.usage property
- GenerationOutput dataclass
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from yunshu_engine.request import RequestOutput
from yunshu_engine.batched_engine import GenerationOutput


# ── 1. GenerationOutput prompt_tokens ──


class TestGenerationOutputPromptTokens:
    """Verify GenerationOutput dataclass carries prompt_tokens correctly."""

    def test_default_prompt_tokens_is_zero(self):
        out = GenerationOutput()
        assert out.prompt_tokens == 0

    def test_prompt_tokens_set_correctly(self):
        out = GenerationOutput(prompt_tokens=42)
        assert out.prompt_tokens == 42

    def test_prompt_tokens_with_all_fields(self):
        out = GenerationOutput(
            text="hello",
            new_text="hello",
            prompt_tokens=10,
            completion_tokens=5,
            finished=True,
            finish_reason="stop",
            cached_tokens=3,
            reasoning_tokens=2,
            ttft_ms=12.5,
        )
        assert out.prompt_tokens == 10
        assert out.completion_tokens == 5
        assert out.finished is True
        assert out.finish_reason == "stop"
        assert out.cached_tokens == 3
        assert out.reasoning_tokens == 2
        assert out.ttft_ms == 12.5


# ── 2. RequestOutput usage property ──


class TestRequestOutputUsage:
    """Verify RequestOutput.usage property computes total_tokens correctly."""

    def test_usage_property(self):
        out = RequestOutput(
            request_id="test",
            prompt_tokens=10,
            completion_tokens=5,
        )
        assert out.usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    def test_usage_with_zero_tokens(self):
        out = RequestOutput(request_id="test")
        assert out.usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def test_usage_large_tokens(self):
        out = RequestOutput(
            request_id="test",
            prompt_tokens=100000,
            completion_tokens=50000,
        )
        assert out.usage["total_tokens"] == 150000


# ── 3. BatchedEngine fast path prompt_tokens ──


class TestBatchedEngineFastPathPromptTokens:
    """Verify prompt_tokens accuracy in BatchedEngine._generate_fast path."""

    def _make_batched_engine(self):
        from yunshu_engine.batched_engine import BatchedEngine
        engine = object.__new__(BatchedEngine)
        engine.model_name = "test-model"
        engine.stream_interval = 1
        engine.enable_thinking = None
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()
        engine._engine_core = None
        engine._loaded = True
        engine._starting = False
        engine._spec_decoder = None
        engine._spec_enabled = False
        engine._mtp_decoder = None
        engine._mtp_strategy = None
        engine._ngram_proposer = None
        engine._ngram_stats = {"proposals": 0, "accepted": 0, "total_draft": 0}
        engine._response_cache_hits = 0
        engine._response_cache_misses = 0
        engine._gpu_rejection_sampler = MagicMock()
        engine._gpu_rejection_enabled = False
        engine._spec_draft_verifier = MagicMock()
        engine._adaptive_spec = None
        engine._lookahead_reasoning = MagicMock()
        engine._medusa_proposer = None
        engine._medusa_strategy = None
        engine._spec_prefill_enabled = False
        engine._spec_prefill_threshold = 8192
        engine._spec_prefill_keep_rate = 0.20
        engine._spec_prefill_draft_model = None
        engine._kv_prefix_cache = MagicMock()
        engine._kv_prefix_cache.get.return_value = (None, None, 0)
        engine._kv_prefix_cache.evict_under_pressure = MagicMock()
        engine._warm_prompt_stats = {}
        engine._prompt_cache = MagicMock()
        engine._prompt_cache.lookup.return_value = None
        engine._preprocessor_registry = MagicMock()
        engine._total_reasoning_tokens = 0
        engine._kv_quant_bits = None
        engine._kv_quant_group_size = 64
        engine._mem_pressure_threshold = 0.0
        engine._lora_manager = None
        engine._compiled = False
        engine._use_compile = False
        engine._metal_kernel_manager = None
        engine._metal_kernels_enabled = False
        engine._engine_loop_default = False
        engine._streaming_pipeline_enabled = False
        engine._thinking_store = None
        engine._active_fast_path_count = 0
        engine._fast_path_lock = __import__('threading').Lock()
        engine._settings = None
        engine._deltanet_inverter = None
        engine._deltanet_inversion_enabled = False
        engine._deltanet_inversion_stats = {}
        engine._response_cache = MagicMock()
        engine._kv_transfer_client = None
        engine._inflight_prefix_tracker = None
        engine._gpu_rejection_enabled = False
        engine._deltanet_inverter = None
        engine._deltanet_inversion_enabled = False
        engine._deltanet_inversion_stats = {}
        engine._lookahead_reasoning = MagicMock()
        engine._medusa_proposer = None
        engine._medusa_strategy = None
        engine._gpu_rejection_sampler = MagicMock()
        engine._thinking_store = None
        engine._max_request_timeout = None
        return engine

    @pytest.mark.asyncio
    async def test_fast_path_prompt_tokens_from_tokenizer(self):
        """Fast path prompt_tokens should match tokenizer.encode().length."""
        engine = self._make_batched_engine()
        engine._tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        engine._tokenizer.eos_token_id = 2
        engine._tokenizer.eos_token_ids = [2]
        engine._tokenizer.decode.return_value = "test"

        mock_detokenizer = MagicMock()
        mock_detokenizer.last_segment = "test"
        mock_detokenizer.finalize.return_value = ""
        engine._tokenizer.detokenizer = mock_detokenizer

        with patch('mlx_lm.generate.generate_step') as mock_step, \
             patch('mlx_lm.sample_utils.make_sampler') as mock_sampler, \
             patch('yunshu_engine.batched_engine._create_prompt_cache_with_quant', return_value=[]), \
             patch('mlx.core') as mock_mx:

            mock_step.return_value = iter([(100, None), (2, None)])
            mock_mx.array.return_value = MagicMock()
            mock_mx.eval.return_value = None
            mock_mx.stream.return_value = MagicMock()
            mock_mx.new_thread_local_stream.return_value = MagicMock()
            mock_mx.random.seed.return_value = None

            engine._model.max_seq_len = 4096

            with patch('yunshu_engine.batched_engine._clean_special_tokens', side_effect=lambda x: x):
                with patch.object(engine, '_apply_chat_template', return_value="test"):
                    result = await engine._generate_fast(
                        prompt="hello world test",
                        max_tokens=10,
                    )

        assert result.prompt_tokens == 5


# ── 4. Engine loop prompt_tokens mapping ──


class TestEngineLoopPromptTokensMapping:
    """Verify prompt_tokens is mapped correctly from engine_core results."""

    def test_engine_core_result_mapped_to_generation_output(self):
        mock_result = MagicMock()
        mock_result.prompt_tokens = 42
        mock_result.completion_tokens = 10
        mock_result.output_text = "hello world"
        mock_result.finish_reason = "stop"
        mock_result.reasoning_tokens = 0
        mock_result.ttft_ms = 15.0
        mock_result.error = None

        output = GenerationOutput(
            text=mock_result.output_text,
            new_text=mock_result.output_text,
            prompt_tokens=mock_result.prompt_tokens,
            completion_tokens=mock_result.completion_tokens,
            finished=True,
            finish_reason=mock_result.finish_reason,
            reasoning_tokens=0,
            cached_tokens=0,
            ttft_ms=15.0,
        )

        assert output.prompt_tokens == 42
        assert output.completion_tokens == 10

    def test_engine_core_none_returns_error_output(self):
        output = GenerationOutput(
            finished=True,
            finish_reason="error",
            error="engine_core returned None",
        )
        assert output.prompt_tokens == 0
        assert output.finish_reason == "error"


# ── 5. VLMEngine prompt_tokens ──


class TestVLMEnginePromptTokens:
    """Verify prompt_tokens accuracy in VLMEngine."""

    @pytest.mark.asyncio
    async def test_vlm_prompt_tokens_from_tokenizer(self):
        """VLM generate() returns prompt_tokens matching tokenizer.encode() length."""
        import concurrent.futures
        from yunshu_engine.vlm_engine import VLMEngine
        engine = object.__new__(VLMEngine)
        engine._model_path = "/models/test-model"
        engine._model = MagicMock()
        engine._tokenizer = MagicMock()
        engine._processor = MagicMock()
        engine._config = {}
        engine._running = True
        engine._active_count = 0
        engine._num_requests_processed = 0
        engine._total_reasoning_tokens = 0
        engine._start_time = 0.0
        engine._has_vision = False
        engine._is_vlm = False
        engine._temp_files = None
        import threading
        engine._temp_files_lock = threading.Lock()
        engine._mrope_info = MagicMock(enabled=False)
        engine._rope_delta_manager = None
        engine._vision_cache = None
        engine._vlm_vision_cache_adapter = None
        engine._encoder_cache = MagicMock()
        engine._text_prompt_cache = MagicMock()
        engine._kv_prefix_states = {}
        engine._multimodal_prefix_cache = {}
        engine._vision_encoder_factory = None
        engine._spec_prefill_enabled = False
        engine._pipeline = MagicMock()
        engine._async_core = None
        engine._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        engine._tokenizer.encode.return_value = [10, 20, 30, 40, 50, 60, 70]
        engine._tokenizer.decode.return_value = "hello"
        engine._tokenizer.eos_token_id = 2
        engine._tokenizer.eos_token_ids = [2]

        engine._extract_images = AsyncMock(return_value=[])
        engine._extract_audio = AsyncMock(return_value=[])
        engine._extract_video_frames = AsyncMock(return_value=[])
        engine._tokenize_with_cache = MagicMock(return_value=MagicMock(ids=[10, 20, 30, 40, 50, 60, 70]))
        engine._get_eos_ids = MagicMock(return_value=[2])

        with patch.object(engine, '_format_prompt', return_value="test prompt"):
            with patch('mlx_lm.generate.generate_step') as mock_step, \
                 patch('mlx_lm.sample_utils.make_sampler') as mock_sampler:
                mock_step.return_value = iter([(100, None), (2, None)])

                result = await engine.generate(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=10,
                )

        assert result["prompt_tokens"] == 7


# ── 6. Streaming prompt_tokens consistency ──


class TestStreamingPromptTokensConsistency:
    """Verify streaming and non-streaming paths produce the same prompt_tokens."""

    def test_request_output_prompt_tokens_in_stream(self):
        out = RequestOutput(
            request_id="test-req",
            new_text="hello",
            prompt_tokens=42,
            completion_tokens=1,
            finish_reason=None,
            finished=False,
        )
        assert out.prompt_tokens == 42
        assert out.finished is False

    def test_request_output_final_stream_has_same_prompt_tokens(self):
        stream_outputs = [
            RequestOutput(request_id="req-1", prompt_tokens=10, completion_tokens=1, new_text="he"),
            RequestOutput(request_id="req-1", prompt_tokens=10, completion_tokens=2, new_text="llo"),
            RequestOutput(request_id="req-1", prompt_tokens=10, completion_tokens=3, new_text="", finish_reason="stop", finished=True),
        ]

        prompt_tokens_set = {o.prompt_tokens for o in stream_outputs}
        assert len(prompt_tokens_set) == 1, f"Expected consistent prompt_tokens, got {prompt_tokens_set}"
        assert 10 in prompt_tokens_set

    def test_generation_output_prompt_tokens_matches_request_output(self):
        req_out = RequestOutput(
            request_id="req-1",
            prompt_tokens=25,
            completion_tokens=10,
            finish_reason="stop",
            finished=True,
        )

        gen_out = GenerationOutput(
            text="final text",
            prompt_tokens=req_out.prompt_tokens,
            completion_tokens=req_out.completion_tokens,
            finish_reason=req_out.finish_reason,
            finished=req_out.finished,
        )

        assert gen_out.prompt_tokens == 25
        assert gen_out.prompt_tokens == req_out.prompt_tokens


# ── 7. RequestStatus finish_reason mapping ──


class TestRequestStatusFinishReason:
    """Verify RequestStatus maps to correct finish_reason strings."""

    def test_finished_stopped(self):
        from yunshu_engine.request import RequestStatus
        assert RequestStatus.finish_reason(RequestStatus.FINISHED_STOPPED) == "stop"

    def test_finished_length(self):
        from yunshu_engine.request import RequestStatus
        assert RequestStatus.finish_reason(RequestStatus.FINISHED_LENGTH) == "length"

    def test_finished_aborted(self):
        from yunshu_engine.request import RequestStatus
        assert RequestStatus.finish_reason(RequestStatus.FINISHED_ABORTED) == "abort"

    def test_finished_error(self):
        from yunshu_engine.request import RequestStatus
        assert RequestStatus.finish_reason(RequestStatus.FINISHED_ERROR) == "error"

    def test_finished_timeout(self):
        from yunshu_engine.request import RequestStatus
        assert RequestStatus.finish_reason(RequestStatus.FINISHED_TIMEOUT) == "timeout"

    def test_running_returns_none(self):
        from yunshu_engine.request import RequestStatus
        assert RequestStatus.finish_reason(RequestStatus.RUNNING) is None

    def test_waiting_returns_none(self):
        from yunshu_engine.request import RequestStatus
        assert RequestStatus.finish_reason(RequestStatus.WAITING) is None

    def test_is_finished(self):
        from yunshu_engine.request import RequestStatus
        assert not RequestStatus.is_finished(RequestStatus.WAITING)
        assert not RequestStatus.is_finished(RequestStatus.RUNNING)
        assert not RequestStatus.is_finished(RequestStatus.PREFILLING)
        assert RequestStatus.is_finished(RequestStatus.FINISHED_STOPPED)
        assert RequestStatus.is_finished(RequestStatus.FINISHED_LENGTH)
        assert RequestStatus.is_finished(RequestStatus.FINISHED_ERROR)
        assert RequestStatus.is_finished(RequestStatus.FINISHED_ABORTED)
        assert RequestStatus.is_finished(RequestStatus.FINISHED_TIMEOUT)


# ── 8. _normalize_finish_reason ──


class TestNormalizeFinishReason:
    """Verify _normalize_finish_reason maps all internal reasons correctly."""

    def test_valid_openai_reasons_passthrough(self):
        from yunshu_gateway.routers.chat import _normalize_finish_reason
        assert _normalize_finish_reason("stop") == "stop"
        assert _normalize_finish_reason("length") == "length"
        assert _normalize_finish_reason("tool_calls") == "tool_calls"
        assert _normalize_finish_reason("content_filter") == "content_filter"

    def test_none_returns_stop(self):
        from yunshu_gateway.routers.chat import _normalize_finish_reason
        assert _normalize_finish_reason(None) == "stop"
        assert _normalize_finish_reason("") == "stop"

    def test_internal_reasons_mapped(self):
        from yunshu_gateway.routers.chat import _normalize_finish_reason
        assert _normalize_finish_reason("abort") == "stop"
        assert _normalize_finish_reason("cancel") == "stop"
        assert _normalize_finish_reason("error") == "length"
        assert _normalize_finish_reason("timeout") == "length"
        assert _normalize_finish_reason("memory_limit") == "length"
        assert _normalize_finish_reason("memory_exceeded") == "length"

    def test_unknown_reason_defaults_to_stop(self):
        from yunshu_gateway.routers.chat import _normalize_finish_reason
        assert _normalize_finish_reason("unknown_reason") == "stop"
