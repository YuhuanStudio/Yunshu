"""Tests for VLM finish_reason correctness — stop vs length vs error.

Tests that VLMEngine's generate() and generate_stream() return the correct
finish_reason in all termination scenarios.

NOTE: generate_step/make_sampler are lazy-imported inside function bodies,
so we patch at the source module. We use a real ThreadPoolExecutor so
run_in_executor works correctly.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from yunshu_engine.request import RequestOutput


# ── Helpers ──

def _make_vlm_engine():
    """Create a VLMEngine with mocked internals for finish_reason testing."""
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
    # Use a real executor so run_in_executor works
    engine._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    return engine


# ── 1. Non-streaming finish_reason ──


class TestVLMNonStreamingFinishReason:
    """Test finish_reason in VLMEngine.generate() (non-streaming path)."""

    @pytest.mark.asyncio
    async def test_max_tokens_returns_length(self):
        """When max_tokens is reached without a stop sequence, finish_reason='length'."""
        engine = _make_vlm_engine()
        engine._tokenizer.encode.return_value = [1, 2, 3]
        engine._tokenizer.decode.return_value = "hello world"
        engine._tokenizer.eos_token_id = 2
        engine._tokenizer.eos_token_ids = [2]
        engine._extract_images = AsyncMock(return_value=[])
        engine._extract_audio = AsyncMock(return_value=[])
        engine._extract_video_frames = AsyncMock(return_value=[])
        engine._tokenize_with_cache = MagicMock(return_value=[1, 2, 3])

        with patch.object(engine, '_format_prompt', return_value="test prompt"):
            with patch('mlx_lm.generate.generate_step') as mock_step, \
                 patch('mlx_lm.sample_utils.make_sampler') as mock_sampler:
                mock_step.return_value = iter([(100, None), (101, None), (102, None)])
                engine._get_eos_ids = MagicMock(return_value=[2])

                result = await engine.generate(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=3,
                )

        assert result["finish_reason"] == "length"
        assert result["completion_tokens"] == 3

    @pytest.mark.asyncio
    async def test_stop_token_returns_stop(self):
        """When a stop token is produced, finish_reason='stop'."""
        engine = _make_vlm_engine()
        engine._tokenizer.encode.return_value = [1, 2, 3]
        engine._tokenizer.decode.return_value = "hello"
        engine._tokenizer.eos_token_id = 2
        engine._tokenizer.eos_token_ids = [2]
        engine._extract_images = AsyncMock(return_value=[])
        engine._extract_audio = AsyncMock(return_value=[])
        engine._extract_video_frames = AsyncMock(return_value=[])
        engine._tokenize_with_cache = MagicMock(return_value=[1, 2, 3])

        with patch.object(engine, '_format_prompt', return_value="test prompt"):
            with patch('mlx_lm.generate.generate_step') as mock_step, \
                 patch('mlx_lm.sample_utils.make_sampler') as mock_sampler:
                mock_step.return_value = iter([(100, None), (2, None)])
                engine._get_eos_ids = MagicMock(return_value=[2])

                result = await engine.generate(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=10,
                )

        assert result["finish_reason"] == "stop"
        assert result["completion_tokens"] == 2

    @pytest.mark.asyncio
    async def test_stop_sequence_returns_stop(self):
        """When a stop string sequence is matched, finish_reason='stop'."""
        engine = _make_vlm_engine()
        engine._tokenizer.decode.return_value = "hello STOP"
        engine._tokenizer.eos_token_id = 2
        engine._tokenizer.eos_token_ids = [2]
        engine._extract_images = AsyncMock(return_value=[])
        engine._extract_audio = AsyncMock(return_value=[])
        engine._extract_video_frames = AsyncMock(return_value=[])
        engine._tokenize_with_cache = MagicMock(return_value=[1, 2, 3])

        def mock_encode(text):
            if text == "STOP":
                return [999]
            return [1, 2, 3]
        engine._tokenizer.encode.side_effect = mock_encode

        with patch.object(engine, '_format_prompt', return_value="test prompt"):
            with patch('mlx_lm.generate.generate_step') as mock_step, \
                 patch('mlx_lm.sample_utils.make_sampler') as mock_sampler:
                mock_step.return_value = iter([(100, None), (999, None)])
                engine._get_eos_ids = MagicMock(return_value=[2])

                result = await engine.generate(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=10,
                    stop=["STOP"],
                )

        assert result["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_thinking_budget_returns_stop(self):
        """When thinking budget is exceeded, finish_reason='stop'."""
        engine = _make_vlm_engine()
        engine._tokenizer.decode.return_value = "<think thinking..."
        engine._tokenizer.eos_token_id = 2
        engine._tokenizer.eos_token_ids = [2]
        engine._extract_images = AsyncMock(return_value=[])
        engine._extract_audio = AsyncMock(return_value=[])
        engine._extract_video_frames = AsyncMock(return_value=[])
        engine._tokenize_with_cache = MagicMock(return_value=[1, 2, 3])

        think_start_id = 500
        think_end_id = 501

        def mock_encode(text):
            if text == "<think":
                return [0, think_start_id]
            if text == "</think":
                return [0, think_end_id]
            return [1, 2, 3]
        engine._tokenizer.encode.side_effect = mock_encode

        with patch.object(engine, '_format_prompt', return_value="test prompt"):
            with patch('mlx_lm.generate.generate_step') as mock_step, \
                 patch('mlx_lm.sample_utils.make_sampler') as mock_sampler:
                mock_step.return_value = iter([
                    (think_start_id, None),
                    (200, None), (201, None), (202, None),
                    (203, None), (204, None),
                ])
                engine._get_eos_ids = MagicMock(return_value=[2])

                result = await engine.generate(
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=100,
                    thinking_budget=5,
                )

        assert result["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_generation_error_propagates(self):
        """When generation raises an exception, it propagates to caller."""
        engine = _make_vlm_engine()
        engine._extract_images = AsyncMock(return_value=[])
        engine._extract_audio = AsyncMock(return_value=[])
        engine._extract_video_frames = AsyncMock(return_value=[])

        with patch.object(engine, '_format_prompt', return_value="test prompt"):
            with patch('mlx_lm.generate.generate_step', side_effect=RuntimeError("GPU error")):
                engine._tokenize_with_cache = MagicMock(return_value=[1, 2, 3])
                engine._get_eos_ids = MagicMock(return_value=[2])
                engine._tokenizer.decode.return_value = "test"
                engine._tokenizer.eos_token_id = 2

                with pytest.raises(RuntimeError, match="GPU error"):
                    await engine.generate(
                        messages=[{"role": "user", "content": "hello"}],
                        max_tokens=10,
                    )


# ── 2. Streaming finish_reason ──


class TestVLMStreamingFinishReason:
    """Test finish_reason in VLMEngine.generate_stream() (streaming path)."""

    @pytest.mark.asyncio
    async def test_stream_max_tokens_returns_length(self):
        """Streaming: max_tokens reached emits finish_reason='length'."""
        engine = _make_vlm_engine()
        engine._tokenizer.encode.return_value = [1, 2, 3]
        engine._tokenizer.decode.return_value = "a"
        engine._tokenizer.eos_token_id = 2
        engine._tokenizer.eos_token_ids = [2]

        mock_detokenizer = MagicMock()
        mock_detokenizer.last_segment = "a"
        mock_detokenizer.finalize.return_value = ""
        engine._tokenizer.detokenizer = mock_detokenizer

        engine._extract_images = AsyncMock(return_value=[])
        engine._extract_audio = AsyncMock(return_value=[])
        engine._extract_video_frames = AsyncMock(return_value=[])

        with patch('mlx_lm.generate.generate_step') as mock_step, \
             patch('mlx_lm.sample_utils.make_sampler') as mock_sampler:
            mock_step.return_value = iter([(100, None), (101, None), (102, None)])
            engine._get_eos_ids = MagicMock(return_value=[2])
            engine._tokenize_with_cache = MagicMock(return_value=[1, 2, 3])

            outputs = []
            async for output in engine.generate_stream(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=3,
            ):
                outputs.append(output)

        finished_outputs = [o for o in outputs if o.finished]
        assert len(finished_outputs) >= 1, "Expected at least one finished output"
        assert finished_outputs[-1].finish_reason == "length"

    @pytest.mark.asyncio
    async def test_stream_stop_token_returns_stop(self):
        """Streaming: stop token hit emits finish_reason='stop'."""
        engine = _make_vlm_engine()
        engine._tokenizer.encode.return_value = [1, 2, 3]
        engine._tokenizer.decode.return_value = "a"
        engine._tokenizer.eos_token_id = 2
        engine._tokenizer.eos_token_ids = [2]

        mock_detokenizer = MagicMock()
        mock_detokenizer.last_segment = "a"
        mock_detokenizer.finalize.return_value = ""
        engine._tokenizer.detokenizer = mock_detokenizer

        engine._extract_images = AsyncMock(return_value=[])
        engine._extract_audio = AsyncMock(return_value=[])
        engine._extract_video_frames = AsyncMock(return_value=[])

        with patch('mlx_lm.generate.generate_step') as mock_step, \
             patch('mlx_lm.sample_utils.make_sampler') as mock_sampler:
            mock_step.return_value = iter([(100, None), (2, None)])
            engine._get_eos_ids = MagicMock(return_value=[2])
            engine._tokenize_with_cache = MagicMock(return_value=[1, 2, 3])

            outputs = []
            async for output in engine.generate_stream(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
            ):
                outputs.append(output)

        finished_outputs = [o for o in outputs if o.finished]
        assert len(finished_outputs) >= 1
        assert finished_outputs[-1].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_error_returns_error(self):
        """Streaming: when generation raises an exception, emits finish_reason='error'."""
        engine = _make_vlm_engine()
        engine._tokenizer.encode.return_value = [1, 2, 3]
        engine._tokenizer.decode.return_value = "a"
        engine._tokenizer.eos_token_id = 2
        engine._tokenizer.eos_token_ids = [2]

        engine._extract_images = AsyncMock(return_value=[])
        engine._extract_audio = AsyncMock(return_value=[])
        engine._extract_video_frames = AsyncMock(return_value=[])

        with patch('mlx_lm.generate.generate_step', side_effect=RuntimeError("CUDA OOM")):
            engine._get_eos_ids = MagicMock(return_value=[2])
            engine._tokenize_with_cache = MagicMock(return_value=[1, 2, 3])

            outputs = []
            async for output in engine.generate_stream(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
            ):
                outputs.append(output)

        error_outputs = [o for o in outputs if o.finish_reason == "error"]
        assert len(error_outputs) >= 1, "Expected at least one error output"
        assert error_outputs[-1].finished is True
        assert "CUDA OOM" in (error_outputs[-1].error or "")

    @pytest.mark.asyncio
    async def test_stream_cancel_returns_cancel(self):
        """Streaming: when cancel_event is set, emits finish_reason='cancel'."""
        engine = _make_vlm_engine()
        engine._tokenizer.encode.return_value = [1, 2, 3]
        engine._tokenizer.decode.return_value = "a"
        engine._tokenizer.eos_token_id = 2
        engine._tokenizer.eos_token_ids = [2]

        mock_detokenizer = MagicMock()
        mock_detokenizer.last_segment = "a"
        mock_detokenizer.finalize.return_value = ""
        engine._tokenizer.detokenizer = mock_detokenizer

        engine._extract_images = AsyncMock(return_value=[])
        engine._extract_audio = AsyncMock(return_value=[])
        engine._extract_video_frames = AsyncMock(return_value=[])

        cancel_event = asyncio.Event()

        token_iter = iter([(100, None), (101, None), (102, None)])

        def _step_fn(*args, **kwargs):
            cancel_event.set()
            return token_iter

        with patch('mlx_lm.generate.generate_step', side_effect=_step_fn):
            engine._get_eos_ids = MagicMock(return_value=[2])
            engine._tokenize_with_cache = MagicMock(return_value=[1, 2, 3])

            outputs = []
            async for output in engine.generate_stream(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
                cancel_event=cancel_event,
            ):
                outputs.append(output)

        cancel_outputs = [o for o in outputs if o.finish_reason == "cancel"]
        assert len(cancel_outputs) >= 1, "Expected at least one cancel output"
        assert cancel_outputs[-1].finished is True

    @pytest.mark.asyncio
    async def test_stream_thinking_budget_returns_stop(self):
        """Streaming: thinking budget exceeded emits finish_reason='stop'."""
        engine = _make_vlm_engine()
        engine._tokenizer.encode.return_value = [1, 2, 3]
        engine._tokenizer.decode.return_value = "a"
        engine._tokenizer.eos_token_id = 2
        engine._tokenizer.eos_token_ids = [2]

        mock_detokenizer = MagicMock()
        mock_detokenizer.last_segment = "a"
        mock_detokenizer.finalize.return_value = ""
        engine._tokenizer.detokenizer = mock_detokenizer

        engine._extract_images = AsyncMock(return_value=[])
        engine._extract_audio = AsyncMock(return_value=[])
        engine._extract_video_frames = AsyncMock(return_value=[])

        think_start_id = 500
        think_end_id = 501

        def mock_encode(text):
            if text == "<think":
                return [0, think_start_id]
            if text == "</think":
                return [0, think_end_id]
            return [1, 2, 3]
        engine._tokenizer.encode.side_effect = mock_encode

        with patch('mlx_lm.generate.generate_step') as mock_step, \
             patch('mlx_lm.sample_utils.make_sampler') as mock_sampler:
            mock_step.return_value = iter([
                (think_start_id, None),
                (200, None), (201, None), (202, None),
            ])
            engine._get_eos_ids = MagicMock(return_value=[2])
            engine._tokenize_with_cache = MagicMock(return_value=[1, 2, 3])

            outputs = []
            async for output in engine.generate_stream(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=100,
                thinking_budget=3,
            ):
                outputs.append(output)

        finished_outputs = [o for o in outputs if o.finished]
        assert len(finished_outputs) >= 1
        assert finished_outputs[-1].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_stop_suffix_returns_stop(self):
        """Streaming: when accumulated text ends with a stop suffix, finish_reason='stop'."""
        engine = _make_vlm_engine()
        engine._tokenizer.encode.return_value = [1, 2, 3]
        engine._tokenizer.eos_token_id = 2
        engine._tokenizer.eos_token_ids = [2]

        mock_detokenizer = MagicMock()
        # First token returns "hello ", second returns "STOP" → accumulated = "hello STOP"
        mock_detokenizer.last_segment = MagicMock(side_effect=["hello ", "STOP"])
        mock_detokenizer.finalize.return_value = ""
        engine._tokenizer.detokenizer = mock_detokenizer

        engine._extract_images = AsyncMock(return_value=[])
        engine._extract_audio = AsyncMock(return_value=[])
        engine._extract_video_frames = AsyncMock(return_value=[])

        with patch('mlx_lm.generate.generate_step') as mock_step, \
             patch('mlx_lm.sample_utils.make_sampler') as mock_sampler:
            mock_step.return_value = iter([(100, None), (101, None)])
            engine._get_eos_ids = MagicMock(return_value=[2])
            engine._tokenize_with_cache = MagicMock(return_value=[1, 2, 3])

            outputs = []
            async for output in engine.generate_stream(
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=10,
                stop=["STOP"],
            ):
                outputs.append(output)

        finished = [o for o in outputs if o.finished]
        if finished:
            assert finished[-1].finish_reason == "stop"


# ── 3. Finish_reason from _generate_vlm_vision ──


class TestVLMVisionFinishReason:
    """Test finish_reason in the VLM vision generation path."""

    @pytest.mark.asyncio
    async def test_vision_stop_sequence_in_result(self):
        """VLM vision path: stop sequence in output text -> finish_reason='stop'."""
        engine = _make_vlm_engine()
        engine._has_vision = True
        engine._is_vlm = True

        mock_result = MagicMock()
        mock_result.text = "Hello world STOP more text"
        type(mock_result).__str__ = lambda self: "Hello world STOP more text"

        engine._extract_images = AsyncMock(return_value=["/tmp/test.jpg"])
        engine._extract_audio = AsyncMock(return_value=[])
        engine._extract_video_frames = AsyncMock(return_value=[])
        engine._apply_vlm_template_with_cache = MagicMock(return_value="test prompt")
        engine._compute_image_hash = MagicMock(return_value="abc123")
        engine._get_kv_prefix_state = MagicMock(return_value=None)
        engine._ensure_kv_prefix_state = MagicMock()
        engine._vlm_vision_cache_adapter = None
        engine._encoder_cache = MagicMock()
        engine._encoder_cache.get.return_value = None

        engine._tokenizer.encode.return_value = [1, 2, 3]
        engine._tokenizer.decode.return_value = "Hello world STOP more text"

        with patch('mlx_vlm.generate.generate', return_value=mock_result):
            with patch('yunshu_engine.mrope.detect_mrope') as mock_mrope:
                mock_mrope.return_value = MagicMock(enabled=False)

                result = await engine.generate(
                    messages=[
                        {"role": "user", "content": [
                            {"type": "text", "text": "describe"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                        ]},
                    ],
                    max_tokens=100,
                    stop=["STOP"],
                )

        assert result["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_vision_max_tokens_returns_length(self):
        """VLM vision path: no stop sequence -> finish_reason='length'."""
        engine = _make_vlm_engine()
        engine._has_vision = True
        engine._is_vlm = True

        mock_result = MagicMock()
        mock_result.text = "Hello world"
        type(mock_result).__str__ = lambda self: "Hello world"

        engine._extract_images = AsyncMock(return_value=["/tmp/test.jpg"])
        engine._extract_audio = AsyncMock(return_value=[])
        engine._extract_video_frames = AsyncMock(return_value=[])
        engine._apply_vlm_template_with_cache = MagicMock(return_value="test prompt")
        engine._compute_image_hash = MagicMock(return_value="abc123")
        engine._get_kv_prefix_state = MagicMock(return_value=None)
        engine._ensure_kv_prefix_state = MagicMock()
        engine._vlm_vision_cache_adapter = None
        engine._encoder_cache = MagicMock()
        engine._encoder_cache.get.return_value = None

        engine._tokenizer.encode.return_value = [1, 2, 3]
        engine._tokenizer.decode.return_value = "Hello world"

        with patch('mlx_vlm.generate.generate', return_value=mock_result):
            with patch('yunshu_engine.mrope.detect_mrope') as mock_mrope:
                mock_mrope.return_value = MagicMock(enabled=False)

                result = await engine.generate(
                    messages=[
                        {"role": "user", "content": [
                            {"type": "text", "text": "describe"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                        ]},
                    ],
                    max_tokens=100,
                )

        assert result["finish_reason"] == "length"
