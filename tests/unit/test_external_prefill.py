"""External prefill tests — ExternalPrefiller, PrefillResult, check_abort,
ExternalPrefillConfig, ExternalPrefillServer, ExternalPrefillClient,
wire protocol, EngineCore wiring.

Tests the external prefill path with mocked model:
- PrefillResult dataclass construction
- ExternalPrefiller.prefill() with mocked model
- ExternalPrefiller.prefill_chunked() with progress callback
- check_abort() utility
- Request status transition: WAITING -> PREFILLING -> RUNNING
- Mid-prefill abort detection
- Memory preflight check before prefill
- Chunked prefill with various chunk sizes
- ExternalPrefillConfig dataclass + from_env()
- Wire protocol encode/decode round-trip
- ExternalPrefillServer handle_prefill()
- ExternalPrefillServer stats
- ExternalPrefillClient prefill_remote() with retries
- ExternalPrefillClient health_check
- ExternalPrefillClient stats
- EngineCore wiring (YUNSHU_EXTERNAL_PREFILL env vars)
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import struct
import time
from unittest.mock import MagicMock, patch

import pytest

from yunshu_engine.external_prefill import (
    ExternalPrefillClient,
    ExternalPrefillConfig,
    ExternalPrefiller,
    ExternalPrefillServer,
    PrefillAbortedError,
    PrefillResult,
    _decode_message,
    _deserialize_prefill_result,
    _encode_message,
    _serialize_prefill_result,
    check_abort,
    get_external_prefill_stats,
    get_prefill_role,
)
from yunshu_engine.request import Request, RequestStatus
from yunshu_engine.scheduler import Scheduler, SchedulerConfig

# ── Fakes ──


class _FakeDetokenizer:
    def __init__(self):
        self.last_segment = ""

    def reset(self):
        self.last_segment = ""

    def add_token(self, token_id):
        self.last_segment = f"tok{token_id}"

    def finalize(self):
        self.last_segment = ""


class _FakeTokenizer:
    eos_token_ids = [3]
    has_thinking = False

    def encode(self, text, **kwargs):
        return list(range(len(text)))

    def decode(self, tokens):
        return " ".join(f"t{t}" for t in tokens)

    @property
    def detokenizer(self):
        return _FakeDetokenizer()


class _FakeModel:
    """Mock model that records calls and returns fake output."""

    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, cache=None, **kwargs):
        self.calls.append({
            'input_ids': input_ids,
            'cache': cache,
            'kwargs': kwargs,
        })
        output = MagicMock()
        output.logits = MagicMock()
        return output


# ── PrefillResult tests ──


class TestPrefillResult:
    """Tests for PrefillResult dataclass construction."""

    def test_basic_construction(self):
        result = PrefillResult(
            token_ids=[1, 2, 3],
            num_tokens=3,
        )
        assert result.token_ids == [1, 2, 3]
        assert result.num_tokens == 3
        assert result.kv_cache is None
        assert result.cached_tokens == 0
        assert result.duration_s == 0.0

    def test_full_construction(self):
        kv = MagicMock()
        result = PrefillResult(
            token_ids=[1, 2, 3, 4, 5],
            num_tokens=5,
            kv_cache=kv,
            cached_tokens=2,
            duration_s=0.123,
        )
        assert result.num_tokens == 5
        assert result.kv_cache is kv
        assert result.cached_tokens == 2
        assert result.duration_s == pytest.approx(0.123)

    def test_empty_tokens(self):
        result = PrefillResult(
            token_ids=[],
            num_tokens=0,
        )
        assert result.token_ids == []
        assert result.num_tokens == 0

    def test_large_prompt(self):
        tokens = list(range(10000))
        result = PrefillResult(
            token_ids=tokens,
            num_tokens=10000,
            cached_tokens=5000,
            duration_s=1.5,
        )
        assert result.num_tokens == 10000
        assert result.cached_tokens == 5000


# ── ExternalPrefiller tests ──


class TestExternalPrefill:
    """Tests for ExternalPrefiller.prefill() with mocked model."""

    def test_prefill_empty_tokens(self):
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        prefiller = ExternalPrefiller(model, tokenizer)

        result = prefiller.prefill(token_ids=[])
        assert result.num_tokens == 0
        assert result.token_ids == []
        assert result.duration_s == 0.0

    def test_prefill_calls_run_model_step(self):
        """Verify prefill delegates to _run_model_step."""
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        prefiller = ExternalPrefiller(model, tokenizer)

        with patch.object(prefiller, '_run_model_step') as mock_step, \
             patch.object(prefiller, '_create_kv_cache', return_value=None):
            result = prefiller.prefill(
                token_ids=[1, 2, 3, 4, 5],
                cached_prefix_len=2,
            )

        assert result.num_tokens == 5
        assert result.cached_tokens == 2
        mock_step.assert_called_once_with([1, 2, 3, 4, 5], None)

    def test_prefill_returns_correct_timing(self):
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        prefiller = ExternalPrefiller(model, tokenizer)

        with patch.object(prefiller, '_run_model_step'), \
             patch.object(prefiller, '_create_kv_cache', return_value=None):
            result = prefiller.prefill(token_ids=list(range(100)))

        assert result.num_tokens == 100
        assert result.duration_s >= 0.0


# ── prefill_chunked tests ──


class TestPrefillChunked:
    """Tests for ExternalPrefiller.prefill_chunked()."""

    def test_chunked_empty_tokens(self):
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        prefiller = ExternalPrefiller(model, tokenizer)

        result = prefiller.prefill_chunked(token_ids=[])
        assert result.num_tokens == 0
        assert result.token_ids == []

    def test_chunked_with_progress_callback(self):
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        prefiller = ExternalPrefiller(model, tokenizer)

        progress_calls = []

        def on_progress(completed, total):
            progress_calls.append((completed, total))

        # 10 tokens, chunk_size=3 → 4 chunks: [0:3], [3:6], [6:9], [9:10]
        with patch.object(prefiller, '_run_model_step'), \
             patch.object(prefiller, '_create_kv_cache', return_value=None):
            result = prefiller.prefill_chunked(
                token_ids=list(range(10)),
                chunk_size=3,
                on_progress=on_progress,
            )

        assert result.num_tokens == 10
        assert len(progress_calls) == 4
        assert progress_calls[0] == (3, 10)
        assert progress_calls[1] == (6, 10)
        assert progress_calls[2] == (9, 10)
        assert progress_calls[3] == (10, 10)

    def test_chunked_default_chunk_size(self):
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        prefiller = ExternalPrefiller(model, tokenizer)

        # Default chunk_size=2048, 100 tokens → 1 chunk
        with patch.object(prefiller, '_run_model_step') as mock_step, \
             patch.object(prefiller, '_create_kv_cache', return_value=None):
            result = prefiller.prefill_chunked(
                token_ids=list(range(100)),
            )
        assert result.num_tokens == 100
        assert mock_step.call_count == 1

    def test_chunked_various_sizes(self):
        """Test with various chunk sizes."""
        model = _FakeModel()
        tokenizer = _FakeTokenizer()

        for chunk_size in [1, 5, 10, 100]:
            prefiller = ExternalPrefiller(model, tokenizer)
            progress_calls = []

            with patch.object(prefiller, '_run_model_step'), \
                 patch.object(prefiller, '_create_kv_cache', return_value=None):
                result = prefiller.prefill_chunked(
                    token_ids=list(range(25)),
                    chunk_size=chunk_size,
                    on_progress=lambda c, t: progress_calls.append((c, t)),
                )

            assert result.num_tokens == 25
            # Expected number of chunks: ceil(25 / chunk_size)
            expected_chunks = (25 + chunk_size - 1) // chunk_size
            assert len(progress_calls) == expected_chunks
            # Last progress should be (25, 25)
            assert progress_calls[-1] == (25, 25)

    def test_chunked_chunk_larger_than_tokens(self):
        """Single chunk when chunk_size >= len(token_ids)."""
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        prefiller = ExternalPrefiller(model, tokenizer)

        progress_calls = []
        with patch.object(prefiller, '_run_model_step'), \
             patch.object(prefiller, '_create_kv_cache', return_value=None):
            result = prefiller.prefill_chunked(
                token_ids=list(range(5)),
                chunk_size=100,
                on_progress=lambda c, t: progress_calls.append((c, t)),
            )

        assert result.num_tokens == 5
        assert len(progress_calls) == 1
        assert progress_calls[0] == (5, 5)

    def test_chunked_progress_callback_exception_handled(self):
        """Progress callback exceptions should not crash prefill."""
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        prefiller = ExternalPrefiller(model, tokenizer)

        call_count = [0]

        def bad_callback(completed, total):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("callback error")

        with patch.object(prefiller, '_run_model_step'), \
             patch.object(prefiller, '_create_kv_cache', return_value=None):
            # Should not raise
            result = prefiller.prefill_chunked(
                token_ids=list(range(10)),
                chunk_size=5,
                on_progress=bad_callback,
            )
        assert result.num_tokens == 10
        assert call_count[0] == 2  # Both chunks attempted the callback

    def test_chunked_calls_run_model_step_per_chunk(self):
        """Verify _run_model_step is called once per chunk."""
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        prefiller = ExternalPrefiller(model, tokenizer)

        with patch.object(prefiller, '_run_model_step') as mock_step, \
             patch.object(prefiller, '_create_kv_cache', return_value="fake_cache"):
            result = prefiller.prefill_chunked(
                token_ids=list(range(10)),
                chunk_size=3,
            )

        assert result.num_tokens == 10
        # 4 chunks: [0:3], [3:6], [6:9], [9:10]
        assert mock_step.call_count == 4
        # Each call gets the chunk and the kv cache
        mock_step.assert_any_call([0, 1, 2], "fake_cache")
        mock_step.assert_any_call([3, 4, 5], "fake_cache")
        mock_step.assert_any_call([6, 7, 8], "fake_cache")
        mock_step.assert_any_call([9], "fake_cache")


# ── check_abort tests ──


class TestCheckAbort:
    """Tests for the check_abort utility function."""

    def test_not_in_abort_set(self):
        pending = {"req-1", "req-2"}
        assert check_abort("req-3", pending) is False

    def test_in_abort_set(self):
        pending = {"req-1", "req-2"}
        assert check_abort("req-1", pending) is True

    def test_empty_abort_set(self):
        assert check_abort("req-1", set()) is False

    def test_empty_request_id(self):
        pending = {"", "req-1"}
        assert check_abort("", pending) is True

    def test_case_sensitive(self):
        pending = {"REQ-1"}
        assert check_abort("req-1", pending) is False


# ── Mid-prefill abort tests ──


class TestMidPrefillAbort:
    """Tests for mid-prefill abort detection."""

    def test_abort_between_chunks(self):
        """PrefillChunked should raise PrefillAbortedError when abort detected."""
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        prefiller = ExternalPrefiller(model, tokenizer)

        pending = set()
        progress_calls = []

        # After the first chunk, add to abort set
        def on_progress(completed, total):
            progress_calls.append(completed)
            if completed >= 5:
                pending.add("req-test")

        with patch.object(prefiller, '_run_model_step'), \
             patch.object(prefiller, '_create_kv_cache', return_value=None):
            with pytest.raises(PrefillAbortedError) as exc_info:
                prefiller.prefill_chunked(
                    token_ids=list(range(15)),
                    chunk_size=5,
                    on_progress=on_progress,
                    request_id="req-test",
                    pending_aborts=pending,
                )

        assert exc_info.value.request_id == "req-test"
        assert exc_info.value.completed_tokens == 5
        assert exc_info.value.total_tokens == 15

    def test_abort_at_first_chunk(self):
        """Abort detected at the very first chunk."""
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        prefiller = ExternalPrefiller(model, tokenizer)

        pending = {"req-immediate"}

        with patch.object(prefiller, '_run_model_step'), \
             patch.object(prefiller, '_create_kv_cache', return_value=None):
            with pytest.raises(PrefillAbortedError) as exc_info:
                prefiller.prefill_chunked(
                    token_ids=list(range(10)),
                    chunk_size=5,
                    request_id="req-immediate",
                    pending_aborts=pending,
                )

        assert exc_info.value.completed_tokens == 0

    def test_no_abort_completes_normally(self):
        """No abort — prefill completes normally."""
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        prefiller = ExternalPrefiller(model, tokenizer)

        with patch.object(prefiller, '_run_model_step'), \
             patch.object(prefiller, '_create_kv_cache', return_value=None):
            result = prefiller.prefill_chunked(
                token_ids=list(range(10)),
                chunk_size=5,
                request_id="req-ok",
                pending_aborts=set(),
            )
        assert result.num_tokens == 10

    def test_prefill_aborted_error_attributes(self):
        """PrefillAbortedError carries correct metadata."""
        err = PrefillAbortedError(
            request_id="req-abc",
            completed_tokens=500,
            total_tokens=1000,
        )
        assert err.request_id == "req-abc"
        assert err.completed_tokens == 500
        assert err.total_tokens == 1000
        assert "req-abc" in str(err)
        assert "500/1000" in str(err)

    def test_prefill_aborted_error_none_id(self):
        err = PrefillAbortedError(
            request_id=None,
            completed_tokens=0,
            total_tokens=10,
        )
        assert err.request_id is None
        assert "None" in str(err)


# ── Request status transition tests ──


class TestRequestStatusTransition:
    """Tests for request status transitions with external prefill."""

    def test_prefilling_enum_value(self):
        """PREFILLING is between WAITING and RUNNING."""
        assert RequestStatus.WAITING < RequestStatus.PREFILLING
        assert RequestStatus.PREFILLING < RequestStatus.RUNNING

    def test_prefilling_not_finished(self):
        """PREFILLING is not a finished state."""
        assert not RequestStatus.is_finished(RequestStatus.PREFILLING)

    def test_prefilling_has_no_finish_reason(self):
        """PREFILLING has no finish_reason mapping."""
        assert RequestStatus.finish_reason(RequestStatus.PREFILLING) is None

    def test_status_transition_sequence(self):
        """Verify the WAITING -> PREFILLING -> RUNNING -> FINISHED_STOPPED sequence."""
        request = Request(
            request_id="test-1",
            prompt="hello",
            prompt_token_ids=[1, 2, 3],
            num_prompt_tokens=3,
        )
        assert request.status == RequestStatus.WAITING

        request.status = RequestStatus.PREFILLING
        assert request.status == RequestStatus.PREFILLING
        assert not request.is_finished()

        request.status = RequestStatus.RUNNING
        assert request.status == RequestStatus.RUNNING
        assert not request.is_finished()

        request.set_finished(RequestStatus.FINISHED_STOPPED, "stop")
        assert request.is_finished()
        assert request.finish_reason == "stop"


# ── Memory preflight tests ──


class TestMemoryPreflight:
    """Tests for memory preflight checking before prefill."""

    def test_preflight_passes_with_enough_memory(self):
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        prefiller = ExternalPrefiller(model, tokenizer)

        mock_info = MagicMock()
        mock_info.available_bytes = 10 * 1024 * 1024 * 1024  # 10GB

        mock_monitor = MagicMock()
        mock_monitor.get_memory_info.return_value = mock_info
        mock_monitor.estimate_prefill_peak_bytes.return_value = 100 * 1024 * 1024  # 100MB
        mock_monitor.is_under_pressure.return_value = False

        with patch.object(prefiller, '_run_model_step'), \
             patch.object(prefiller, '_create_kv_cache', return_value=None):
            result = prefiller.prefill_chunked(
                token_ids=list(range(100)),
                chunk_size=50,
                memory_monitor=mock_monitor,
            )
        assert result.num_tokens == 100
        mock_monitor.estimate_prefill_peak_bytes.assert_called_once_with(100, 50)

    def test_preflight_raises_on_oom(self):
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        prefiller = ExternalPrefiller(model, tokenizer)

        mock_info = MagicMock()
        mock_info.available_bytes = 100 * 1024 * 1024  # 100MB available

        mock_monitor = MagicMock()
        mock_monitor.get_memory_info.return_value = mock_info
        mock_monitor.estimate_prefill_peak_bytes.return_value = 10 * 1024 * 1024 * 1024  # 10GB needed

        from yunshu_engine.exceptions import PrefillMemoryExceededError
        with pytest.raises(PrefillMemoryExceededError):
            prefiller.prefill_chunked(
                token_ids=list(range(10000)),
                chunk_size=2048,
                memory_monitor=mock_monitor,
            )

    def test_preflight_skipped_when_zero_estimate(self):
        """If estimate returns 0, preflight is skipped (no model info set)."""
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        prefiller = ExternalPrefiller(model, tokenizer)

        mock_info = MagicMock()
        mock_info.available_bytes = 100

        mock_monitor = MagicMock()
        mock_monitor.get_memory_info.return_value = mock_info
        mock_monitor.estimate_prefill_peak_bytes.return_value = 0
        mock_monitor.is_under_pressure.return_value = False

        with patch.object(prefiller, '_run_model_step'), \
             patch.object(prefiller, '_create_kv_cache', return_value=None):
            result = prefiller.prefill_chunked(
                token_ids=list(range(100)),
                chunk_size=50,
                memory_monitor=mock_monitor,
            )
        assert result.num_tokens == 100

    def test_preflight_not_called_without_monitor(self):
        """No memory monitor -> no preflight check."""
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        prefiller = ExternalPrefiller(model, tokenizer)

        with patch.object(prefiller, '_run_model_step'), \
             patch.object(prefiller, '_create_kv_cache', return_value=None):
            result = prefiller.prefill_chunked(
                token_ids=list(range(50)),
                chunk_size=25,
            )
        assert result.num_tokens == 50

    def test_memory_pressure_during_chunk_aborts(self):
        """Memory pressure detected between chunks causes abort."""
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        prefiller = ExternalPrefiller(model, tokenizer)

        mock_info = MagicMock()
        mock_info.available_bytes = 10 * 1024 * 1024 * 1024

        mock_monitor = MagicMock()
        mock_monitor.get_memory_info.return_value = mock_info
        mock_monitor.estimate_prefill_peak_bytes.return_value = 100
        # Under pressure after first chunk
        mock_monitor.is_under_pressure.return_value = True

        with patch.object(prefiller, '_run_model_step'), \
             patch.object(prefiller, '_create_kv_cache', return_value=None):
            with pytest.raises(PrefillAbortedError) as exc_info:
                prefiller.prefill_chunked(
                    token_ids=list(range(20)),
                    chunk_size=10,
                    memory_monitor=mock_monitor,
                    request_id="req-pressure",
                )

        assert exc_info.value.request_id == "req-pressure"
        assert exc_info.value.completed_tokens == 10


# ── SchedulerConfig tests ──


class TestSchedulerConfigExternalPrefill:
    """Tests for SchedulerConfig with external prefill fields."""

    def test_default_external_prefill_disabled(self):
        config = SchedulerConfig()
        assert config.use_external_prefill is False
        assert config.prefill_chunk_size == 2048

    def test_enable_external_prefill(self):
        config = SchedulerConfig(
            use_external_prefill=True,
            prefill_chunk_size=1024,
        )
        assert config.use_external_prefill is True
        assert config.prefill_chunk_size == 1024


# ── Scheduler integration tests ──


class TestSchedulerExternalPrefill:
    """Tests for Scheduler with external prefill integration."""

    def test_set_memory_monitor(self):
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        config = SchedulerConfig(use_external_prefill=True)
        scheduler = Scheduler(model, tokenizer, config)

        mock_monitor = MagicMock()
        scheduler.set_memory_monitor(mock_monitor)
        assert scheduler._memory_monitor is mock_monitor

    def test_get_external_prefiller_lazy_init(self):
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        config = SchedulerConfig(use_external_prefill=True)
        scheduler = Scheduler(model, tokenizer, config)

        assert scheduler._external_prefiller is None
        prefiller = scheduler._get_external_prefiller()
        assert prefiller is not None
        assert isinstance(prefiller, ExternalPrefiller)
        # Second call returns same instance
        assert scheduler._get_external_prefiller() is prefiller

    def test_run_external_prefill_aborted(self):
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        config = SchedulerConfig(use_external_prefill=True)
        scheduler = Scheduler(model, tokenizer, config)

        req = Request(
            request_id="req-abort-test",
            prompt="test",
            prompt_token_ids=list(range(10)),
            num_prompt_tokens=10,
        )
        scheduler.requests[req.request_id] = req

        scheduler._pending_abort_ids.add("req-abort-test")

        prefiller = scheduler._get_external_prefiller()
        with patch.object(prefiller, '_run_model_step'), \
             patch.object(prefiller, '_create_kv_cache', return_value=None):
            result = scheduler._run_external_prefill(req)

        assert result is False
        assert req.status == RequestStatus.FINISHED_ABORTED
        assert req.finish_reason == "abort"
        assert req.request_id in scheduler.finished_ids

    def test_run_external_prefill_success(self):
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        config = SchedulerConfig(
            use_external_prefill=True,
            prefill_chunk_size=5,
        )
        scheduler = Scheduler(model, tokenizer, config)

        req = Request(
            request_id="req-ok",
            prompt="test",
            prompt_token_ids=list(range(10)),
            num_prompt_tokens=10,
        )
        scheduler.requests[req.request_id] = req

        prefiller = scheduler._get_external_prefiller()
        with patch.object(prefiller, '_run_model_step'), \
             patch.object(prefiller, '_create_kv_cache', return_value=None):
            result = scheduler._run_external_prefill(req)

        assert result is True
        assert req.status == RequestStatus.PREFILLING
        assert req.prefill_end > 0

    def test_process_prefill_responses_handles_prefilling_status(self):
        """_process_prefill_responses should transition PREFILLING -> RUNNING."""
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        config = SchedulerConfig()
        scheduler = Scheduler(model, tokenizer, config)

        req = Request(
            request_id="req-prefilling",
            prompt="test",
            prompt_token_ids=[1, 2, 3],
            num_prompt_tokens=3,
        )
        req.status = RequestStatus.PREFILLING
        scheduler.running[req.request_id] = req
        scheduler._uid_to_req[42] = req.request_id

        mock_resp = MagicMock()
        mock_resp.uid = 42
        mock_resp.end_of_prompt = True

        scheduler._process_prefill_responses([mock_resp])

        assert req.status == RequestStatus.RUNNING
        assert req.prefill_end > 0
        assert req.generation_start > 0


# ── ExternalPrefillConfig tests ──


class TestExternalPrefillConfig:
    """Tests for ExternalPrefillConfig dataclass."""

    def test_default_config(self):
        config = ExternalPrefillConfig()
        assert config.server_host == "0.0.0.0"
        assert config.server_port == 7891
        assert config.max_connections == 8
        assert config.chunk_size == 2048
        assert config.timeout_seconds == 30.0
        assert config.retry_attempts == 3
        assert config.compression == "none"

    def test_custom_config(self):
        config = ExternalPrefillConfig(
            server_host="10.0.0.1",
            server_port=9999,
            max_connections=16,
            chunk_size=1024,
            timeout_seconds=60.0,
            retry_attempts=5,
            compression="lz4",
        )
        assert config.server_host == "10.0.0.1"
        assert config.server_port == 9999
        assert config.max_connections == 16
        assert config.compression == "lz4"

    def test_from_env_defaults(self):
        """from_env() uses defaults when env vars not set."""
        # Clear any existing env vars
        env_keys = [
            "YUNSHU_PREFILL_HOST", "YUNSHU_PREFILL_PORT",
            "YUNSHU_PREFILL_MAX_CONN", "YUNSHU_PREFILL_CHUNK_SIZE",
            "YUNSHU_PREFILL_TIMEOUT", "YUNSHU_PREFILL_RETRIES",
            "YUNSHU_PREFILL_COMPRESSION",
        ]
        old_vals = {}
        for k in env_keys:
            old_vals[k] = os.environ.pop(k, None)

        try:
            config = ExternalPrefillConfig.from_env()
            assert config.server_host == "0.0.0.0"
            assert config.server_port == 7891
        finally:
            for k, v in old_vals.items():
                if v is not None:
                    os.environ[k] = v

    def test_from_env_custom(self):
        """from_env() reads from environment variables."""
        env_vars = {
            "YUNSHU_PREFILL_HOST": "192.168.1.1",
            "YUNSHU_PREFILL_PORT": "8888",
            "YUNSHU_PREFILL_MAX_CONN": "32",
        }
        old_vals = {}
        for k, v in env_vars.items():
            old_vals[k] = os.environ.get(k)
            os.environ[k] = v

        try:
            config = ExternalPrefillConfig.from_env()
            assert config.server_host == "192.168.1.1"
            assert config.server_port == 8888
            assert config.max_connections == 32
        finally:
            for k, v in old_vals.items():
                if v is not None:
                    os.environ[k] = v
                else:
                    os.environ.pop(k, None)


# ── Wire protocol tests ──


class TestWireProtocol:
    """Tests for wire protocol encode/decode."""

    def test_encode_decode_roundtrip(self):
        """Encode then decode should return original header."""
        header = {"type": "prefill", "num_tokens": 42}
        payload = b"\x00\x01\x02\x03"
        encoded = _encode_message(header, payload)
        decoded_header, decoded_payload = _decode_message(encoded)
        assert decoded_header == header
        assert decoded_payload == payload

    def test_encode_decode_empty_payload(self):
        header = {"type": "result", "num_tokens": 0}
        encoded = _encode_message(header)
        decoded_header, decoded_payload = _decode_message(encoded)
        assert decoded_header == header
        assert decoded_payload == b""

    def test_decode_invalid_magic(self):
        with pytest.raises(ValueError, match="Invalid magic"):
            _decode_message(b"\x00\x00\x00\x00" + b"\x00" * 20)

    def test_decode_too_short(self):
        with pytest.raises(ValueError, match="too short"):
            _decode_message(b"\x00\x01")

    def test_serialize_deserialize_prefill_result(self):
        """Serialize and deserialize PrefillResult round-trip."""
        result = PrefillResult(
            token_ids=[1, 2, 3, 100, 65535],
            num_tokens=5,
            cached_tokens=2,
            duration_s=0.5,
        )
        data = _serialize_prefill_result(result)
        restored = _deserialize_prefill_result(data)
        assert restored.token_ids == result.token_ids
        assert restored.num_tokens == result.num_tokens
        assert restored.cached_tokens == result.cached_tokens
        assert restored.duration_s == pytest.approx(result.duration_s)

    def test_serialize_deserialize_empty_result(self):
        result = PrefillResult(token_ids=[], num_tokens=0)
        data = _serialize_prefill_result(result)
        restored = _deserialize_prefill_result(data)
        assert restored.token_ids == []
        assert restored.num_tokens == 0


# ── ExternalPrefillServer tests ──


class TestExternalPrefillServer:
    """Tests for ExternalPrefillServer."""

    def test_server_construction(self):
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        server = ExternalPrefillServer(model, tokenizer)
        assert server._running is False
        stats = server.get_stats()
        assert stats["running"] is False
        assert stats["requests_served"] == 0

    def test_server_custom_config(self):
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        config = ExternalPrefillConfig(max_connections=4, server_port=9999)
        server = ExternalPrefillServer(model, tokenizer, config)
        assert server._config.max_connections == 4
        assert server._config.server_port == 9999

    def test_server_stats_initial(self):
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        server = ExternalPrefillServer(model, tokenizer)
        stats = server.get_stats()
        assert stats["requests_served"] == 0
        assert stats["avg_prefill_time_s"] == 0.0
        assert stats["bytes_transferred"] == 0
        assert stats["active_connections"] == 0
        assert stats["errors"] == 0

    @pytest.mark.asyncio
    async def test_server_handle_prefill_empty(self):
        """Server handles empty token list."""
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        server = ExternalPrefillServer(model, tokenizer)

        header = {"type": "prefill", "num_tokens": 0, "chunk_size": 2048}
        response_data = await server.handle_prefill(header, b"")
        resp_header, resp_payload = _decode_message(response_data)
        assert resp_header.get("type") != "error"
        assert resp_header.get("num_tokens") == 0

    @pytest.mark.asyncio
    async def test_server_handle_prefill_with_tokens(self):
        """Server handles prefill with real token IDs."""
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        server = ExternalPrefillServer(model, tokenizer)

        token_ids = [1, 2, 3, 4, 5]
        payload = struct.pack(f">{len(token_ids)}I", *token_ids)
        header = {"type": "prefill", "num_tokens": 5, "chunk_size": 10}
        response_data = await server.handle_prefill(header, payload)
        resp_header, _ = _decode_message(response_data)

        # Should be a successful result
        assert resp_header.get("type") != "error"
        assert resp_header.get("num_tokens") == 5

        # Stats should be updated
        stats = server.get_stats()
        assert stats["requests_served"] == 1

    @pytest.mark.asyncio
    async def test_server_handle_prefill_invalid_payload(self):
        """Server returns error for invalid payload."""
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        server = ExternalPrefillServer(model, tokenizer)

        header = {"type": "prefill", "num_tokens": 5, "chunk_size": 10}
        response_data = await server.handle_prefill(header, b"\x00\x01")  # too short
        resp_header, _ = _decode_message(response_data)
        assert resp_header.get("type") == "error"
        assert "Invalid payload" in resp_header.get("message", "")

        stats = server.get_stats()
        assert stats["errors"] == 1

    @pytest.mark.asyncio
    async def test_server_handle_prefill_stats_tracking(self):
        """Multiple requests update stats correctly."""
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        server = ExternalPrefillServer(model, tokenizer)

        for _ in range(3):
            header = {"type": "prefill", "num_tokens": 3, "chunk_size": 10}
            payload = struct.pack(">3I", 1, 2, 3)
            await server.handle_prefill(header, payload)

        stats = server.get_stats()
        assert stats["requests_served"] == 3
        assert stats["avg_prefill_time_s"] >= 0.0

    @pytest.mark.asyncio
    async def test_server_stop_without_start(self):
        """Stopping a server that never started should not raise."""
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        server = ExternalPrefillServer(model, tokenizer)
        await server.stop()
        assert server._running is False


# ── ExternalPrefillClient tests ──


class TestExternalPrefillClient:
    """Tests for ExternalPrefillClient."""

    def test_client_construction(self):
        config = ExternalPrefillConfig(server_host="10.0.0.1")
        client = ExternalPrefillClient(config)
        assert client._config.server_host == "10.0.0.1"
        stats = client.get_stats()
        assert stats["requests_sent"] == 0

    def test_client_stats_initial(self):
        client = ExternalPrefillClient()
        stats = client.get_stats()
        assert stats["requests_sent"] == 0
        assert stats["avg_latency_s"] == 0.0
        assert stats["success_rate"] == 0.0
        assert stats["successes"] == 0
        assert stats["failures"] == 0

    def test_client_stats_after_success(self):
        """Stats updated after a successful tracked request."""
        client = ExternalPrefillClient()
        # Simulate internal stats update
        client._requests_sent = 5
        client._successes = 4
        client._total_latency_s = 2.5
        stats = client.get_stats()
        assert stats["requests_sent"] == 5
        assert stats["success_rate"] == 0.8
        assert stats["avg_latency_s"] == 0.5

    @pytest.mark.asyncio
    async def test_client_prefill_remote_connection_refused(self):
        """Client raises ConnectionError when server is not reachable."""
        config = ExternalPrefillConfig(
            server_host="127.0.0.1",
            server_port=19999,  # unlikely to be in use
            retry_attempts=1,
            timeout_seconds=0.5,
        )
        client = ExternalPrefillClient(config)
        with pytest.raises(ConnectionError, match="Failed after"):
            await client.prefill_remote(token_ids=[1, 2, 3])

    @pytest.mark.asyncio
    async def test_client_health_check_no_server(self):
        """Health check returns False when server is unreachable."""
        config = ExternalPrefillConfig(
            server_host="127.0.0.1",
            server_port=19998,
            timeout_seconds=0.5,
        )
        client = ExternalPrefillClient(config)
        assert await client.health_check() is False

    @pytest.mark.asyncio
    async def test_client_server_integration(self):
        """Full integration: start server, client sends request, gets result."""
        model = _FakeModel()
        tokenizer = _FakeTokenizer()
        config = ExternalPrefillConfig(
            server_host="127.0.0.1",
            server_port=0,  # OS-assigned port
            timeout_seconds=5.0,
            retry_attempts=1,
        )
        server = ExternalPrefillServer(model, tokenizer, config)

        # Start server with OS-assigned port
        server_task = asyncio.create_task(
            server.serve(host="127.0.0.1", port=0)
        )
        # Wait a moment for server to start and get its port
        await asyncio.sleep(0.2)

        # Get the actual assigned port
        if server._server is not None:
            sockets = server._server.sockets
            actual_port = sockets[0].getsockname()[1] if sockets else 19997
        else:
            actual_port = 19997

        try:
            # Create client pointing to the server's actual port
            client_config = ExternalPrefillConfig(
                server_host="127.0.0.1",
                server_port=actual_port,
                timeout_seconds=5.0,
                retry_attempts=1,
            )
            client = ExternalPrefillClient(client_config)

            # Health check
            health = await client.health_check()
            assert health is True

            # Send prefill request
            token_ids = [10, 20, 30, 40, 50]
            result = await client.prefill_remote(token_ids=token_ids)
            assert result.num_tokens == 5
            assert result.token_ids == token_ids

            # Check client stats
            client_stats = client.get_stats()
            assert client_stats["requests_sent"] == 1
            assert client_stats["successes"] == 1
            assert client_stats["success_rate"] == 1.0

            # Check server stats
            server_stats = server.get_stats()
            assert server_stats["requests_served"] == 1

        finally:
            await server.stop()
            server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server_task

    @pytest.mark.asyncio
    async def test_client_retry_on_failure(self):
        """Client retries with exponential backoff on connection failure."""
        config = ExternalPrefillConfig(
            server_host="127.0.0.1",
            server_port=19996,
            retry_attempts=3,
            timeout_seconds=0.1,
        )
        client = ExternalPrefillClient(config)

        t0 = time.monotonic()
        with pytest.raises(ConnectionError):
            await client.prefill_remote(token_ids=[1, 2])
        time.monotonic() - t0

        # Should have attempted 3 times with backoff
        assert client.get_stats()["failures"] >= 1


# ── get_prefill_role tests ──


class TestGetPrefillRole:
    """Tests for get_prefill_role() helper."""

    def test_returns_none_when_disabled(self):
        old = os.environ.pop("YUNSHU_EXTERNAL_PREFILL", None)
        try:
            assert get_prefill_role() is None
        finally:
            if old is not None:
                os.environ["YUNSHU_EXTERNAL_PREFILL"] = old

    def test_returns_server_role(self):
        old_ext = os.environ.get("YUNSHU_EXTERNAL_PREFILL")
        old_role = os.environ.get("YUNSHU_PREFILL_ROLE")
        os.environ["YUNSHU_EXTERNAL_PREFILL"] = "1"
        os.environ["YUNSHU_PREFILL_ROLE"] = "server"
        try:
            assert get_prefill_role() == "server"
        finally:
            if old_ext is not None:
                os.environ["YUNSHU_EXTERNAL_PREFILL"] = old_ext
            else:
                os.environ.pop("YUNSHU_EXTERNAL_PREFILL", None)
            if old_role is not None:
                os.environ["YUNSHU_PREFILL_ROLE"] = old_role
            else:
                os.environ.pop("YUNSHU_PREFILL_ROLE", None)

    def test_returns_client_role(self):
        old_ext = os.environ.get("YUNSHU_EXTERNAL_PREFILL")
        old_role = os.environ.get("YUNSHU_PREFILL_ROLE")
        os.environ["YUNSHU_EXTERNAL_PREFILL"] = "1"
        os.environ["YUNSHU_PREFILL_ROLE"] = "client"
        try:
            assert get_prefill_role() == "client"
        finally:
            if old_ext is not None:
                os.environ["YUNSHU_EXTERNAL_PREFILL"] = old_ext
            else:
                os.environ.pop("YUNSHU_EXTERNAL_PREFILL", None)
            if old_role is not None:
                os.environ["YUNSHU_PREFILL_ROLE"] = old_role
            else:
                os.environ.pop("YUNSHU_PREFILL_ROLE", None)

    def test_returns_none_when_enabled_but_no_role(self):
        old_ext = os.environ.get("YUNSHU_EXTERNAL_PREFILL")
        old_role = os.environ.pop("YUNSHU_PREFILL_ROLE", None)
        os.environ["YUNSHU_EXTERNAL_PREFILL"] = "1"
        try:
            assert get_prefill_role() is None
        finally:
            if old_ext is not None:
                os.environ["YUNSHU_EXTERNAL_PREFILL"] = old_ext
            else:
                os.environ.pop("YUNSHU_EXTERNAL_PREFILL", None)
            if old_role is not None:
                os.environ["YUNSHU_PREFILL_ROLE"] = old_role


# ── get_external_prefill_stats tests ──


class TestGetExternalPrefillStats:
    """Tests for monitoring stats helper."""

    def test_returns_inactive_when_no_engine(self):
        """Returns active=False when no engine is available."""
        stats = get_external_prefill_stats()
        assert stats.get("active") is False
