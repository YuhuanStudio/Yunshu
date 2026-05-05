"""Tests for yunshu_engine.benchmark — BenchmarkRunner and data classes.

All tests use a mock engine (no real model required).
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from yunshu_engine.benchmark import (
    BatchBenchmarkResult,
    BenchmarkResult,
    BenchmarkRunner,
    BenchmarkSuite,
    PrefillDecodePoint,
    _make_prompt,
    _percentile,
)


# ── Fixtures ──


def _make_mock_engine(
    prompt_tokens: int = 50,
    completion_tokens: int = 10,
    delay_per_token: float = 0.001,
) -> AsyncMock:
    """Create a mock engine that simulates streaming generation.

    The mock yields completion_tokens chunks with a small delay to simulate
    realistic timing. prompt_tokens and completion_tokens are reported in the
    output.
    """
    engine = AsyncMock()

    async def _generate(prompt: str, max_tokens: int = 256, **kwargs):
        return type("GenResult", (), {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "text": "x " * completion_tokens,
            "finished": True,
            "finish_reason": "length",
        })()

    async def _stream(prompt: str, max_tokens: int = 256, temperature: float = 0.0, **kwargs):
        for i in range(completion_tokens):
            await asyncio.sleep(delay_per_token)
            chunk = type("Chunk", (), {
                "new_text": "x ",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": i + 1,
                "finished": i == completion_tokens - 1,
                "finish_reason": "length" if i == completion_tokens - 1 else None,
            })()
            yield chunk

    engine.generate = _generate
    engine.stream_generate = _stream
    return engine


# ── Data Class Tests ──


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass construction and derived calculations."""

    def test_construction(self):
        r = BenchmarkResult(
            prompt_tokens=128,
            completion_tokens=256,
            ttft_ms=12.5,
            tpot_ms=4.3,
            total_latency_ms=1100.0,
            throughput_tps=230.0,
            prefill_time_ms=12.5,
            decode_time_ms=1087.5,
        )
        assert r.prompt_tokens == 128
        assert r.completion_tokens == 256
        assert r.ttft_ms == 12.5
        assert r.tpot_ms == 4.3
        assert r.total_latency_ms == 1100.0
        assert r.throughput_tps == 230.0
        assert r.prefill_time_ms == 12.5
        assert r.decode_time_ms == 1087.5

    def test_to_dict(self):
        r = BenchmarkResult(
            prompt_tokens=10, completion_tokens=20,
            ttft_ms=1.0, tpot_ms=2.0, total_latency_ms=3.0,
            throughput_tps=4.0, prefill_time_ms=5.0, decode_time_ms=6.0,
        )
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["prompt_tokens"] == 10
        assert d["completion_tokens"] == 20
        assert d["ttft_ms"] == 1.0

    def test_zero_values(self):
        r = BenchmarkResult(
            prompt_tokens=0, completion_tokens=0,
            ttft_ms=0.0, tpot_ms=0.0, total_latency_ms=0.0,
            throughput_tps=0.0, prefill_time_ms=0.0, decode_time_ms=0.0,
        )
        assert r.prompt_tokens == 0
        assert r.completion_tokens == 0
        assert r.throughput_tps == 0.0


class TestBatchBenchmarkResult:
    """Test BatchBenchmarkResult percentile calculations."""

    def _make_single_result(self, latency: float) -> BenchmarkResult:
        return BenchmarkResult(
            prompt_tokens=10, completion_tokens=20,
            ttft_ms=latency * 0.1, tpot_ms=1.0, total_latency_ms=latency,
            throughput_tps=10.0, prefill_time_ms=latency * 0.1, decode_time_ms=latency * 0.9,
        )

    def test_construction_and_percentiles(self):
        per_req = [self._make_single_result(float(i * 100)) for i in range(1, 11)]
        batch = BatchBenchmarkResult(
            num_requests=10,
            total_tokens=sum(r.prompt_tokens + r.completion_tokens for r in per_req),
            total_time_s=1.0,
            aggregate_tps=300.0,
            per_request=per_req,
            latency_p50_ms=550.0,
            latency_p95_ms=950.0,
            latency_p99_ms=990.0,
        )
        assert batch.num_requests == 10
        assert batch.latency_p50_ms == 550.0
        assert batch.latency_p95_ms == 950.0
        assert batch.latency_p99_ms == 990.0

    def test_to_dict(self):
        batch = BatchBenchmarkResult(
            num_requests=2, total_tokens=60, total_time_s=0.5,
            aggregate_tps=120.0,
            per_request=[
                self._make_single_result(100.0),
                self._make_single_result(200.0),
            ],
            latency_p50_ms=150.0, latency_p95_ms=190.0, latency_p99_ms=198.0,
        )
        d = batch.to_dict()
        assert isinstance(d, dict)
        assert len(d["per_request"]) == 2
        assert d["aggregate_tps"] == 120.0


class TestPercentileHelper:
    """Test the _percentile helper function."""

    def test_empty(self):
        assert _percentile([], 50) == 0.0

    def test_single(self):
        assert _percentile([10.0], 50) == 10.0

    def test_p50(self):
        data = sorted([float(i) for i in range(100)])
        p50 = _percentile(data, 50)
        assert 49.0 <= p50 <= 51.0

    def test_p95(self):
        data = sorted([float(i) for i in range(100)])
        p95 = _percentile(data, 95)
        assert 93.0 <= p95 <= 96.0

    def test_p99(self):
        data = sorted([float(i) for i in range(100)])
        p99 = _percentile(data, 99)
        assert p99 == 99.0


class TestMakePrompt:
    """Test _make_prompt helper."""

    def test_returns_string(self):
        assert isinstance(_make_prompt(128), str)

    def test_non_empty(self):
        assert len(_make_prompt(10)) > 0

    def test_larger_prompt_is_longer(self):
        assert len(_make_prompt(512)) > len(_make_prompt(128))


# ── BenchmarkRunner Tests ──


class TestBenchSingleRequest:
    """Test BenchmarkRunner.bench_single_request with mocked engine."""

    @pytest.mark.asyncio
    async def test_basic_result(self):
        engine = _make_mock_engine(prompt_tokens=50, completion_tokens=10)
        runner = BenchmarkRunner(engine)
        result = await runner.bench_single_request(
            prompt_tokens=50, max_tokens=10,
        )
        assert isinstance(result, BenchmarkResult)
        assert result.prompt_tokens == 50
        assert result.completion_tokens == 10
        assert result.ttft_ms > 0
        assert result.total_latency_ms > 0
        assert result.throughput_tps > 0

    @pytest.mark.asyncio
    async def test_zero_max_tokens(self):
        """Edge case: engine produces 0 tokens."""
        engine = _make_mock_engine(prompt_tokens=10, completion_tokens=0)

        # Override stream to produce nothing
        async def _empty_stream(prompt, max_tokens=0, **kwargs):
            return
            yield  # make this an async generator

        engine.stream_generate = _empty_stream
        runner = BenchmarkRunner(engine)
        result = await runner.bench_single_request(
            prompt_tokens=10, max_tokens=0,
        )
        assert isinstance(result, BenchmarkResult)
        assert result.ttft_ms >= 0
        assert result.throughput_tps == 0.0

    @pytest.mark.asyncio
    async def test_single_token(self):
        """Edge case: engine produces exactly 1 token (no decode phase)."""
        engine = _make_mock_engine(prompt_tokens=10, completion_tokens=1)
        runner = BenchmarkRunner(engine)
        result = await runner.bench_single_request(
            prompt_tokens=10, max_tokens=1,
        )
        assert isinstance(result, BenchmarkResult)
        assert result.ttft_ms > 0
        assert result.completion_tokens == 1


class TestBenchBatch:
    """Test BenchmarkRunner.bench_batch with mocked engine."""

    @pytest.mark.asyncio
    async def test_batch_results(self):
        engine = _make_mock_engine(prompt_tokens=30, completion_tokens=5)
        runner = BenchmarkRunner(engine)
        result = await runner.bench_batch(
            prompts=3, max_tokens=5, concurrency=3, prompt_tokens=30,
        )
        assert isinstance(result, BatchBenchmarkResult)
        assert result.num_requests == 3
        assert result.total_tokens > 0
        assert result.total_time_s > 0
        assert result.aggregate_tps > 0
        assert len(result.per_request) == 3

    @pytest.mark.asyncio
    async def test_batch_with_string_prompts(self):
        engine = _make_mock_engine(prompt_tokens=20, completion_tokens=3)
        runner = BenchmarkRunner(engine)
        result = await runner.bench_batch(
            prompts=["hello world", "test prompt", "another one"],
            max_tokens=3,
            concurrency=2,
        )
        assert result.num_requests == 3

    @pytest.mark.asyncio
    async def test_batch_percentiles(self):
        engine = _make_mock_engine(prompt_tokens=20, completion_tokens=5)
        runner = BenchmarkRunner(engine)
        result = await runner.bench_batch(
            prompts=5, max_tokens=5, concurrency=5, prompt_tokens=20,
        )
        assert result.latency_p50_ms > 0
        assert result.latency_p95_ms >= result.latency_p50_ms
        assert result.latency_p99_ms >= result.latency_p95_ms


class TestBenchPrefillVsDecode:
    """Test BenchmarkRunner.bench_prefill_vs_decode."""

    @pytest.mark.asyncio
    async def test_sweep(self):
        engine = _make_mock_engine(prompt_tokens=50, completion_tokens=4)
        runner = BenchmarkRunner(engine)
        points = await runner.bench_prefill_vs_decode(
            prompt_lengths=[64, 128],
            max_decode_tokens=4,
        )
        assert len(points) == 2
        assert all(isinstance(p, PrefillDecodePoint) for p in points)
        assert points[0].prompt_length == 64
        assert points[1].prompt_length == 128

    @pytest.mark.asyncio
    async def test_default_lengths(self):
        engine = _make_mock_engine(prompt_tokens=50, completion_tokens=2)
        runner = BenchmarkRunner(engine)
        points = await runner.bench_prefill_vs_decode(
            max_decode_tokens=2,
        )
        assert len(points) == 6  # default lengths: [128, 512, 1024, 2048, 4096, 8192]


class TestRunSuite:
    """Test BenchmarkRunner.run_suite returns structured results."""

    @pytest.mark.asyncio
    async def test_default_suite(self):
        engine = _make_mock_engine(prompt_tokens=50, completion_tokens=5)
        runner = BenchmarkRunner(engine)
        suite = await runner.run_suite("test-model", suite_name="default")
        assert isinstance(suite, BenchmarkSuite)
        assert suite.name == "default"
        assert suite.model == "test-model"
        assert len(suite.results) == 5

        # First 3 should be single request results
        for i in range(3):
            assert isinstance(suite.results[i], BenchmarkResult), f"Result {i} should be BenchmarkResult"

        # Last 2 should be batch results
        for i in range(3, 5):
            assert isinstance(suite.results[i], BatchBenchmarkResult), f"Result {i} should be BatchBenchmarkResult"

    @pytest.mark.asyncio
    async def test_suite_timestamp(self):
        engine = _make_mock_engine(prompt_tokens=10, completion_tokens=2)
        runner = BenchmarkRunner(engine)
        suite = await runner.run_suite("model")
        # Timestamp should be ISO 8601
        parsed = datetime.fromisoformat(suite.timestamp)
        assert parsed is not None

    @pytest.mark.asyncio
    async def test_suite_to_dict(self):
        engine = _make_mock_engine(prompt_tokens=10, completion_tokens=2)
        runner = BenchmarkRunner(engine)
        suite = await runner.run_suite("model")
        d = suite.to_dict()
        assert isinstance(d, dict)
        assert "results" in d
        assert "model" in d
        # Should be JSON-serializable
        json.dumps(d)


class TestFormatResults:
    """Test format_results produces valid markdown table."""

    def test_format_single(self):
        r = BenchmarkResult(
            prompt_tokens=128, completion_tokens=256,
            ttft_ms=12.5, tpot_ms=4.3, total_latency_ms=1100.0,
            throughput_tps=230.0, prefill_time_ms=12.5, decode_time_ms=1087.5,
        )
        output = BenchmarkRunner.format_results(r)
        assert "### Single Request Benchmark" in output
        assert "| Metric | Value |" in output
        assert "128" in output
        assert "230.0 tok/s" in output

    def test_format_batch(self):
        r = BatchBenchmarkResult(
            num_requests=4, total_tokens=1000, total_time_s=2.5,
            aggregate_tps=400.0, per_request=[],
            latency_p50_ms=200.0, latency_p95_ms=300.0, latency_p99_ms=350.0,
        )
        output = BenchmarkRunner.format_results(r)
        assert "### Batch Benchmark" in output
        assert "400.0 tok/s" in output
        assert "300.0 ms" in output

    def test_format_suite(self):
        suite = BenchmarkSuite(
            name="test",
            model="test-model",
            chip="Apple M2",
            results=[
                BenchmarkResult(
                    prompt_tokens=10, completion_tokens=20,
                    ttft_ms=1.0, tpot_ms=2.0, total_latency_ms=3.0,
                    throughput_tps=4.0, prefill_time_ms=5.0, decode_time_ms=6.0,
                ),
                BatchBenchmarkResult(
                    num_requests=2, total_tokens=60, total_time_s=0.5,
                    aggregate_tps=120.0, per_request=[],
                    latency_p50_ms=100.0, latency_p95_ms=150.0, latency_p99_ms=180.0,
                ),
            ],
            timestamp="2025-01-01T00:00:00+00:00",
        )
        output = BenchmarkRunner.format_results(suite)
        assert "### Benchmark Suite: test" in output
        assert "test-model" in output
        assert "Apple M2" in output
        assert "#### Test 1: Single Request" in output
        assert "#### Test 2: Batch" in output

    def test_format_unknown_type(self):
        output = BenchmarkRunner.format_results("not a result")
        assert output == "not a result"

    def test_format_results_is_valid_markdown_table(self):
        """Verify the output has proper table formatting."""
        r = BenchmarkResult(
            prompt_tokens=10, completion_tokens=20,
            ttft_ms=1.0, tpot_ms=2.0, total_latency_ms=3.0,
            throughput_tps=4.0, prefill_time_ms=5.0, decode_time_ms=6.0,
        )
        output = BenchmarkRunner.format_results(r)
        lines = output.strip().split("\n")
        # Count separator lines (contain |---|)
        separators = [l for l in lines if "|---" in l]
        assert len(separators) >= 1, "Should have at least one table separator"


class TestEdgeCases:
    """Test edge cases."""

    @pytest.mark.asyncio
    async def test_empty_prompt_string(self):
        """Engine receives empty prompt string."""
        engine = _make_mock_engine(prompt_tokens=1, completion_tokens=2)

        # Override generate to simulate empty prompt
        async def _gen(prompt, **kwargs):
            return type("R", (), {
                "prompt_tokens": 0,
                "completion_tokens": 2,
                "text": "x",
                "finished": True,
                "finish_reason": "length",
            })()

        engine.generate = _gen

        runner = BenchmarkRunner(engine)
        # _make_prompt(0) still produces a string
        result = await runner.bench_single_request(
            prompt_tokens=0, max_tokens=2,
        )
        assert isinstance(result, BenchmarkResult)

    def test_benchmark_result_json_serializable(self):
        r = BenchmarkResult(
            prompt_tokens=0, completion_tokens=0,
            ttft_ms=0.0, tpot_ms=0.0, total_latency_ms=0.0,
            throughput_tps=0.0, prefill_time_ms=0.0, decode_time_ms=0.0,
        )
        serialized = json.dumps(r.to_dict())
        assert isinstance(serialized, str)

    def test_prefill_decode_point(self):
        p = PrefillDecodePoint(
            prompt_length=1024,
            prefill_time_ms=50.0,
            decode_tps=100.0,
            decode_time_ms=200.0,
            completion_tokens=20,
        )
        assert p.prompt_length == 1024
        assert p.decode_tps == 100.0
