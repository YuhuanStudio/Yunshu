from __future__ import annotations

"""Yunshu Benchmark Engine — LLM inference performance measurement.

.. deprecated:: This module is not used in the production pipeline. Kept for reference only.


Provides BenchmarkRunner that works with any Engine or BatchedEngine instance.
All engine interactions are async (generate / stream_generate) so the runner
itself is fully async. For CLI usage the caller wraps in asyncio.run().

Design goals:
- No real model required — takes an engine object that can be mocked
- Separates prefill vs decode timing via streaming token-by-token measurement
- Produces structured dataclasses suitable for JSON serialization or markdown tables
"""

import asyncio
import platform
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

# ── Data Classes ──


@dataclass
class BenchmarkResult:
    """Metrics from a single generation request."""

    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float  # Time to first token (ms)
    tpot_ms: float  # Time per output token (ms)
    total_latency_ms: float
    throughput_tps: float  # tokens per second
    prefill_time_ms: float
    decode_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchBenchmarkResult:
    """Aggregate metrics from multiple concurrent requests."""

    num_requests: int
    total_tokens: int
    total_time_s: float
    aggregate_tps: float
    per_request: list[BenchmarkResult]
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class PrefillDecodePoint:
    """One data point in a prefill-vs-decode sweep."""

    prompt_length: int
    prefill_time_ms: float
    decode_tps: float
    decode_time_ms: float
    completion_tokens: int


@dataclass
class BenchmarkSuite:
    """Complete benchmark run results."""

    name: str
    model: str
    chip: str
    results: list[BenchmarkResult | BatchBenchmarkResult]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


# ── Engine Protocol ──


@runtime_checkable
class _EngineProto(Protocol):
    """Minimal interface BenchmarkRunner needs from an engine.

    Both Engine and BatchedEngine satisfy this protocol via their
    generate / stream_generate methods.
    """

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Any: ...

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Any: ...


# ── Helpers ──


def _percentile(sorted_data: list[float], p: float) -> float:
    """Compute percentile from already-sorted list."""
    if not sorted_data:
        return 0.0
    idx = int(len(sorted_data) * p / 100.0)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


def _make_prompt(token_count: int) -> str:
    """Build a prompt of approximately *token_count* tokens.

    Uses repeated words since most tokenizers encode common English words
    as single tokens. The resulting string will be close to the target
    but not exact — the engine's tokenizer will determine the real count.
    """
    # "word " is roughly 1 token for most tokenizers
    return ("benchmark " * (token_count + 1)).strip()


def _get_chip_name() -> str:
    """Best-effort Apple Silicon chip identification."""
    return platform.processor() or "Apple Silicon"


# ── BenchmarkRunner ──


class BenchmarkRunner:
    """Benchmark an Engine or BatchedEngine instance.

    Usage::

        engine = BatchedEngine(model_name="Qwen2.5-0.5B-Instruct-4bit")
        await engine.start()

        runner = BenchmarkRunner(engine)
        result = await runner.bench_single_request(prompt_tokens=128, max_tokens=256)
        print(result)

        suite = await runner.run_suite("Qwen2.5-0.5B-Instruct-4bit")
        print(BenchmarkRunner.format_results(suite))
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    # ── Single request ──

    async def bench_single_request(
        self,
        prompt_tokens: int = 128,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> BenchmarkResult:
        """Benchmark a single request, measuring TTFT, TPOT, and throughput.

        Uses streaming (stream_generate) to separate prefill from decode timing.
        """
        prompt = _make_prompt(prompt_tokens)

        # Phase 1: Non-streaming generate to get actual prompt token count
        gen_result = await self.engine.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        actual_prompt_tokens = (
            getattr(gen_result, "prompt_tokens", prompt_tokens) or prompt_tokens
        )
        actual_completion_tokens = (
            getattr(gen_result, "completion_tokens", max_tokens) or max_tokens
        )

        # Phase 2: Streaming to measure TTFT / TPOT
        ttft_ms = 0.0
        first_token_time = 0.0
        token_times: list[float] = []
        start = time.perf_counter()
        token_count = 0

        async for chunk in self.engine.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            now = time.perf_counter()
            token_count += 1
            if token_count == 1:
                ttft_ms = (now - start) * 1000.0
                first_token_time = now
            token_times.append(now)

            finished = getattr(chunk, "finished", False)
            if finished:
                # Update actual counts from the final chunk
                pt = getattr(chunk, "prompt_tokens", None)
                ct = getattr(chunk, "completion_tokens", None)
                if pt is not None and pt > 0:
                    actual_prompt_tokens = pt
                if ct is not None and ct > 0:
                    actual_completion_tokens = ct
                break

        end = time.perf_counter()
        total_latency_ms = (end - start) * 1000.0

        if token_count < 2:
            # Degenerate: only 0 or 1 token produced
            decode_time_ms = 0.0
            prefill_time_ms = total_latency_ms
            tpot_ms = 0.0
            throughput_tps = 0.0
        else:
            prefill_time_ms = ttft_ms
            decode_time_ms = (token_times[-1] - first_token_time) * 1000.0
            decode_tokens = token_count - 1  # first token is part of prefill
            if decode_tokens > 0 and decode_time_ms > 0:
                tpot_ms = decode_time_ms / decode_tokens
                throughput_tps = decode_tokens / (decode_time_ms / 1000.0)
            else:
                tpot_ms = 0.0
                throughput_tps = 0.0

        return BenchmarkResult(
            prompt_tokens=actual_prompt_tokens,
            completion_tokens=actual_completion_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_latency_ms=total_latency_ms,
            throughput_tps=throughput_tps,
            prefill_time_ms=prefill_time_ms,
            decode_time_ms=decode_time_ms,
        )

    # ── Batch ──

    async def bench_batch(
        self,
        prompts: list[str] | int = 4,
        max_tokens: int = 256,
        concurrency: int | None = None,
        prompt_tokens: int = 128,
        temperature: float = 0.0,
    ) -> BatchBenchmarkResult:
        """Benchmark multiple concurrent requests.

        Args:
            prompts: Either a list of prompt strings, or an int specifying how
                many synthetic prompts to create.
            max_tokens: Maximum output tokens per request.
            concurrency: Maximum concurrent requests (defaults to len(prompts)).
            prompt_tokens: Approximate token count for synthetic prompts.
            temperature: Sampling temperature.
        """
        if isinstance(prompts, int):
            num = prompts
            prompts = [_make_prompt(prompt_tokens) for _ in range(num)]
        else:
            num = len(prompts)

        if concurrency is None:
            concurrency = num

        semaphore = asyncio.Semaphore(concurrency)
        per_request_results: list[BenchmarkResult] = [None] * num  # type: ignore[list-item]

        async def _one_request(idx: int) -> None:
            async with semaphore:
                result = await self.bench_single_request(
                    prompt_tokens=prompt_tokens,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                per_request_results[idx] = result

        t0 = time.perf_counter()
        # return_exceptions=True so one failed request doesn't
        # kill the whole benchmark and orphan sibling tasks.
        await asyncio.gather(
            *[_one_request(i) for i in range(num)],
            return_exceptions=True,
        )
        total_time_s = time.perf_counter() - t0

        # Filter out any None (shouldn't happen but be safe)
        valid_results: list[BenchmarkResult] = [
            r for r in per_request_results if r is not None
        ]

        total_tokens = sum(r.prompt_tokens + r.completion_tokens for r in valid_results)
        aggregate_tps = total_tokens / total_time_s if total_time_s > 0 else 0.0

        latencies = sorted(r.total_latency_ms for r in valid_results)

        return BatchBenchmarkResult(
            num_requests=len(valid_results),
            total_tokens=total_tokens,
            total_time_s=total_time_s,
            aggregate_tps=aggregate_tps,
            per_request=valid_results,
            latency_p50_ms=_percentile(latencies, 50),
            latency_p95_ms=_percentile(latencies, 95),
            latency_p99_ms=_percentile(latencies, 99),
        )

    # ── Prefill vs Decode sweep ──

    async def bench_prefill_vs_decode(
        self,
        prompt_lengths: Sequence[int] | None = None,
        max_decode_tokens: int = 64,
        temperature: float = 0.0,
    ) -> list[PrefillDecodePoint]:
        """Vary prompt length and measure prefill time + decode throughput separately.

        Returns a list of data points suitable for plotting.
        """
        if prompt_lengths is None:
            prompt_lengths = [128, 512, 1024, 2048, 4096, 8192]

        points: list[PrefillDecodePoint] = []
        for plen in prompt_lengths:
            result = await self.bench_single_request(
                prompt_tokens=plen,
                max_tokens=max_decode_tokens,
                temperature=temperature,
            )
            points.append(
                PrefillDecodePoint(
                    prompt_length=plen,
                    prefill_time_ms=result.prefill_time_ms,
                    decode_tps=result.throughput_tps,
                    decode_time_ms=result.decode_time_ms,
                    completion_tokens=result.completion_tokens,
                )
            )
        return points

    # ── Suite ──

    async def run_suite(
        self,
        model_name: str,
        suite_name: str = "default",
    ) -> BenchmarkSuite:
        """Run a predefined benchmark suite.

        Suite "default":
        1. Short prompt (128 tokens) + 256 decode
        2. Medium prompt (1024 tokens) + 512 decode
        3. Long prompt (4096 tokens) + 256 decode
        4. Batch: 4 concurrent x medium prompt
        5. Batch: 8 concurrent x short prompt
        """
        results: list[BenchmarkResult | BatchBenchmarkResult] = []

        # 1. Short prompt
        results.append(
            await self.bench_single_request(prompt_tokens=128, max_tokens=256)
        )

        # 2. Medium prompt
        results.append(
            await self.bench_single_request(prompt_tokens=1024, max_tokens=512)
        )

        # 3. Long prompt
        results.append(
            await self.bench_single_request(prompt_tokens=4096, max_tokens=256)
        )

        # 4. Batch: 4 x medium
        results.append(
            await self.bench_batch(
                prompts=4,
                max_tokens=256,
                concurrency=4,
                prompt_tokens=1024,
            )
        )

        # 5. Batch: 8 x short
        results.append(
            await self.bench_batch(
                prompts=8,
                max_tokens=128,
                concurrency=8,
                prompt_tokens=128,
            )
        )

        return BenchmarkSuite(
            name=suite_name,
            model=model_name,
            chip=_get_chip_name(),
            results=results,
            timestamp=datetime.now(UTC).isoformat(),
        )

    # ── Formatting ──

    @staticmethod
    def format_results(
        results: BenchmarkSuite | BenchmarkResult | BatchBenchmarkResult,
    ) -> str:
        """Format benchmark results as a markdown table."""
        if isinstance(results, BenchmarkSuite):
            return BenchmarkRunner._format_suite(results)
        if isinstance(results, BatchBenchmarkResult):
            return BenchmarkRunner._format_batch(results)
        if isinstance(results, BenchmarkResult):
            return BenchmarkRunner._format_single(results)
        return str(results)

    @staticmethod
    def _format_single(r: BenchmarkResult) -> str:
        lines = [
            "### Single Request Benchmark",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Prompt tokens | {r.prompt_tokens} |",
            f"| Completion tokens | {r.completion_tokens} |",
            f"| TTFT | {r.ttft_ms:.1f} ms |",
            f"| TPOT | {r.tpot_ms:.2f} ms |",
            f"| Throughput | {r.throughput_tps:.1f} tok/s |",
            f"| Prefill time | {r.prefill_time_ms:.1f} ms |",
            f"| Decode time | {r.decode_time_ms:.1f} ms |",
            f"| Total latency | {r.total_latency_ms:.1f} ms |",
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_batch(r: BatchBenchmarkResult) -> str:
        lines = [
            "### Batch Benchmark",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Requests | {r.num_requests} |",
            f"| Total tokens | {r.total_tokens} |",
            f"| Total time | {r.total_time_s:.2f} s |",
            f"| Aggregate throughput | {r.aggregate_tps:.1f} tok/s |",
            f"| Latency P50 | {r.latency_p50_ms:.1f} ms |",
            f"| Latency P95 | {r.latency_p95_ms:.1f} ms |",
            f"| Latency P99 | {r.latency_p99_ms:.1f} ms |",
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_suite(suite: BenchmarkSuite) -> str:
        lines = [
            f"### Benchmark Suite: {suite.name}",
            f"**Model**: {suite.model}  ",
            f"**Chip**: {suite.chip}  ",
            f"**Timestamp**: {suite.timestamp}",
            "",
        ]
        for i, r in enumerate(suite.results, 1):
            if isinstance(r, BenchmarkResult):
                lines.append(f"#### Test {i}: Single Request")
                lines.append("")
                lines.append(BenchmarkRunner._format_single(r))
            elif isinstance(r, BatchBenchmarkResult):
                lines.append(f"#### Test {i}: Batch")
                lines.append("")
                lines.append(BenchmarkRunner._format_batch(r))
            lines.append("")
        return "\n".join(lines)
