from __future__ import annotations
"""Yunshu Benchmark API Router — Roofline, Latency, Throughput benchmarks.

Based on oMLX's benchmark.py pattern but adapted for Yunshu's architecture.
Uses mx.core for GPU operations with proper warmup and synchronization.
"""


import asyncio
import logging
import time
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/bench", tags=["benchmark"])

# ── State ──

import threading

_lock = threading.Lock()
_active_benchmark: Optional[str] = None
_benchmark_results: dict = {}


# ── Schemas ──


class RooflineRequest(BaseModel):
    sizes: list[int] = Field(
        default=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
        description="Matrix sizes to benchmark",
    )
    num_warmup: int = Field(default=5, ge=1)
    num_iters: int = Field(default=20, ge=1)
    dtype: str = Field(default="float16")


def _validate_base_url(url: str) -> str:
    """Prevent SSRF — only allow localhost and 127.0.0.1."""
    import re
    host = re.sub(r'^https?://', '', url).split(':')[0].split('/')[0]
    if host not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError(f"base_url must target localhost, got '{host}'")
    return url


class LatencyRequest(BaseModel):
    base_url: str = "http://localhost:8000"
    prompt_lengths: list[int] = Field(default=[32, 128, 512])
    max_tokens_list: list[int] = Field(default=[32, 128])
    num_requests: int = Field(default=3, ge=1)

    validate_url = field_validator("base_url")(_validate_base_url)


class ThroughputRequest(BaseModel):
    base_url: str = "http://localhost:8000"
    concurrency_levels: list[int] = Field(default=[1, 2, 4])
    num_requests: int = Field(default=5, ge=1)
    prompt_tokens: int = Field(default=128, ge=1)
    max_tokens: int = Field(default=128, ge=1)

    validate_url = field_validator("base_url")(_validate_base_url)


# ── Roofline Benchmark ──


def _run_roofline(request: RooflineRequest) -> dict:
    """Run GEMM roofline benchmark using MLX."""
    try:
        import mlx.core as mx
    except ImportError:
        return {"error": "MLX not available"}

    dtype_map = {"float16": mx.float16, "float32": mx.float32, "bfloat16": mx.bfloat16}
    dtype = dtype_map.get(request.dtype, mx.float16)

    sizes = request.sizes
    gflops_list = []
    bandwidth_list = []

    # Warmup
    for _ in range(3):
        a = mx.zeros((256, 256), dtype=dtype)
        b = mx.zeros((256, 256), dtype=dtype)
        mx.eval(a @ b)

    for size in sizes:
        a = mx.random.normal((size, size), dtype=dtype)
        b = mx.random.normal((size, size), dtype=dtype)

        # Warmup for this size
        for _ in range(request.num_warmup):
            c = a @ b
            mx.eval(c)

        # Timed runs
        start = time.perf_counter()
        for _ in range(request.num_iters):
            c = a @ b
            mx.eval(c)
        elapsed = time.perf_counter() - start

        # GFLOPS = 2 * M^3 / time (each multiply-add = 2 FLOPs)
        ops = 2 * (size ** 3) * request.num_iters
        gflops = ops / elapsed / 1e9
        gflops_list.append(round(gflops, 1))

        # Memory bandwidth estimate (read A + B, write C = 3 * M^2 * dtype_bytes)
        bytes_moved = 3 * (size ** 2) * 2 * request.num_iters  # 2 bytes for float16
        bandwidth = bytes_moved / elapsed / 1e9  # GB/s
        bandwidth_list.append(round(bandwidth, 1))

    return {
        "sizes": sizes,
        "gflops": gflops_list,
        "bandwidth_gb_s": bandwidth_list,
        "peak_gflops": max(gflops_list),
        "peak_bandwidth_gb_s": max(bandwidth_list),
        "device": "Apple GPU",
        "dtype": request.dtype,
        "num_iters": request.num_iters,
    }


# ── Latency Benchmark ──


async def _run_latency(request: LatencyRequest) -> dict:
    """Run E2E latency benchmark against the server."""
    import json
    import urllib.request

    results = []

    for prompt_len in request.prompt_lengths:
        for max_tok in request.max_tokens_list:
            latencies = []
            ttfts = []

            # Generate prompt of roughly prompt_len tokens
            prompt = "The quick brown fox jumps over the lazy dog. " * (prompt_len // 10 + 1)

            for _ in range(request.num_requests):
                start = time.perf_counter()
                try:
                    req_data = json.dumps({
                        "model": "default",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tok,
                        "stream": False,
                    }).encode()

                    req = urllib.request.Request(
                        f"{request.base_url}/v1/chat/completions",
                        data=req_data,
                        headers={"Content-Type": "application/json"},
                    )

                    with urllib.request.urlopen(req, timeout=120) as resp:
                        await asyncio.to_thread(resp.read)
                    elapsed = time.perf_counter() - start
                    latencies.append(elapsed)
                except Exception as e:
                    logger.warning(f"Latency benchmark request failed: {e}")
                    continue

            if not latencies:
                continue

            latencies.sort()
            results.append({
                "prompt_length": prompt_len,
                "max_tokens": max_tok,
                "num_requests": len(latencies),
                "p50_ms": round(latencies[len(latencies) // 2] * 1000, 1),
                "p95_ms": round(latencies[int(len(latencies) * 0.95)] * 1000, 1),
                "p99_ms": round(latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)] * 1000, 1),
                "avg_ms": round(sum(latencies) / len(latencies) * 1000, 1),
                "min_ms": round(min(latencies) * 1000, 1),
                "max_ms": round(max(latencies) * 1000, 1),
            })

    return {
        "results": results,
        "base_url": request.base_url,
        "timestamp": time.time(),
    }


# ── Throughput Benchmark ──


async def _run_throughput(request: ThroughputRequest) -> dict:
    """Run concurrent throughput benchmark."""
    import json
    import urllib.request
    import concurrent.futures

    results = []

    prompt = "The quick brown fox jumps over the lazy dog. " * (request.prompt_tokens // 10 + 1)

    for concurrency in request.concurrency_levels:
        total_requests = request.num_requests

        def single_request(_i: int) -> float:
            start = time.perf_counter()
            try:
                req_data = json.dumps({
                    "model": "default",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": request.max_tokens,
                    "stream": False,
                }).encode()

                req = urllib.request.Request(
                    f"{request.base_url}/v1/chat/completions",
                    data=req_data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    resp.read()
            except Exception as e:
                logger.warning(f"Throughput benchmark request failed: {e}")
            return time.perf_counter() - start

        wall_start = time.perf_counter()

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [loop.run_in_executor(executor, single_request, i) for i in range(total_requests)]
            await asyncio.gather(*futures)

        wall_time = time.perf_counter() - wall_start
        requests_per_s = total_requests / wall_time if wall_time > 0 else 0

        # Estimate tokens per second
        est_total_tokens = total_requests * request.max_tokens
        tokens_per_s = est_total_tokens / wall_time if wall_time > 0 else 0

        results.append({
            "concurrency": concurrency,
            "total_requests": total_requests,
            "wall_time_s": round(wall_time, 2),
            "requests_per_s": round(requests_per_s, 2),
            "tokens_per_s": round(tokens_per_s, 1),
            "prompt_tokens": request.prompt_tokens,
            "max_tokens": request.max_tokens,
        })

    return {
        "results": results,
        "base_url": request.base_url,
        "peak_throughput": max(r["tokens_per_s"] for r in results) if results else 0,
        "timestamp": time.time(),
    }


# ── Endpoints ──


@router.post("/roofline")
async def bench_roofline(request: RooflineRequest, bg: BackgroundTasks):
    """Run GEMM roofline benchmark on Apple GPU."""
    with _lock:
        if _active_benchmark:
            raise HTTPException(status_code=409, detail=f"Benchmark '{_active_benchmark}' is already running")
        _active_benchmark = "roofline"

    try:
        result = await asyncio.to_thread(_run_roofline, request)
        _benchmark_results["roofline"] = result
        return result
    except Exception as e:
        logger.error(f"Roofline benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        with _lock:
            _active_benchmark = None


@router.post("/latency")
async def bench_latency(request: LatencyRequest):
    """Run E2E latency benchmark."""
    with _lock:
        if _active_benchmark:
            raise HTTPException(status_code=409, detail=f"Benchmark '{_active_benchmark}' is already running")
        _active_benchmark = "latency"

    try:
        result = await _run_latency(request)
        _benchmark_results["latency"] = result
        return result
    except Exception as e:
        logger.error(f"Latency benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        with _lock:
            _active_benchmark = None


@router.post("/throughput")
async def bench_throughput(request: ThroughputRequest):
    """Run concurrent throughput benchmark."""
    with _lock:
        if _active_benchmark:
            raise HTTPException(status_code=409, detail=f"Benchmark '{_active_benchmark}' is already running")
        _active_benchmark = "throughput"
    try:
        result = await _run_throughput(request)
        _benchmark_results["throughput"] = result
        return result
    except Exception as e:
        logger.error(f"Throughput benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        with _lock:
            _active_benchmark = None


@router.get("/status")
async def bench_status():
    """Get current benchmark status."""
    with _lock:
        active = _active_benchmark
    return {
        "active": active,
        "results_available": list(_benchmark_results.keys()),
    }


# ── Model Benchmark (engine-level) ──


class ModelBenchRequest(BaseModel):
    prompt_lengths: list[int] = Field(default=[32, 128, 512, 1024])
    max_tokens_list: list[int] = Field(default=[32, 128])
    num_requests: int = Field(default=3, ge=1)
    stream: bool = False


@router.post("/model")
async def bench_model(request: ModelBenchRequest):
    """Run model-level benchmark using BatchedEngine directly (no HTTP overhead).

    This uses the engine's BenchmarkRunner for in-process benchmarking,
    measuring pure GPU inference speed without network latency.
    """
    with _lock:
        if _active_benchmark:
            raise HTTPException(status_code=409, detail=f"Benchmark '{_active_benchmark}' is already running")
        _active_benchmark = "model"

    try:
        from yunshu_engine.model_manager import ModelManager
        from yunshu_engine.benchmark import BenchmarkRunner

        mgr = ModelManager()
        engine = mgr.get_engine()
        if engine is None:
            raise HTTPException(status_code=503, detail="No model loaded")

        runner = BenchmarkRunner(engine)
        suite = await runner.run_suite(
            prompt_lengths=request.prompt_lengths,
            max_tokens_list=request.max_tokens_list,
            num_requests=request.num_requests,
            stream=request.stream,
        )
        _benchmark_results["model"] = suite.to_dict()
        return suite.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        with _lock:
            _active_benchmark = None


@router.post("/batch")
async def bench_batch(concurrency: int = 4, num_requests: int = 8, prompt_tokens: int = 128, max_tokens: int = 32):
    """Run batch benchmark measuring concurrent request throughput."""
    with _lock:
        if _active_benchmark:
            raise HTTPException(status_code=409, detail=f"Benchmark '{_active_benchmark}' is already running")
        _active_benchmark = "batch"

    try:
        from yunshu_engine.model_manager import ModelManager
        from yunshu_engine.benchmark import BenchmarkRunner

        mgr = ModelManager()
        engine = mgr.get_engine()
        if engine is None:
            raise HTTPException(status_code=503, detail="No model loaded")

        runner = BenchmarkRunner(engine)
        result = await runner.bench_batch(
            concurrency=concurrency,
            num_requests=num_requests,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
        )
        _benchmark_results["batch"] = result.to_dict()
        return result.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        with _lock:
            _active_benchmark = None


# ── Roofline Model Analysis ──


class RooflineModelRequest(BaseModel):
    chip: str = Field(default="auto", description="Chip name (e.g., M4_Max) or 'auto' to detect")
    gemm_sizes: list[list[int]] = Field(
        default=[[1, 4096, 4096], [32, 4096, 4096], [1, 4096, 11008]],
        description="GEMM sizes [M, N, K] to analyze",
    )


@router.post("/roofline-model")
async def bench_roofline_model(request: RooflineModelRequest):
    """Analyze compute vs memory boundedness using the roofline model.

    Uses Yunshu's RooflineModel engine module for analytical throughput
    prediction based on Apple Silicon chip parameters.
    """
    with _lock:
        if _active_benchmark:
            raise HTTPException(status_code=409, detail=f"Benchmark '{_active_benchmark}' is already running")
        _active_benchmark = "roofline-model"

    try:
        from yunshu_engine.roofline import RooflineModel

        chip = request.chip
        if chip == "auto":
            from yunshu_engine.utils.hardware import get_chip_name
            chip = get_chip_name()

        rm = RooflineModel(chip)
        results = []
        for m, n, k in request.gemm_sizes:
            report = rm.compute_gemm_roofline(M=m, N=n, K=k)
            results.append({
                "gemm_size": [m, n, k],
                "arithmetic_intensity": round(report.arithmetic_intensity, 2),
                "compute_bound": report.is_compute_bound,
                "predicted_throughput_gbps": round(report.predicted_throughput, 2),
                "roofline_tflops": round(report.roofline_tflops, 2),
            })

        _benchmark_results["roofline-model"] = {
            "chip": rm.chip_name,
            "bandwidth_gbps": rm.bandwidth_gbps,
            "compute_tflops_fp16": rm.compute_tflops_fp16,
            "analyses": results,
        }
        return _benchmark_results["roofline-model"]
    except Exception as e:
        logger.error(f"Roofline model analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        with _lock:
            _active_benchmark = None


# ── BFCL Function Calling Evaluation ──


class BFCLEvalRequest(BaseModel):
    categories: list[str] = Field(
        default=["simple", "parallel", "multiple", "parallel_multiple"],
        description="BFCL categories to evaluate",
    )
    max_samples: int = Field(default=10, ge=0, description="Max test cases per category (0=all)")


@router.post("/bfcl-eval")
async def bench_bfcl_eval(request: BFCLEvalRequest):
    """Run BFCL function calling evaluation against the loaded model.

    Evaluates tool/function calling accuracy using the BFCLEvaluator
    from yunshu_engine. Requires a loaded model with tool calling support.
    """
    with _lock:
        if _active_benchmark:
            raise HTTPException(status_code=409, detail=f"Benchmark '{_active_benchmark}' is already running")
        _active_benchmark = "bfcl-eval"

    try:
        from yunshu_engine.model_manager import ModelManager
        from yunshu_engine.bfcl_eval import BFCLEvaluator, BFCLEvalConfig

        mgr = ModelManager()
        engine = mgr.get_engine()
        if engine is None:
            raise HTTPException(status_code=503, detail="No model loaded")

        config = BFCLEvalConfig(
            model_name=engine.model_name or "unknown",
            test_categories=[c for c in request.categories if c in ("simple", "parallel", "multiple", "parallel_multiple")],
            max_samples=request.max_samples,
        )
        evaluator = BFCLEvaluator(config, engine=engine)

        # Run in thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, evaluator.run_all)

        result_dicts = []
        for r in results:
            result_dicts.append({
                "category": r.category,
                "total": r.total,
                "correct": r.correct,
                "accuracy": round(r.accuracy, 3),
                "avg_latency_ms": round(r.avg_latency_ms, 1),
                "errors": r.errors[:5],  # Limit error details
            })

        _benchmark_results["bfcl-eval"] = {
            "model": config.model_name,
            "categories": result_dicts,
            "timestamp": time.time(),
        }
        return _benchmark_results["bfcl-eval"]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"BFCL evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        with _lock:
            _active_benchmark = None