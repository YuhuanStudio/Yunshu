from __future__ import annotations

"""Yunshu Benchmark API Router — Roofline, Latency, Throughput benchmarks.

Uses mx.core for GPU operations with proper warmup and synchronization.

Security: All benchmark endpoints require authentication (deny-by-default).
Benchmarks are resource-intensive and can cause DoS if left unprotected.
Set YUNSHU_AUTH_TOKEN or YUNSHU_AUTH_DISABLED=true for access.
"""


import asyncio
import logging
import os
import time
import urllib.request

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/bench", tags=["benchmark"])


def _check_permission(request: Request) -> None:
    """Check auth on benchmark endpoints (deny-by-default).

    Benchmarks are resource-intensive GPU operations that can degrade
    inference performance for all users.

    Security: verifies the request actually presents a valid token,
    not merely that a token is configured.
    """
    if os.environ.get("YUNSHU_AUTH_DISABLED", "").lower() in ("true", "1", "yes"):
        return
    # RBAC key set by TenantAuthMiddleware (ys_-prefixed API keys)
    rbac_key = getattr(request.state, "rbac_key", None)
    if rbac_key is not None:
        if not rbac_key.has_permission("can_benchmark"):
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions for benchmark endpoints",
            )
        return
    # Tenant set by TenantAuthMiddleware (legacy tenant auth)
    tenant = getattr(request.state, "tenant", None)
    if tenant is not None:
        # benchmarks are GPU-heavy, all-users-degrading admin-class ops
        # (can_benchmark — a permission USER RBAC keys lack). The blanket tenant-allow
        # let any legacy tenant trigger them (privesc keystone). Require admin role.
        _role = str(getattr(request.state, "role", "") or "")
        if _role.lower() in ("admin", "system", "owner") or _role.upper().endswith(
            "ADMIN"
        ):
            return
        raise HTTPException(
            status_code=403, detail="Insufficient permissions for benchmark endpoints"
        )
    # Static token auth — must verify the request actually provides it
    auth_token = os.environ.get("YUNSHU_AUTH_TOKEN")
    if auth_token:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            import hmac

            if hmac.compare_digest(auth[7:], auth_token):
                return
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # No auth configured — deny
    raise HTTPException(
        status_code=401,
        detail="Benchmark requires authentication. Set YUNSHU_AUTH_TOKEN or YUNSHU_AUTH_DISABLED=true.",
    )


# ── State ──

import threading

_lock = threading.Lock()
_active_benchmark: str | None = None
_benchmark_results: dict = {}


# ── Schemas ──


class RooflineRequest(BaseModel):
    sizes: list[int] = Field(
        default=[64, 128, 256, 512, 1024, 2048, 4096, 8192],
        description="Matrix sizes to benchmark",
        max_length=64,
    )
    num_warmup: int = Field(default=5, ge=1, le=100)
    num_iters: int = Field(default=20, ge=1, le=1000)
    dtype: str = Field(default="float16")

    @field_validator("dtype")
    @classmethod
    def _validate_dtype(cls, v):
        if v not in ("float16", "float32", "bfloat16"):
            raise ValueError(f"dtype must be float16, float32, or bfloat16, got '{v}'")
        return v


def _validate_base_url(url: str) -> str:
    """Prevent SSRF — only allow http/https to localhost addresses."""
    import re

    # Must start with http:// or https://
    if not re.match(r"^https?://", url, re.IGNORECASE):
        raise ValueError(
            f"base_url must use http:// or https:// scheme, got '{url[:50]}'"
        )
    # Strip scheme and extract host
    host = (
        re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
        .split(":")[0]
        .split("/")[0]
        .split("?")[0]
    )
    # Normalize: lowercase, strip brackets from IPv6
    host_lower = host.lower().strip("[]")
    # Allow only loopback addresses
    _ALLOWED = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "[::1]"}
    if host not in _ALLOWED and host_lower not in _ALLOWED:
        raise ValueError(f"base_url must target localhost, got '{host}'")
    return url


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """block SSRF-via-redirect. _validate_base_url only checks the
    initial host, but urllib follows 3xx by default — a localhost service could 302 the
    POST to http://169.254.169.254/… (cloud metadata) or any internal host. Returning
    None here makes urllib NOT follow the redirect and surface the 3xx response instead,
    so the benchmark just counts it as a failed request rather than fetching the target.
    """

    def redirect_request(self, *args, **kwargs):  # noqa: D102
        return None


def _no_redirect_open(req, timeout):
    """Open a urllib request with redirects disabled (SSRF defense)."""
    opener = urllib.request.build_opener(_NoRedirect)
    return opener.open(req, timeout=timeout)


def _validate_pos_int_list(values: list[int], name: str, max_val: int) -> list[int]:
    """Ensure each element is in [1, max_val]. Lifts validation off the engine."""
    for v in values:
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f"{name}: items must be integers")
        if v < 1:
            raise ValueError(f"{name}: items must be >= 1, got {v}")
        if v > max_val:
            raise ValueError(f"{name}: items must be <= {max_val}, got {v}")
    return values


class LatencyRequest(BaseModel):
    base_url: str = "http://localhost:8000"
    model: str = "default"
    prompt_lengths: list[int] = Field(default=[32, 128, 512], max_length=64)
    max_tokens_list: list[int] = Field(default=[32, 128], max_length=32)
    num_requests: int = Field(default=3, ge=1, le=1000)

    validate_url = field_validator("base_url")(_validate_base_url)

    @field_validator("prompt_lengths")
    @classmethod
    def _check_prompt_lengths(cls, v: list[int]) -> list[int]:
        return _validate_pos_int_list(v, "prompt_lengths", 32768)

    @field_validator("max_tokens_list")
    @classmethod
    def _check_max_tokens_list(cls, v: list[int]) -> list[int]:
        return _validate_pos_int_list(v, "max_tokens_list", 131072)


class ThroughputRequest(BaseModel):
    base_url: str = "http://localhost:8000"
    model: str = "default"
    concurrency_levels: list[int] = Field(default=[1, 2, 4], max_length=32)
    num_requests: int = Field(default=5, ge=1, le=1000)
    prompt_tokens: int = Field(default=128, ge=1, le=32768)
    max_tokens: int = Field(default=128, ge=1, le=131072)

    validate_url = field_validator("base_url")(_validate_base_url)

    @field_validator("concurrency_levels")
    @classmethod
    def _check_concurrency_levels(cls, v: list[int]) -> list[int]:
        return _validate_pos_int_list(v, "concurrency_levels", 1024)


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
        ops = 2 * (size**3) * request.num_iters
        gflops = ops / elapsed / 1e9
        gflops_list.append(round(gflops, 1))

        # Memory bandwidth estimate (read A + B, write C = 3 * M^2 * dtype_bytes)
        bytes_moved = 3 * (size**2) * 2 * request.num_iters  # 2 bytes for float16
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

            # Generate prompt of roughly prompt_len tokens
            prompt = "The quick brown fox jumps over the lazy dog. " * (
                prompt_len // 10 + 1
            )

            for _ in range(request.num_requests):
                start = time.perf_counter()
                try:
                    req_data = json.dumps(
                        {
                            "model": request.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_tok,
                            "stream": False,
                        }
                    ).encode()

                    req = urllib.request.Request(
                        f"{request.base_url}/v1/chat/completions",
                        data=req_data,
                        headers={"Content-Type": "application/json"},
                    )

                    def _blocking_request():
                        with _no_redirect_open(req, timeout=120) as resp:
                            return resp.read()

                    await asyncio.to_thread(_blocking_request)
                    elapsed = time.perf_counter() - start
                    latencies.append(elapsed)
                except Exception as e:
                    logger.warning(f"Latency benchmark request failed: {e}")
                    continue

            if not latencies:
                continue

            latencies.sort()
            results.append(
                {
                    "prompt_length": prompt_len,
                    "max_tokens": max_tok,
                    "num_requests": len(latencies),
                    "p50_ms": round(latencies[len(latencies) // 2] * 1000, 1),
                    "p95_ms": round(latencies[int(len(latencies) * 0.95)] * 1000, 1),
                    "p99_ms": round(
                        latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)]
                        * 1000,
                        1,
                    ),
                    "avg_ms": round(sum(latencies) / len(latencies) * 1000, 1),
                    "min_ms": round(min(latencies) * 1000, 1),
                    "max_ms": round(max(latencies) * 1000, 1),
                }
            )

    return {
        "results": results,
        "base_url": request.base_url,
        "timestamp": time.time(),
    }


# ── Throughput Benchmark ──


async def _run_throughput(request: ThroughputRequest) -> dict:
    """Run concurrent throughput benchmark."""
    import concurrent.futures
    import json
    import urllib.request

    results = []

    prompt = "The quick brown fox jumps over the lazy dog. " * (
        request.prompt_tokens // 10 + 1
    )

    for concurrency in request.concurrency_levels:
        total_requests = request.num_requests

        def single_request(_i: int) -> float:
            start = time.perf_counter()
            try:
                req_data = json.dumps(
                    {
                        "model": request.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": request.max_tokens,
                        "stream": False,
                    }
                ).encode()

                req = urllib.request.Request(
                    f"{request.base_url}/v1/chat/completions",
                    data=req_data,
                    headers={"Content-Type": "application/json"},
                )
                with _no_redirect_open(req, timeout=120) as resp:
                    resp.read()
            except Exception as e:
                logger.warning(f"Throughput benchmark request failed: {e}")
            return time.perf_counter() - start

        wall_start = time.perf_counter()

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                loop.run_in_executor(executor, single_request, i)
                for i in range(total_requests)
            ]
            await asyncio.gather(*futures)

        wall_time = time.perf_counter() - wall_start
        requests_per_s = total_requests / wall_time if wall_time > 0 else 0

        # Estimate tokens per second
        est_total_tokens = total_requests * request.max_tokens
        tokens_per_s = est_total_tokens / wall_time if wall_time > 0 else 0

        results.append(
            {
                "concurrency": concurrency,
                "total_requests": total_requests,
                "wall_time_s": round(wall_time, 2),
                "requests_per_s": round(requests_per_s, 2),
                "tokens_per_s": round(tokens_per_s, 1),
                "prompt_tokens": request.prompt_tokens,
                "max_tokens": request.max_tokens,
            }
        )

    return {
        "results": results,
        "base_url": request.base_url,
        "peak_throughput": max(r["tokens_per_s"] for r in results) if results else 0,
        "timestamp": time.time(),
    }


# ── Endpoints ──


@router.post("/roofline")
async def bench_roofline(req: Request, bench_req: RooflineRequest):
    """Run GEMM roofline benchmark on Apple GPU."""
    _check_permission(req)
    global _active_benchmark
    with _lock:
        if _active_benchmark:
            raise HTTPException(
                status_code=409,
                detail=f"Benchmark '{_active_benchmark}' is already running",
            )
        _active_benchmark = "roofline"

    try:
        result = await asyncio.to_thread(_run_roofline, bench_req)
        _benchmark_results["roofline"] = result
        return result
    except Exception as e:
        logger.error(f"Roofline benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from None
    finally:
        with _lock:
            _active_benchmark = None


@router.post("/latency")
async def bench_latency(req: Request, bench_req: LatencyRequest):
    """Run E2E latency benchmark."""
    _check_permission(req)
    global _active_benchmark
    with _lock:
        if _active_benchmark:
            raise HTTPException(
                status_code=409,
                detail=f"Benchmark '{_active_benchmark}' is already running",
            )
        _active_benchmark = "latency"

    try:
        result = await _run_latency(bench_req)
        _benchmark_results["latency"] = result
        return result
    except Exception as e:
        logger.error(f"Latency benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from None
    finally:
        with _lock:
            _active_benchmark = None


@router.post("/throughput")
async def bench_throughput(req: Request, bench_req: ThroughputRequest):
    """Run concurrent throughput benchmark."""
    _check_permission(req)
    global _active_benchmark
    with _lock:
        if _active_benchmark:
            raise HTTPException(
                status_code=409,
                detail=f"Benchmark '{_active_benchmark}' is already running",
            )
        _active_benchmark = "throughput"
    try:
        result = await _run_throughput(bench_req)
        _benchmark_results["throughput"] = result
        return result
    except Exception as e:
        logger.error(f"Throughput benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from None
    finally:
        with _lock:
            _active_benchmark = None


@router.get("/status")
async def bench_status(req: Request):
    """Get current benchmark status."""
    _check_permission(req)
    with _lock:
        active = _active_benchmark
    return {
        "active": active,
        "results_available": list(_benchmark_results.keys()),
    }


# ── Model Benchmark (engine-level) ──


class ModelBenchRequest(BaseModel):
    model: str | None = Field(
        default=None,
        description="Model id to benchmark. Defaults to the first loaded model.",
    )
    prompt_lengths: list[int] = Field(default=[32, 128, 512, 1024], max_length=64)
    max_tokens_list: list[int] = Field(default=[32, 128], max_length=32)
    num_requests: int = Field(default=3, ge=1, le=1000)
    stream: bool = False


def _resolve_loaded_engine(model_id: str | None):
    """Pick a loaded engine from the gateway, falling back across modes.

    Returns (engine, resolved_model_id) or (None, None).
    """
    from yunshu_gateway.engine import get_engine, get_model_manager

    # Single-engine mode
    engine = get_engine()
    if engine is not None:
        return engine, getattr(engine, "model_name", None)

    # Multi-model mode: use given model_id or pick the first loaded entry
    manager = get_model_manager()
    if manager is None:
        return None, None

    if model_id:
        entry = manager.get_entry(model_id) if hasattr(manager, "get_entry") else None
        if entry is None:
            for e in manager.list_entries():
                if e.model_id == model_id:
                    entry = e
                    break
        if (
            entry is not None
            and entry.is_loaded
            and getattr(entry, "engine", None) is not None
        ):
            return entry.engine, entry.model_id
        return None, None

    for entry in manager.list_entries():
        if entry.is_loaded and getattr(entry, "engine", None) is not None:
            return entry.engine, entry.model_id
    return None, None


@router.post("/model")
async def bench_model(req: Request, bench_req: ModelBenchRequest):
    """Run model-level benchmark using BatchedEngine directly (no HTTP overhead).

    This uses the engine's BenchmarkRunner for in-process benchmarking,
    measuring pure GPU inference speed without network latency.
    """
    _check_permission(req)
    global _active_benchmark
    with _lock:
        if _active_benchmark:
            raise HTTPException(
                status_code=409,
                detail=f"Benchmark '{_active_benchmark}' is already running",
            )
        _active_benchmark = "model"

    try:
        from yunshu_engine.benchmark import BenchmarkRunner

        engine, resolved_id = _resolve_loaded_engine(bench_req.model)
        if engine is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model '{bench_req.model}' not loaded"
                    if bench_req.model
                    else "No model loaded"
                ),
            )

        runner = BenchmarkRunner(engine)
        results = []
        for plen in bench_req.prompt_lengths:
            for mtok in bench_req.max_tokens_list:
                for _ in range(bench_req.num_requests):
                    r = await runner.bench_single_request(
                        prompt_tokens=plen,
                        max_tokens=mtok,
                    )
                    rd = r.to_dict() if hasattr(r, "to_dict") else r.__dict__
                    rd.update({"prompt_length": plen, "max_tokens": mtok})
                    results.append(rd)

        out = {
            "model": resolved_id,
            "results": results,
            "timestamp": time.time(),
        }
        _benchmark_results["model"] = out
        return out
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from None
    finally:
        with _lock:
            _active_benchmark = None


@router.post("/batch")
async def bench_batch(
    req: Request,
    concurrency: int = Query(default=4, ge=1, le=64),
    num_requests: int = Query(default=8, ge=1, le=1024),
    prompt_tokens: int = Query(default=128, ge=1, le=32768),
    max_tokens: int = Query(default=32, ge=1, le=131072),
):
    """Run batch benchmark measuring concurrent request throughput."""
    _check_permission(req)
    global _active_benchmark
    with _lock:
        if _active_benchmark:
            raise HTTPException(
                status_code=409,
                detail=f"Benchmark '{_active_benchmark}' is already running",
            )
        _active_benchmark = "batch"

    try:
        from yunshu_engine.benchmark import BenchmarkRunner

        engine, _ = _resolve_loaded_engine(None)
        if engine is None:
            raise HTTPException(status_code=503, detail="No model loaded")

        runner = BenchmarkRunner(engine)
        result = await runner.bench_batch(
            concurrency=concurrency,
            prompts=num_requests,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
        )
        _benchmark_results["batch"] = result.to_dict()
        return result.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch benchmark failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from None
    finally:
        with _lock:
            _active_benchmark = None


# ── Roofline Model Analysis ──


class RooflineModelRequest(BaseModel):
    chip: str = Field(
        default="auto", description="Chip name (e.g., M4_Max) or 'auto' to detect"
    )
    gemm_sizes: list[list[int]] = Field(
        default=[[1, 4096, 4096], [32, 4096, 4096], [1, 4096, 11008]],
        description="GEMM sizes [M, N, K] to analyze",
        max_length=128,
    )

    @model_validator(mode="after")
    def _check_gemm_sizes(self):
        # each GEMM size must be exactly [M, N, K]. Without this, a sublist of
        # length != 3 raised ValueError at the `for m, n, k in ...` unpack → caught by the
        # generic handler → 500 instead of a clean 422 for malformed input.
        for i, g in enumerate(self.gemm_sizes):
            if len(g) != 3:
                raise ValueError(
                    f"gemm_sizes[{i}] must be exactly [M, N, K] (got {len(g)} values)"
                )
            if any(v <= 0 for v in g):
                raise ValueError(f"gemm_sizes[{i}] values must be positive")
        return self


@router.post("/roofline-model")
async def bench_roofline_model(req: Request, bench_req: RooflineModelRequest):
    """Analyze compute vs memory boundedness using the roofline model.

    Uses Yunshu's RooflineModel engine module for analytical throughput
    prediction based on Apple Silicon chip parameters.
    """
    _check_permission(req)
    global _active_benchmark
    with _lock:
        if _active_benchmark:
            raise HTTPException(
                status_code=409,
                detail=f"Benchmark '{_active_benchmark}' is already running",
            )
        _active_benchmark = "roofline-model"

    try:
        from yunshu_engine.roofline import RooflineModel

        chip = bench_req.chip
        if chip == "auto":
            from yunshu_engine.utils.hardware import get_chip_name

            chip = get_chip_name()

        rm = RooflineModel(chip)
        results = []
        for m, n, k in bench_req.gemm_sizes:
            report = rm.compute_gemm_roofline(M=m, N=n, K=k)
            results.append(
                {
                    "gemm_size": [m, n, k],
                    "arithmetic_intensity": round(report.operational_intensity, 2),
                    "compute_bound": report.bound == "compute",
                    "bound": report.bound,
                    "predicted_gflops": round(report.predicted_gflops, 2),
                    "peak_gflops": round(report.peak_gflops, 2),
                    "flops": report.flops,
                    "bytes_accessed": report.bytes_accessed,
                }
            )

        _benchmark_results["roofline-model"] = {
            "chip": rm.chip_key,
            "bandwidth_gbps": rm.bandwidth_gbs,
            "compute_tflops_fp16": rm.compute_tflops,
            "analyses": results,
        }
        return _benchmark_results["roofline-model"]
    except Exception as e:
        logger.error(f"Roofline model analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from None
    finally:
        with _lock:
            _active_benchmark = None


# ── BFCL Function Calling Evaluation ──


class BFCLEvalRequest(BaseModel):
    categories: list[str] = Field(
        default=["simple", "parallel", "multiple", "parallel_multiple"],
        description="BFCL categories to evaluate",
    )
    # cap max_samples (0=all). The eval is serialized by the benchmark lock and
    # gated behind admin-class can_benchmark, but an explicit upper bound keeps a single call
    # from running unboundedly long on a huge category set.
    max_samples: int = Field(
        default=10, ge=0, le=100000, description="Max test cases per category (0=all)"
    )


@router.post("/bfcl-eval")
async def bench_bfcl_eval(req: Request, bench_req: BFCLEvalRequest):
    """Run BFCL function calling evaluation against the loaded model.

    Evaluates tool/function calling accuracy using the BFCLEvaluator
    from yunshu_engine. Requires a loaded model with tool calling support.
    """
    _check_permission(req)
    global _active_benchmark
    with _lock:
        if _active_benchmark:
            raise HTTPException(
                status_code=409,
                detail=f"Benchmark '{_active_benchmark}' is already running",
            )
        _active_benchmark = "bfcl-eval"

    try:
        from yunshu_engine.bfcl_eval import BFCLEvalConfig, BFCLEvaluator

        engine, _ = _resolve_loaded_engine(None)
        if engine is None:
            raise HTTPException(status_code=503, detail="No model loaded")

        config = BFCLEvalConfig(
            model_name=engine.model_name or "unknown",
            test_categories=[
                c
                for c in bench_req.categories
                if c in ("simple", "parallel", "multiple", "parallel_multiple")
            ],
            max_samples=bench_req.max_samples,
        )
        evaluator = BFCLEvaluator(config, engine=engine)

        # Run in thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, evaluator.run_all)

        result_dicts = []
        for r in results:
            result_dicts.append(
                {
                    "category": r.category,
                    "total": r.total,
                    "correct": r.correct,
                    "accuracy": round(r.accuracy, 3),
                    "avg_latency_ms": round(r.avg_latency_ms, 1),
                    "errors": r.errors[:5],  # Limit error details
                }
            )

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
        raise HTTPException(status_code=500, detail=str(e)) from None
    finally:
        with _lock:
            _active_benchmark = None
