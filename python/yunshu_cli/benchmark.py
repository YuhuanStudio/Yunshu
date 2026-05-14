"""Yunshu CLI — bench subcommand.

Run performance benchmarks: roofline, latency, throughput, memory.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
bench_app = typer.Typer(help="Run benchmarks.", no_args_is_help=True)


@bench_app.command("roofline")
def bench_roofline(
    min_size: int = typer.Option(64, "--min", help="Min matrix dimension."),
    max_size: int = typer.Option(8192, "--max", help="Max matrix dimension."),
    dtype: str = typer.Option("float16", "--dtype", help="Data type (float16, bfloat16, float32)."),
    steps: int = typer.Option(8, "--steps", help="Number of size steps."),
):
    """Apple GPU roofline benchmark — measure GEMM throughput vs matrix size."""
    import mlx.core as mx
    import numpy as np

    dtype_map = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}
    mx_dtype = dtype_map.get(dtype)
    if mx_dtype is None:
        console.print(f"[red]Unknown dtype: {dtype}[/]")
        raise typer.Exit(1)

    console.print(f"[bold]Roofline Benchmark[/] dtype={dtype}")

    sizes = np.logspace(
        np.log10(min_size), np.log10(max_size), steps, dtype=int
    ).tolist()

    results = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        for size in sizes:
            progress.update(progress.add_task(f"M×K×N = {size}×{size}×{size}", total=None))

            # Warmup
            a = mx.random.normal((size, size), dtype=mx_dtype)
            b = mx.random.normal((size, size), dtype=mx_dtype)
            _ = a @ b
            mx.synchronize()

            # Benchmark
            num_iters = max(1, min(100, 2**20 // (size * size)))
            t0 = time.perf_counter()
            for _ in range(num_iters):
                c = a @ b
            mx.synchronize()
            elapsed = time.perf_counter() - t0

            # Compute FLOPs: 2*M*K*N for matmul
            flops = 2.0 * size * size * size * num_iters
            tflops = flops / elapsed / 1e12

            results.append({
                "size": size,
                "flops_per_s": tflops,
                "time_ms": elapsed / num_iters * 1000,
            })

    # Display results
    table = Table(title="GEMM Roofline", show_lines=True)
    table.add_column("M=K=N", justify="right")
    table.add_column("TFLOPS", justify="right", style="bold green")
    table.add_column("Time (ms)", justify="right")

    peak = max(r["flops_per_s"] for r in results)
    for r in results:
        bar_len = int(r["flops_per_s"] / peak * 20)
        bar = "█" * bar_len
        table.add_row(
            str(r["size"]),
            f"{r['flops_per_s']:.2f} {bar}",
            f"{r['time_ms']:.3f}",
        )

    console.print(table)
    console.print(f"\n[dim]Peak: {peak:.2f} TFLOPS ({dtype})[/]")


@bench_app.command("latency")
def bench_latency(
    url: str = typer.Option("http://localhost:8000", "--url", "-u", help="Server URL."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name."),
    prompt_tokens: int = typer.Option(32, "--prompt", help="Prompt length."),
    max_tokens: int = typer.Option(64, "--max-tokens", help="Max output tokens."),
    num_requests: int = typer.Option(10, "--requests", "-n", help="Number of requests."),
):
    """Benchmark inference latency against a running server."""
    import httpx
    import json
    import numpy as np

    prompt = "Hello! " * (prompt_tokens // 7 + 1)

    latencies = []
    ttft_list = []  # Time to first token

    console.print(f"[bold]Latency Benchmark[/] — {num_requests} requests to {url}")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Sending requests...", total=num_requests)

        for i in range(num_requests):
            payload = {
                "model": model or "default",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": False,
            }

            t0 = time.perf_counter()
            try:
                resp = httpx.post(f"{url}/v1/chat/completions", json=payload, timeout=120)
                elapsed = time.perf_counter() - t0

                if resp.status_code == 200:
                    data = resp.json()
                    usage = data.get("usage", {})
                    comp_tokens = usage.get("completion_tokens", max_tokens)
                    latency_per_tok = elapsed / comp_tokens if comp_tokens > 0 else 0
                    latencies.append({
                        "total": elapsed,
                        "per_token": latency_per_tok,
                        "completion_tokens": comp_tokens,
                    })
                else:
                    console.print(f"  [red]Request {i+1} failed: {resp.status_code}[/]")
            except Exception as e:
                logger.debug("Latency benchmark request %d failed", i + 1, exc_info=True)
                console.print(f"  [red]Request {i+1} error: {e}[/]")

            progress.update(task, advance=1)

    if not latencies:
        console.print("[red]No successful requests.[/]")
        return

    totals = [l["total"] for l in latencies]
    per_tok = [l["per_token"] for l in latencies]

    table = Table(title="Latency Results")
    table.add_column("Metric", style="bold")
    table.add_column("P50", justify="right")
    table.add_column("P95", justify="right")
    table.add_column("P99", justify="right")

    for label, data in [("Total Latency (s)", totals), ("Per-token (ms)", [d * 1000 for d in per_tok])]:
        arr = sorted(data)
        p50 = arr[len(arr) // 2]
        p95 = arr[int(len(arr) * 0.95)]
        p99 = arr[min(int(len(arr) * 0.99), len(arr) - 1)]
        table.add_row(label, f"{p50:.3f}", f"{p95:.3f}", f"{p99:.3f}")

    console.print(table)


@bench_app.command("throughput")
def bench_throughput(
    url: str = typer.Option("http://localhost:8000", "--url", "-u", help="Server URL."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name."),
    concurrency: int = typer.Option(4, "--concurrency", "-c", help="Concurrent requests."),
    num_requests: int = typer.Option(20, "--requests", "-n", help="Total requests."),
    max_tokens: int = typer.Option(128, "--max-tokens", help="Max output tokens."),
):
    """Benchmark concurrent throughput against a running server."""
    import httpx
    import asyncio

    console.print(f"[bold]Throughput Benchmark[/] — {num_requests} requests, {concurrency} concurrent")

    async def _run():
        connector = httpx.AsyncHTTPConnector(limit=concurrency)
        async with httpx.AsyncClient(connector=connector, timeout=120) as client:
            semaphore = asyncio.Semaphore(concurrency)
            completed = 0
            total_tokens = 0

            async def _single_req(idx):
                nonlocal completed, total_tokens
                async with semaphore:
                    payload = {
                        "model": model or "default",
                        "messages": [{"role": "user", "content": f"Tell me about topic {idx % 10}."}],
                        "max_tokens": max_tokens,
                        "stream": False,
                    }
                    resp = await client.post(f"{url}/v1/chat/completions", json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        total_tokens += data.get("usage", {}).get("completion_tokens", 0)
                    completed += 1

            t0 = time.perf_counter()
            await asyncio.gather(*[_single_req(i) for i in range(num_requests)])
            elapsed = time.perf_counter() - t0

        return elapsed, total_tokens, completed

    elapsed, total_tokens, completed = asyncio.run(_run())
    throughput = total_tokens / elapsed if elapsed > 0 else 0
    rps = completed / elapsed if elapsed > 0 else 0

    table = Table(title="Throughput Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Total time", f"{elapsed:.2f}s")
    table.add_row("Completed", str(completed))
    table.add_row("Tokens/sec", f"{throughput:.1f}")
    table.add_row("Requests/sec", f"{rps:.2f}")
    table.add_row("Concurrency", str(concurrency))
    console.print(table)


@bench_app.command("memory")
def bench_memory():
    """Show current Apple Silicon memory layout and MLX allocation."""
    import mlx.core as mx

    console.print("[bold]Memory Information[/]\n")

    # System memory
    try:
        import subprocess
        result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        total_mem = int(result.stdout.strip())
    except Exception:
        logger.debug("Failed to read system memory via sysctl", exc_info=True)
        total_mem = 0

    active_mem = mx.get_active_memory()
    peak_mem = mx.get_peak_memory()
    cache_mem = mx.get_cache_memory()

    table = Table(show_lines=True)
    table.add_column("Category", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total Unified Memory", _fmt_bytes(total_mem))
    table.add_row("MLX Active", _fmt_bytes(active_mem))
    table.add_row("MLX Peak", _fmt_bytes(peak_mem))
    table.add_row("MLX Cache", _fmt_bytes(cache_mem))
    if total_mem > 0:
        table.add_row("MLX Active %", f"{active_mem / total_mem * 100:.1f}%")
        table.add_row("Available", _fmt_bytes(total_mem - active_mem))

    console.print(table)

    # GPU info
    console.print("\n[bold]GPU Information[/]")
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.split("\n"):
            line = line.strip()
            if any(k in line for k in ("Chipset", "VRAM", "Metal", "Total", "Cores")):
                console.print(f"  {line}")
    except Exception:
        logger.debug("Failed to query GPU info via system_profiler", exc_info=True)
        console.print("  [dim]Unable to query GPU info[/]")


def _fmt_bytes(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"


@bench_app.command("inference")
def bench_inference(
    model_path: str = typer.Option("models/Qwen3.5-9B-MLX-4bit", "--model", "-m", help="Model path."),
    prompt_tokens: int = typer.Option(32, "--prompt-len", help="Approximate prompt length."),
    max_tokens: int = typer.Option(128, "--max-tokens", help="Max tokens to generate."),
    num_requests: int = typer.Option(5, "--requests", "-n", help="Number of requests."),
    batch_size: int = typer.Option(1, "--batch", "-b", help="Concurrent requests in batch."),
    warmup: int = typer.Option(2, "--warmup", help="Warmup iterations."),
):
    """Direct model inference benchmark (no server needed)."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator, generation_stream
    from mlx_lm.sample_utils import make_sampler
    import numpy as np

    console.print(f"[bold]Inference Benchmark[/]")
    console.print(f"Model: {model_path}")
    console.print(f"Prompt ≈{prompt_tokens} tokens, max_tokens={max_tokens}, batch={batch_size}")

    # Load model
    t0 = time.perf_counter()
    model, tokenizer = load(model_path)
    load_time = time.perf_counter() - t0
    console.print(f"Model loaded in {load_time:.1f}s")

    mem_before = mx.get_active_memory()

    # Prepare prompts
    base_prompt = "Write a short story. " * (prompt_tokens // 6 + 1)
    prompt_tokens_actual = tokenizer.encode(base_prompt, add_special_tokens=False)
    console.print(f"Actual prompt tokens: {len(prompt_tokens_actual)}")

    sampler = make_sampler(temp=0.0)

    # Warmup
    for _ in range(warmup):
        bg = BatchGenerator(
            model, max_tokens=max_tokens, sampler=sampler,
            prefill_batch_size=4, completion_batch_size=32,
            prefill_step_size=2048, stream=generation_stream,
        )
        bg.insert(prompts=[prompt_tokens_actual], max_tokens=[max_tokens], samplers=[sampler])
        bg.next()
        for _ in range(max_tokens + 10):
            gen = bg.next_generated()
            if not gen or any(r.finish_reason for r in gen):
                break
        bg.close()

    mx.synchronize()
    mem_after_warmup = mx.get_active_memory()
    console.print(f"Memory: model={_fmt_bytes(mem_after_warmup)}")

    # Benchmark runs
    ttft_list = []  # Time to first token
    tpot_list = []  # Time per output token
    total_tokens_list = []
    total_time_list = []

    for run in range(num_requests):
        bg = BatchGenerator(
            model, max_tokens=max_tokens, sampler=sampler,
            prefill_batch_size=4, completion_batch_size=32,
            prefill_step_size=2048, stream=generation_stream,
        )

        prompts = [prompt_tokens_actual] * batch_size
        bg.insert(prompts=prompts, max_tokens=[max_tokens] * batch_size, samplers=[sampler] * batch_size)

        # Prefill
        t0 = time.perf_counter()
        prompt_res, _ = bg.next()
        ttft = time.perf_counter() - t0
        ttft_list.append(ttft)

        # Decode
        all_tokens = {i: [] for i in range(batch_size)}
        t_decode_start = time.perf_counter()
        for _ in range(max_tokens + 10):
            gen = bg.next_generated()
            if not gen:
                break
            for r in gen:
                all_tokens[r.uid].append(r.token)
                if r.finish_reason:
                    pass
        t_decode_end = time.perf_counter()

        decode_time = t_decode_end - t_decode_start
        total_tokens = sum(len(v) for v in all_tokens.values())
        tpot = decode_time / total_tokens if total_tokens > 0 else 0
        throughput = total_tokens / decode_time if decode_time > 0 else 0

        tpot_list.append(tpot)
        total_tokens_list.append(total_tokens)
        total_time_list.append(ttft + decode_time)

        console.print(
            f"  Run {run+1}: TTFT={ttft*1000:.0f}ms, "
            f"{total_tokens/batch_size:.0f} tok/req, "
            f"TPOT={tpot*1000:.1f}ms, "
            f"{throughput:.1f} tok/s"
        )
        bg.close()

    # Summary
    mem_final = mx.get_active_memory()

    table = Table(title="Inference Benchmark Results", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Model Load Time", f"{load_time:.1f}s")
    table.add_row("Memory Used", _fmt_bytes(mem_final))
    table.add_row("Prompt Tokens", str(len(prompt_tokens_actual)))
    table.add_row("Batch Size", str(batch_size))
    table.add_row("", "")

    avg_ttft = np.mean(ttft_list)
    p50_ttot = np.percentile(total_time_list, 50)
    avg_tpot = np.mean(tpot_list)
    avg_throughput = np.mean([
        t / max(d, 1e-9)
        for t, d in zip(total_tokens_list, [total_time_list[i] - ttft_list[i] for i in range(num_requests)])
    ])
    aggregate_throughput = sum(total_tokens_list) / sum(
        [total_time_list[i] - ttft_list[i] for i in range(num_requests)]
    )

    table.add_row("TTFT (avg)", f"{avg_ttft*1000:.0f}ms")
    table.add_row("TPOT (avg)", f"{avg_tpot*1000:.2f}ms")
    table.add_row("Throughput (per-req)", f"{1/avg_tpot:.1f} tok/s")
    table.add_row("Aggregate Throughput", f"{aggregate_throughput:.1f} tok/s")
    table.add_row("Total Latency P50", f"{p50_ttot*1000:.0f}ms")

    console.print(table)

    # oMLX comparison
    console.print("\n[bold]oMLX Comparison[/]")
    console.print("  [dim]oMLX reference (Qwen2.5-7B-4bit, M2 Ultra): ~45-55 tok/s single request")
    console.print(f"  Yunshu: {1/avg_tpot:.1f} tok/s single request")
    ratio = (1/avg_tpot) / 50 * 100
    console.print(f"  Performance: {ratio:.0f}% of oMLX reference")
