"""Yunshu EngineCore — continuous batching orchestrator (oMLX pattern).

Studied from oMLX's engine_core.py, written from scratch:
- EngineCore orchestrates Scheduler + RequestOutputCollector + RequestStreamState
- All scheduler.step() calls run on the MLX executor thread (serialized GPU work)
- Output distribution happens on the event loop (low latency)
- stream_outputs() uses get_nowait() or await get() pattern from vLLM
- generate() waits on asyncio.Event then drains collector
- Request lifecycle: add → schedule → step → distribute → collect

Architecture:
  Gateway → EngineCore.add_request()
    → Scheduler.add_request() on MLX executor
  EngineCore._engine_loop()
    → Scheduler.step() on MLX executor
    → distribute outputs to per-request RequestOutputCollector
  Gateway → EngineCore.stream_outputs(request_id)
    → collector.get_nowait() or await collector.get()
"""
from __future__ import annotations

import asyncio
import gc
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class EngineCoreConfig:
    """EngineCore tuning parameters (maps to oMLX's EngineConfig)."""
    step_interval: float = 0.001
    stream_interval: int = 1
    completion_batch_size: int = 32
    prefill_batch_size: int = 8
    prefill_step_size: int = 2048
    max_kv_size: int | None = None
    deferred_clear_delay: int = 8
    cache_cleanup_interval: int = 512
    # Paged KV cache (oMLX PagedAttention pattern)
    enable_paged_kv: bool = True  # C11: enabled by default for radix tree + memory efficiency
    kv_block_size: int = 64
    kv_cache_ratio: float = 0.25  # fraction of UMA for KV cache
    # Model architecture (for KV cache memory budget)
    num_layers: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    kv_num_blocks: int = 0  # pre-computed block count (0 = auto-compute)


class EngineCore:
    """MLX-native continuous batching engine core (oMLX EngineCore pattern).

    Orchestrates:
    - Scheduler: manages BatchGenerator + request lifecycle
    - RequestOutputCollector: per-request output buffer with smart aggregation
    - RequestStreamState: stream_interval batching
    - asyncio.Event: per-request completion signaling

    Threading: scheduler.step() runs on MLX executor (single GPU thread).
    Everything else runs on the asyncio event loop.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: EngineCoreConfig | None = None,
        executor: Any = None,
    ) -> None:
        from .scheduler import Scheduler, SchedulerConfig
        from .mlx_executor import get_mlx_executor

        self.config = config or EngineCoreConfig()
        self._executor = executor or get_mlx_executor()

        scheduler_config = SchedulerConfig(
            completion_batch_size=self.config.completion_batch_size,
            prefill_batch_size=self.config.prefill_batch_size,
            prefill_step_size=self.config.prefill_step_size,
            max_kv_size=self.config.max_kv_size,
            deferred_clear_delay=self.config.deferred_clear_delay,
            cache_cleanup_interval=self.config.cache_cleanup_interval,
        )

        if self.config.enable_paged_kv:
            # Paged KV requires model architecture info
            has_arch = (
                self.config.num_layers > 0
                and self.config.num_kv_heads > 0
                and self.config.head_dim > 0
            )
            if not has_arch:
                logger.info(
                    "Paged KV requested but model arch info missing "
                    f"(layers={self.config.num_layers}, kv_heads={self.config.num_kv_heads}, "
                    f"head_dim={self.config.head_dim}). Falling back to non-paged scheduler."
                )
                self.config.enable_paged_kv = False
            else:
                try:
                    from .paged_scheduler import PagedScheduler
                    from yunshu_kv.manager import KVCacheManager, KVCacheConfig, compute_num_blocks

                    kv_config = KVCacheConfig(
                        block_size=self.config.kv_block_size,
                        num_layers=self.config.num_layers,
                        num_kv_heads=self.config.num_kv_heads,
                        head_dim=self.config.head_dim,
                    )

                    # Determine block count
                    num_blocks = self.config.kv_num_blocks
                    if num_blocks <= 0:
                        # Auto-compute from UMA budget
                        try:
                            from .memory_monitor import MemoryMonitor
                            monitor = MemoryMonitor()
                            uma_bytes = monitor.get_stats().get("total_memory_bytes", 0)
                            if uma_bytes > 0:
                                num_blocks = compute_num_blocks(kv_config, uma_bytes, 0)
                            else:
                                num_blocks = 1024  # safe default
                        except Exception:
                            num_blocks = 1024  # safe default

                    kv_manager = KVCacheManager(kv_config, num_blocks=num_blocks)

                    # Wrap with TieredKVCacheManager if SSD cache is configured
                    ssd_dir = os.environ.get("YUNSHU_SSD_CACHE_DIR")
                    if ssd_dir:
                        from yunshu_kv.tiered import TieredKVCacheManager, SSDCacheStore
                        ssd_store = SSDCacheStore(
                            cache_dir=ssd_dir,
                            block_size=self.config.kv_block_size,
                        )
                        kv_manager = TieredKVCacheManager(
                            hot_manager=kv_manager,
                            ssd_store=ssd_store,
                            warm_tier=kv_manager._warm_tier,
                        )
                        logger.info(f"TieredKVCacheManager enabled: SSD dir={ssd_dir}")

                    self.scheduler = PagedScheduler(model, tokenizer, scheduler_config, kv_manager)
                    self._kv_manager = kv_manager
                    logger.info(
                        f"PagedScheduler enabled: block_size={self.config.kv_block_size}, "
                        f"num_blocks={num_blocks}"
                    )
                except Exception as e:
                    logger.warning(f"Paged KV init failed ({e}), falling back to non-paged")
                    self.config.enable_paged_kv = False

        if not self.config.enable_paged_kv:
            self.scheduler = Scheduler(model, tokenizer, scheduler_config)
            self._kv_manager = None

        # Wire ServerMetrics + PrefillProgressTracker into scheduler
        try:
            from .server_metrics import get_server_metrics
            self.scheduler.set_server_metrics(get_server_metrics())
        except Exception:
            logger.debug("server_metrics unavailable", exc_info=True)
        try:
            from .prefill_progress import get_prefill_tracker
            self.scheduler.set_prefill_tracker(get_prefill_tracker())
        except Exception:
            logger.debug("prefill_progress tracker unavailable", exc_info=True)

        # Per-request output management
        self._output_collectors: dict[str, Any] = {}
        self._stream_states: dict[str, Any] = {}
        self._finished_events: dict[str, asyncio.Event] = {}

        # Tokenizer reference (for chat template, encoding)
        self._tokenizer = tokenizer
        self._model = model

        # Memory guard (created after model info is available)
        self._memory_guard: Any = None

        # Lifecycle
        self._running = False
        self._loop_task: asyncio.Task | None = None
        self._start_time: float | None = None

        # Stats
        self._num_requests_processed: int = 0

    def set_prefix_cache(self, cache: Any) -> None:
        """Set KV prefix cache for batch-path insert_segments (C16)."""
        self.scheduler.set_prefix_cache(cache)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def has_active_requests(self) -> bool:
        return self.scheduler.has_requests()

    def setup_memory_guard(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_attention_heads: int | None = None,
        max_concurrent_requests: int = 64,
    ) -> None:
        """Create and configure the MemoryGuard after model info is available."""
        from .memory_monitor import MemoryMonitor
        from .memory_guard import MemoryGuard

        monitor = MemoryMonitor()
        monitor.set_model_info(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
        )
        monitor.set_baseline_memory()

        self._memory_guard = MemoryGuard(
            memory_monitor=monitor,
            max_concurrent_requests=max_concurrent_requests,
        )
        logger.info(
            f"MemoryGuard configured: {num_layers}L, {num_kv_heads} KV heads, "
            f"{head_dim}d, max_concurrent={max_concurrent_requests}"
        )

    async def start(self) -> None:
        """Start the engine loop."""
        if self._running:
            return
        self._running = True
        self._start_time = time.monotonic()
        self._loop_task = asyncio.get_running_loop().create_task(self._engine_loop())
        logger.info("EngineCore started")

    async def stop(self) -> None:
        """Stop the engine and release all resources (oMLX EngineCore.close pattern)."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

        # Signal all active collectors with sentinel
        for collector in self._output_collectors.values():
            try:
                collector.put(None)
            except Exception:
                pass
        for event in self._finished_events.values():
            event.set()

        self._output_collectors.clear()
        self._stream_states.clear()
        self._finished_events.clear()

        self.scheduler.shutdown()

        # Release model/tokenizer refs + GC + cache clear on executor
        self._model = None
        self._tokenizer = None
        gc.collect()
        loop = asyncio.get_running_loop()
        from .mlx_executor import sync_and_clear_cache
        try:
            await loop.run_in_executor(self._executor, sync_and_clear_cache)
        except Exception:
            pass

        logger.info("EngineCore stopped")

    async def add_request(
        self,
        prompt: str | list[int] | list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        stop: list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        seed: int | None = None,
        request_id: str | None = None,
        enable_thinking: bool | None = None,
        json_schema: dict | str | None = None,
        thinking_budget: int | None = None,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        **kwargs,
    ) -> str:
        """Add a generation request. Returns request_id for streaming/abort.

        If MemoryGuard preflight check fails, creates an error output
        immediately instead of adding to the scheduler.
        """
        from .request import Request, SamplingParams

        req_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

        # Encode prompt
        if isinstance(prompt, str):
            token_ids = self._tokenizer.encode(prompt)
        elif isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            text = self._messages_to_text(prompt, enable_thinking)
            token_ids = self._tokenizer.encode(text)
        else:
            token_ids = list(prompt)

        num_prompt_tokens = len(token_ids)

        # Memory guard preflight check — reject before adding to scheduler
        if self._memory_guard is not None:
            ok, reason = self._memory_guard.preflight_check(
                num_prompt_tokens=num_prompt_tokens,
                max_tokens=max_tokens,
            )
            if not ok:
                # Set up output collector with error response
                from .output_collector import RequestOutputCollector, RequestStreamState
                from .request import RequestOutput
                self._output_collectors[req_id] = RequestOutputCollector(aggregate=True)
                self._stream_states[req_id] = RequestStreamState(
                    stream_interval=self.config.stream_interval
                )
                self._finished_events[req_id] = asyncio.Event()

                error_output = RequestOutput(
                    request_id=req_id,
                    finished=True,
                    finish_reason="memory_exceeded",
                    error=f"Memory guard rejected: {reason}",
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=0,
                )
                self._output_collectors[req_id].put(error_output)
                self._output_collectors[req_id].put(None)  # sentinel
                self._finished_events[req_id].set()
                return req_id

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
            stop=stop or [],
            stop_token_ids=stop_token_ids or [],
            seed=seed,
            json_schema=json_schema,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )

        request = Request(
            request_id=req_id,
            prompt=prompt if isinstance(prompt, str) else token_ids,
            sampling_params=sampling_params,
            prompt_token_ids=token_ids,
            num_prompt_tokens=num_prompt_tokens,
            enable_thinking=enable_thinking,
        )

        # Set up per-request output management (oMLX EngineCore pattern)
        from .output_collector import RequestOutputCollector, RequestStreamState
        self._output_collectors[req_id] = RequestOutputCollector(aggregate=True)
        self._stream_states[req_id] = RequestStreamState(
            stream_interval=self.config.stream_interval
        )
        self._finished_events[req_id] = asyncio.Event()

        # Add to scheduler on MLX executor (thread-safe)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor, self.scheduler.add_request, request
        )

        return req_id

    async def abort_request(self, request_id: str) -> None:
        """Deferred abort (oMLX pattern: enqueued, processed at next step)."""
        from .request import RequestOutput
        self.scheduler.abort_request(request_id)
        # Put error output to wake up any waiting consumer
        collector = self._output_collectors.get(request_id)
        if collector is not None:
            collector.put(RequestOutput(
                request_id=request_id,
                finished=True,
                finish_reason="abort",
                error="Request aborted",
            ))
            collector.put(None)  # sentinel

    async def abort_all_requests(self) -> None:
        """Abort all active requests (error recovery)."""
        failed_ids = self.scheduler.fail_all_requests()
        for req_id in failed_ids:
            self._signal_finished(req_id)

    async def stream_outputs(self, request_id: str) -> AsyncIterator[Any]:
        """Stream outputs for a request (oMLX/vLLM pattern).

        Fast path: collector.get_nowait() avoids task switch under load.
        Slow path: await collector.get() for efficient waiting when idle.
        Sentinel (None) signals stream end.
        """
        collector = self._output_collectors.get(request_id)
        if collector is None:
            return

        try:
            while True:
                output = collector.get_nowait()
                if output is not None:
                    yield output
                    if output.finished:
                        break
                    continue

                # No buffered output — check if stream already ended
                if collector._sentinel:
                    break

                # Wait for new output from engine loop
                output = await collector.get()
                if output is None:
                    break
                yield output
                if output.finished:
                    break
        except asyncio.CancelledError:
            pass
        finally:
            self._cleanup_request(request_id)

    async def generate(
        self,
        **kwargs,
    ) -> Any:
        """Non-streaming generate: add request, wait for completion, return result."""
        from .request import RequestOutput
        req_id = await self.add_request(**kwargs)

        # Wait for completion
        event = self._finished_events.get(req_id)
        if event:
            await event.wait()

        # Drain collector
        collector = self._output_collectors.get(req_id)
        result = None
        if collector:
            while True:
                output = collector.get_nowait()
                if output is None:
                    break
                if output.finished:
                    result = output
                elif result is None:
                    result = output
                else:
                    result = collector._merge(result, output)
            self._cleanup_request(req_id)

        return result

    # ── Engine Loop ──

    async def _engine_loop(self) -> None:
        """Main engine loop — drives continuous batching (oMLX _engine_loop pattern).

        1. Run scheduler.step() on MLX executor (serialized GPU work)
        2. Distribute outputs to per-request collectors (on event loop)
        3. Signal finished events
        4. Yield to event loop
        """
        loop = asyncio.get_running_loop()
        use_simple_streaming = self.config.stream_interval == 1

        while self._running:
            if not self.scheduler.has_requests():
                await asyncio.sleep(self.config.step_interval)
                continue

            try:
                # Run scheduler step on MLX executor thread
                scheduler_output = await loop.run_in_executor(
                    self._executor, self.scheduler.step
                )
            except Exception as e:
                logger.error(f"Scheduler step error: {e}", exc_info=True)
                failed = self.scheduler.fail_all_requests()
                for req_id in failed:
                    self._signal_finished(req_id)
                await asyncio.sleep(0.1)
                continue

            # Distribute outputs to per-request collectors
            # Iterate live dict (abort may insert between steps, we need to see it)
            active_ids = list(self._output_collectors.keys())

            for req_output in scheduler_output.outputs:
                rid = req_output.request_id
                collector = self._output_collectors.get(rid)
                if collector is None:
                    continue

                if use_simple_streaming:
                    collector.put(req_output)
                else:
                    stream_state = self._stream_states.get(rid)
                    if stream_state and stream_state.should_send(
                        req_output.completion_tokens, req_output.finished
                    ):
                        collector.put(req_output)
                        stream_state.mark_sent(req_output.completion_tokens)

                if req_output.finished:
                    self._signal_finished(rid)
                    self._num_requests_processed += 1

            await asyncio.sleep(0)

    def _signal_finished(self, request_id: str) -> None:
        """Signal request completion."""
        event = self._finished_events.get(request_id)
        if event:
            event.set()

    def _cleanup_request(self, request_id: str) -> None:
        """Remove per-request output management state."""
        self._output_collectors.pop(request_id, None)
        self._stream_states.pop(request_id, None)
        self._finished_events.pop(request_id, None)
        self.scheduler.remove_finished_request(request_id)

    def _messages_to_text(
        self,
        messages: list[dict],
        enable_thinking: bool | None = None,
    ) -> str:
        """Convert chat messages to text using the model's chat template."""
        if self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                clean = [
                    {"role": m.get("role", "user"), "content": m.get("content", "")}
                    for m in messages
                ]
                kwargs: dict[str, Any] = {
                    "tokenize": False,
                    "add_generation_prompt": True,
                }
                if enable_thinking is not None:
                    kwargs["enable_thinking"] = enable_thinking
                text = self._tokenizer.apply_chat_template(clean, **kwargs)
                if text:
                    return text
            except Exception:
                logger.debug("chat template failed, using fallback", exc_info=True)

        # Generic fallback
        parts = []
        for m in messages:
            parts.append(f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def get_stats(self) -> dict:
        """Return engine core stats (oMLX pattern)."""
        uptime = time.monotonic() - self._start_time if self._start_time else 0
        scheduler_stats = self.scheduler.get_stats()
        return {
            "running": self._running,
            "num_requests_processed": self._num_requests_processed,
            "active_collectors": len(self._output_collectors),
            "uptime_seconds": round(uptime, 1),
            **{f"scheduler_{k}": v for k, v in scheduler_stats.items()},
        }

    def get_kv_cache_stats(self) -> dict:
        """Return KV prefix cache statistics (for admin endpoint)."""
        if self._kv_manager is None:
            return {"enabled": False}
        mgr = self._kv_manager
        pool = mgr.block_pool
        return {
            "enabled": True,
            "block_size": mgr.block_size,
            "total_blocks": pool.num_blocks,
            "free_blocks": mgr.num_free_blocks,
            "used_blocks": pool.num_blocks - 1 - mgr.num_free_blocks,
            "usage": round(mgr.usage, 3),
            "cached_hashes": len(pool._hash_to_block),
            "hit_rate": round(mgr.hit_rate, 4),
            "total_lookups": mgr._total_lookups,
            "total_hits": mgr._total_hits,
            "active_block_tables": len(
                getattr(self.scheduler, '_block_tables', {})
            ),
        }


class AsyncEngineCore:
    """Async context manager wrapper around EngineCore (oMLX AsyncEngineCore pattern).

    Usage:
        async with AsyncEngineCore(model, tokenizer) as engine:
            req_id = await engine.add_request("Hello")
            async for output in engine.stream_outputs_corrected(req_id):
                print(output.new_text)
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: EngineCoreConfig | None = None,
    ) -> None:
        self._core = EngineCore(model, tokenizer, config)

    async def __aenter__(self) -> AsyncEngineCore:
        await self._core.start()
        return self

    async def __aexit__(self, *args) -> None:
        await self._core.stop()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._core, name)
