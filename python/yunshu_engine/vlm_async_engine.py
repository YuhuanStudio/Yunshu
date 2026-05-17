from __future__ import annotations
"""Yunshu VLM Async Engine Core — concurrent VLM inference (§18.5).

Implements the oMLX VLMBatchedEngine pattern:
- Multiple VLM requests processed concurrently
- Vision encoding runs per-request (image features are request-specific)
- Text decode can overlap across requests (GPU serialization via executor)
- Per-request output collectors for streaming/non-streaming

Unlike the LLM EngineCore which uses BatchGenerator for true batched
decode, VLM uses request-level concurrency because:
1. mlx_vlm doesn't expose a batchable API
2. Vision encoding is per-request (different images)
3. Apple Silicon unified memory avoids GPU/CPU copy overhead

Architecture:
  VLMAsyncEngineCore
    → asyncio semaphore for max concurrency control
    → per-request tasks dispatched to MLX executor thread
    → output collectors for streaming/non-streaming response assembly
    → queue-based request lifecycle management
"""

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


@dataclass
class VLMRequestConfig:
    """Configuration for a single VLM request."""
    request_id: str = ""
    messages: list[dict] = field(default_factory=list)
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    repetition_penalty: float = 1.0
    stop: list[str] = field(default_factory=list)
    enable_thinking: bool | None = None
    stream: bool = False
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logit_bias: dict[int, float] | None = None
    json_schema: dict | str | None = None
    tools: list[dict] | None = None
    stop_token_ids: list[int] | None = None
    min_p: float = 0.0
    xtc_probability: float = 0.0
    xtc_threshold: float = 0.0
    stop: list[str] = field(default_factory=list)
    enable_thinking: bool | None = None
    stream: bool = False
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logit_bias: dict[int, float] | None = None
    json_schema: dict | str | None = None
    tools: list[dict] | None = None


@dataclass
class VLMStreamChunk:
    """A single chunk from VLM streaming output."""
    token_text: str = ""
    token_id: int = 0
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0


@dataclass
class _VLMRequestState:
    """Internal state for a VLM request."""
    config: VLMRequestConfig
    output_queue: asyncio.Queue
    finished_event: asyncio.Event
    start_time: float = 0.0
    done: bool = False


class VLMAsyncEngineCore:
    """Concurrent VLM inference engine (oMLX VLMBatchedEngine pattern).

    Manages multiple concurrent VLM requests with:
    - Semaphore-based concurrency control
    - Per-request output queues for streaming
    - GPU serialization via MLX executor thread
    - Request lifecycle tracking

    Usage:
        core = VLMAsyncEngineCore(vlm_engine, max_concurrent=4)
        await core.start()
        req_id = await core.add_request(messages=[...], max_tokens=256)
        async for chunk in core.stream_outputs(req_id):
            print(chunk.token_text)
        await core.stop()
    """

    def __init__(
        self,
        vlm_engine: Any,
        max_concurrent: int = 4,
    ) -> None:
        self._engine = vlm_engine
        self._max_concurrent = max_concurrent
        if os.environ.get("YUNSHU_VLM_MAX_CONCURRENT"):
            self._max_concurrent = int(os.environ["YUNSHU_VLM_MAX_CONCURRENT"])

        self._semaphore: asyncio.Semaphore | None = None
        self._requests: dict[str, _VLMRequestState] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._running = False
        self._stats = {
            "total_requests": 0,
            "active_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "total_tokens_generated": 0,
        }

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def num_active(self) -> int:
        return self._stats["active_requests"]

    async def start(self) -> None:
        """Start the async engine."""
        if self._running:
            return
        self._running = True
        self._semaphore = asyncio.Semaphore(self._max_concurrent)
        logger.info(
            f"VLMAsyncEngineCore started: max_concurrent={self._max_concurrent}"
        )

    async def stop(self) -> None:
        """Stop the engine and cancel all pending requests."""
        self._running = False

        # Cancel all tracked tasks
        for req_id, task in list(self._tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._tasks.clear()

        # Signal all active requests
        for state in self._requests.values():
            if not state.done:
                state.done = True
                try:
                    await state.output_queue.put(VLMStreamChunk(finish_reason="abort"))
                except Exception:
                    logger.debug("failed", exc_info=True)
                state.finished_event.set()

        self._requests.clear()
        self._stats["active_requests"] = 0
        logger.info("VLMAsyncEngineCore stopped")

    async def add_request(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: int | None = None,
        repetition_penalty: float = 1.0,
        stop: list[str] | None = None,
        enable_thinking: bool | None = None,
        stream: bool = False,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        json_schema: dict | str | None = None,
        tools: list[dict] | None = None,
        stop_token_ids: list[int] | None = None,
        min_p: float = 0.0,
        xtc_probability: float = 0.0,
        xtc_threshold: float = 0.0,
        **kwargs,
    ) -> str:
        """Add a VLM generation request. Returns request_id."""
        if not self._running:
            raise RuntimeError("Engine not started")

        req_id = f"vlm-{uuid.uuid4().hex[:8]}"

        config = VLMRequestConfig(
            request_id=req_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            repetition_penalty=repetition_penalty,
            stop=stop or [],
            enable_thinking=enable_thinking,
            stream=stream,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logit_bias=logit_bias,
            json_schema=json_schema,
            tools=tools,
            stop_token_ids=stop_token_ids,
            min_p=min_p,
            xtc_probability=xtc_probability,
            xtc_threshold=xtc_threshold,
        )

        state = _VLMRequestState(
            config=config,
            output_queue=asyncio.Queue(maxsize=1024),
            finished_event=asyncio.Event(),
            start_time=time.monotonic(),
        )

        self._requests[req_id] = state
        self._stats["total_requests"] += 1
        self._stats["active_requests"] += 1

        # Launch the request as a background task (stored for cancellation on shutdown)
        task = asyncio.create_task(self._process_request(state))
        self._tasks[req_id] = task
        task.add_done_callback(lambda t, rid=req_id: self._tasks.pop(rid, None))

        return req_id

    async def _process_request(self, state: _VLMRequestState) -> None:
        """Process a single VLM request with concurrency control."""
        config = state.config
        try:
            async with self._semaphore:
                if not self._running or state.done:
                    return

                if config.stream:
                    await self._process_streaming(state)
                else:
                    await self._process_non_streaming(state)

        except Exception as e:
            logger.error(f"VLM request {config.request_id} failed: {e}")
            self._stats["failed_requests"] += 1
            try:
                await state.output_queue.put(
                    VLMStreamChunk(finish_reason="error")
                )
            except Exception:
                logger.debug("failed", exc_info=True)
        finally:
            state.done = True
            state.finished_event.set()
            self._stats["active_requests"] = max(
                0, self._stats["active_requests"] - 1
            )
            self._stats["completed_requests"] += 1

    async def _process_non_streaming(self, state: _VLMRequestState) -> None:
        """Process a non-streaming VLM request."""
        config = state.config
        t0 = time.monotonic()

        result = await self._engine.generate(
            messages=config.messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            min_p=config.min_p,
            seed=config.seed,
            repetition_penalty=config.repetition_penalty,
            stop=config.stop,
            stop_token_ids=config.stop_token_ids,
            enable_thinking=config.enable_thinking,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
            logit_bias=config.logit_bias,
            json_schema=config.json_schema,
            xtc_probability=config.xtc_probability,
            xtc_threshold=config.xtc_threshold,
        )

        text = result.get("text", "")
        ttft = (time.monotonic() - t0) * 1000
        tokens = len(text) // 4 if text else 0
        self._stats["total_tokens_generated"] += tokens

        chunk = VLMStreamChunk(
            token_text=text,
            finish_reason=result.get("finish_reason", "stop"),
            prompt_tokens=result.get("prompt_tokens", 0),
            completion_tokens=result.get("completion_tokens", tokens),
            ttft_ms=ttft,
        )
        await state.output_queue.put(chunk)

    async def _process_streaming(self, state: _VLMRequestState) -> None:
        """Process a streaming VLM request."""
        config = state.config
        t0 = time.monotonic()
        first_token = True

        async for output in self._engine.generate_stream(
            messages=config.messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            min_p=config.min_p,
            seed=config.seed,
            repetition_penalty=config.repetition_penalty,
            stop=config.stop,
            stop_token_ids=config.stop_token_ids,
            enable_thinking=config.enable_thinking,
            frequency_penalty=config.frequency_penalty,
            presence_penalty=config.presence_penalty,
            logit_bias=config.logit_bias,
            json_schema=config.json_schema,
            xtc_probability=config.xtc_probability,
            xtc_threshold=config.xtc_threshold,
        ):
            if state.done:
                break

            ttft = (time.monotonic() - t0) * 1000 if first_token else 0.0
            first_token = False

            chunk = VLMStreamChunk(
                token_text=getattr(output, "token_text", ""),
                token_id=getattr(output, "token_id", 0),
                finish_reason=getattr(output, "finish_reason", None),
                ttft_ms=ttft,
            )
            await state.output_queue.put(chunk)
            self._stats["total_tokens_generated"] += 1

    async def stream_outputs(self, request_id: str) -> AsyncIterator[VLMStreamChunk]:
        """Stream output chunks for a request."""
        state = self._requests.get(request_id)
        if state is None:
            return

        try:
            while True:
                if not self._running and state.output_queue.empty():
                    break
                try:
                    chunk = await asyncio.wait_for(
                        state.output_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    if state.done:
                        break
                    continue

                yield chunk
                if chunk.finish_reason is not None:
                    break
        except asyncio.CancelledError:
            pass
        finally:
            self._cleanup_request(request_id)

    async def generate(self, **kwargs) -> VLMStreamChunk | None:
        """Non-streaming generate: add request, wait for completion."""
        kwargs.setdefault("stream", False)
        req_id = await self.add_request(**kwargs)

        state = self._requests.get(req_id)
        if state is None:
            return None

        await state.finished_event.wait()

        # Drain the queue
        result = None
        while not state.output_queue.empty():
            chunk = state.output_queue.get_nowait()
            if chunk is not None:
                result = chunk

        self._cleanup_request(req_id)
        return result

    async def abort_request(self, request_id: str) -> None:
        """Abort a pending request."""
        state = self._requests.get(request_id)
        if state is not None:
            state.done = True
            state.finished_event.set()
            try:
                await state.output_queue.put(
                    VLMStreamChunk(finish_reason="abort")
                )
            except Exception:
                logger.debug("failed", exc_info=True)

    def _cleanup_request(self, request_id: str) -> None:
        """Remove request state."""
        self._requests.pop(request_id, None)

    def get_stats(self) -> dict[str, Any]:
        """Return engine statistics."""
        return {
            "running": self._running,
            "max_concurrent": self._max_concurrent,
            **self._stats,
        }
