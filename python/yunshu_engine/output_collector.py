from __future__ import annotations
"""Yunshu Output Collector — low-latency streaming with smart buffering.

Studied from oMLX's output_collector.py (vLLM pattern):
- Non-blocking get_nowait() to avoid task switches under load
- Output aggregation when producer is faster than consumer
- Event-based signaling for efficient waiting
- stream_interval batching for configurable send frequency
- Sentinel-based stream termination

Written from scratch for Yunshu, importing from our request module.
"""

import asyncio
from dataclasses import dataclass
from typing import ClassVar, Optional

from .request import RequestOutput


class RequestOutputCollector:
    """Per-request output collector with smart buffering (vLLM/oMLX pattern).

    Producer side (engine loop): collector.put(output)
    Consumer side (streaming generator): output = collector.get_nowait() or await collector.get()

    Passing None signals stream end (sentinel pattern).
    Aggregation merges consecutive outputs when producer gets ahead.
    """

    _waiting_consumers: ClassVar[int] = 0

    def __init__(self, aggregate: bool = True):
        self.output: Optional[RequestOutput] = None
        self.ready = asyncio.Event()
        self.aggregate = aggregate
        self._is_waiting = False
        self._sentinel = False

    def put(self, output: RequestOutput | None) -> None:
        if output is None:
            self._sentinel = True
            self.ready.set()
            return
        if self.output is None:
            self.output = output
        elif self.aggregate:
            self.output = self._merge(self.output, output)
        else:
            self.output = output
        self.ready.set()

    def get_nowait(self) -> Optional[RequestOutput]:
        output = self.output
        if output is not None:
            self.output = None
            self.ready.clear()
            return output
        if self._sentinel:
            return None
        return None

    async def get(self) -> Optional[RequestOutput]:
        if not self._is_waiting:
            self._is_waiting = True
            RequestOutputCollector._waiting_consumers += 1
        try:
            while self.output is None and not self._sentinel:
                await self.ready.wait()
            return self.get_nowait()
        finally:
            if self._is_waiting:
                self._is_waiting = False
                RequestOutputCollector._waiting_consumers = max(
                    0, RequestOutputCollector._waiting_consumers - 1
                )

    def _merge(self, existing: RequestOutput, new: RequestOutput) -> RequestOutput:
        # Accumulate logprobs across steps (fast path gives per-step lists)
        _lp = existing.logprobs
        if new.logprobs is not None:
            if _lp is None:
                _lp = new.logprobs
            elif isinstance(_lp, list) and isinstance(new.logprobs, list):
                _lp = _lp + new.logprobs
            else:
                _lp = new.logprobs
        # Take max reasoning_tokens (scheduler provides cumulative count, not incremental)
        _reasoning = max(existing.reasoning_tokens or 0, new.reasoning_tokens or 0)
        # Take max cached_tokens (monotonic, not cumulative)
        _cached = max(existing.cached_tokens or 0, new.cached_tokens or 0)
        # Preserve TTFT: use existing if set (> 0, first-token timing), else new
        _ttft = existing.ttft_ms if existing.ttft_ms > 0 else (new.ttft_ms if new.ttft_ms > 0 else 0.0)
        return RequestOutput(
            request_id=new.request_id,
            new_token_ids=existing.new_token_ids + new.new_token_ids,
            new_text=existing.new_text + new.new_text,
            output_token_ids=new.output_token_ids,
            output_text=new.output_text,
            finished=new.finished,
            finish_reason=new.finish_reason,
            prompt_tokens=new.prompt_tokens,
            completion_tokens=new.completion_tokens,
            logprobs=_lp,
            current_state=new.current_state,
            reasoning_tokens=_reasoning,
            cached_tokens=_cached,
            error=new.error if new.error is not None else existing.error,
            ttft_ms=_ttft,
        )

    def clear(self) -> None:
        self.output = None
        self.ready.clear()
        self._sentinel = False
        if self._is_waiting:
            self._is_waiting = False
            RequestOutputCollector._waiting_consumers = max(
                0, RequestOutputCollector._waiting_consumers - 1
            )

    @classmethod
    def has_waiting_consumers(cls) -> bool:
        return cls._waiting_consumers > 0


@dataclass
class RequestStreamState:
    """Tracks stream_interval batching for a request (oMLX pattern)."""

    stream_interval: int = 1
    sent_tokens: int = 0

    def should_send(self, total_tokens: int, finished: bool) -> bool:
        if finished:
            return True
        if self.sent_tokens == 0:
            return True
        return (total_tokens - self.sent_tokens) >= self.stream_interval

    def mark_sent(self, total_tokens: int) -> None:
        self.sent_tokens = total_tokens
