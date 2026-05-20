from __future__ import annotations
"""Forward batch representation — multi-level batch hierarchy (vLLM/SGLang pattern).

SGLang uses a 3-level batch representation:
  ScheduleBatch → ModelWorkerBatch → ForwardBatch

Yunshu adopts a similar hierarchy optimized for Apple Silicon + MLX:
  ScheduleBatch — scheduler-level: requests + priorities + metadata
  ForwardBatch  — GPU-level: token arrays + position IDs + KV indices
  BatchResult   — output-level: generated tokens + logits + metadata

This separation allows:
  - Scheduler to reorder/priorize without touching GPU arrays
  - GPU forward to work with compact arrays (no gaps)
  - Output to be distributed back to per-request collectors efficiently
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RequestSlot:
    """A single request's slot in the schedule batch."""
    request_id: str
    prompt_tokens: list[int]
    generated_tokens: list[int] = field(default_factory=list)
    max_tokens: int = 512
    priority: int = 0
    is_prefill: bool = True
    sampling_params: Any = None
    # KV cache tracking
    kv_block_ids: list[int] = field(default_factory=list)
    kv_slots_used: int = 0
    # Spec decode
    spec_draft_tokens: list[int] = field(default_factory=list)
    spec_proposer: str = ""
    # Timing
    arrival_time: float = 0.0
    first_token_time: float | None = None
    # Context
    num_prompt_tokens: int = 0
    enable_thinking: bool = False
    thinking_budget: int | None = None
    eos_token_ids: list[int] = field(default_factory=lambda: [2])

    @property
    def total_tokens(self) -> int:
        # Use len(prompt_tokens) as the authoritative count when the list
        # is non-empty.  num_prompt_tokens may be stale if only prompt_tokens
        # was set (e.g. from_schedule_batch relies on this property for
        # token budget accounting).
        prompt_count = self.num_prompt_tokens or len(self.prompt_tokens)
        return prompt_count + len(self.generated_tokens)

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - len(self.generated_tokens))

    @property
    def is_finished(self) -> bool:
        if len(self.generated_tokens) >= self.max_tokens:
            return True
        if self.eos_token_ids and any(t in self.eos_token_ids for t in self.generated_tokens):
            return True
        return False

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_time is not None and self.arrival_time > 0:
            return (self.first_token_time - self.arrival_time) * 1000
        return None


@dataclass
class ScheduleBatch:
    """Scheduler-level batch representation.

    Contains per-request metadata, priorities, and sampling params.
    The scheduler reorders slots by priority, groups prefill vs decode,
    and produces a ForwardBatch for GPU execution.
    """

    slots: list[RequestSlot] = field(default_factory=list)
    max_batch_size: int = 32
    max_prefill_batch: int = 8
    max_decode_batch: int = 32
    created_at: float = field(default_factory=time.monotonic)

    @property
    def num_slots(self) -> int:
        return len(self.slots)

    @property
    def prefill_slots(self) -> list[RequestSlot]:
        return [s for s in self.slots if s.is_prefill]

    @property
    def decode_slots(self) -> list[RequestSlot]:
        return [s for s in self.slots if not s.is_prefill]

    @property
    def total_tokens(self) -> int:
        return sum(s.total_tokens for s in self.slots)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(s.num_prompt_tokens or len(s.prompt_tokens) for s in self.slots)

    @property
    def total_generated_tokens(self) -> int:
        return sum(len(s.generated_tokens) for s in self.slots)

    def add_slot(self, slot: RequestSlot) -> None:
        self.slots.append(slot)

    def remove_slot(self, request_id: str) -> RequestSlot | None:
        for i, s in enumerate(self.slots):
            if s.request_id == request_id:
                return self.slots.pop(i)
        return None

    def get_slot(self, request_id: str) -> RequestSlot | None:
        for s in self.slots:
            if s.request_id == request_id:
                return s
        return None

    def reorder_by_priority(self) -> None:
        self.slots.sort(key=lambda s: -s.priority)

    def split_prefill_decode(self) -> tuple[ScheduleBatch, ScheduleBatch]:
        """Split into prefill batch and decode batch (Sarathi pattern).

        Overflow slots (those that don't fit in either sub-batch) are placed
        back into ``self`` so the caller can re-queue them.  Without this,
        slots were silently dropped, orphaning requests from the scheduler
        pipeline and causing them to never be processed.
        """
        prefill = ScheduleBatch(
            max_batch_size=self.max_prefill_batch,
            created_at=self.created_at,
        )
        decode = ScheduleBatch(
            max_batch_size=self.max_decode_batch,
            created_at=self.created_at,
        )
        overflow: list[RequestSlot] = []
        for slot in self.slots:
            if slot.is_prefill:
                if prefill.num_slots < self.max_prefill_batch:
                    prefill.add_slot(slot)
                else:
                    overflow.append(slot)
            else:
                if decode.num_slots < self.max_decode_batch:
                    decode.add_slot(slot)
                else:
                    overflow.append(slot)
        # Retain overflow in self so the caller can re-schedule them.
        # This prevents silent request loss.
        self.slots = overflow
        return prefill, decode

    def remove_finished(self) -> list[RequestSlot]:
        """Remove and return finished slots."""
        finished = [s for s in self.slots if s.is_finished]
        self.slots = [s for s in self.slots if not s.is_finished]
        return finished

    def compact(self) -> None:
        """Remove empty/finished slots."""
        self.slots = [s for s in self.slots if not s.is_finished]

    def get_stats(self) -> dict:
        prefill_count = len(self.prefill_slots)
        decode_count = len(self.decode_slots)
        avg_priority = (
            sum(s.priority for s in self.slots) / len(self.slots)
            if self.slots
            else 0.0
        )
        ttfts = [s.ttft_ms for s in self.slots if s.ttft_ms is not None]
        return {
            "num_slots": self.num_slots,
            "num_prefill": prefill_count,
            "num_decode": decode_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_generated_tokens": self.total_generated_tokens,
            "avg_priority": round(avg_priority, 2),
            "ttft_avg_ms": round(sum(ttfts) / len(ttfts), 2) if ttfts else None,
            "ttft_p50_ms": round(sorted(ttfts)[len(ttfts) // 2], 2) if ttfts else None,
        }


@dataclass
class ForwardBatch:
    """GPU-level batch representation.

    Contains compact arrays ready for MLX model forward pass.
    No gaps, no metadata — just raw token/position/KV data.
    """

    # Input token IDs [batch_size, seq_len] or [total_tokens]
    input_ids: Any = None  # mx.array
    # Position IDs [batch_size, seq_len] or [total_tokens]
    position_ids: Any = None  # mx.array
    # Per-request boundaries (for ragged batching)
    request_lengths: list[int] = field(default_factory=list)
    request_ids: list[str] = field(default_factory=list)
    # KV cache block indices per request
    kv_block_tables: Any = None  # mx.array or None
    kv_slot_indices: Any = None  # mx.array or None
    # Which slots are prefill vs decode
    is_prefill_mask: list[bool] = field(default_factory=list)
    # Spec decode draft tokens (if applicable)
    spec_draft_tokens: list[list[int]] = field(default_factory=list)
    spec_draft_lengths: list[int] = field(default_factory=list)
    # Batch metadata
    batch_size: int = 0
    total_tokens: int = 0
    max_seq_len: int = 0
    created_at: float = field(default_factory=time.monotonic)

    @classmethod
    def from_schedule_batch(cls, batch: ScheduleBatch) -> ForwardBatch:
        """Convert a ScheduleBatch to ForwardBatch for GPU execution.

        Flattens all token IDs into a single array, with request_lengths
        to track per-request boundaries (ragged batch).
        """
        all_tokens: list[int] = []
        request_lengths: list[int] = []
        request_ids: list[str] = []
        is_prefill_mask: list[bool] = []
        spec_draft_tokens: list[list[int]] = []
        spec_draft_lengths: list[int] = []

        for slot in batch.slots:
            if slot.is_prefill:
                tokens = slot.prompt_tokens
            else:
                tokens = slot.generated_tokens[-1:] if slot.generated_tokens else [0]

            all_tokens.extend(tokens)
            request_lengths.append(len(tokens))
            request_ids.append(slot.request_id)
            is_prefill_mask.append(slot.is_prefill)
            spec_draft_tokens.append(slot.spec_draft_tokens)
            spec_draft_lengths.append(len(slot.spec_draft_tokens))

        total = len(all_tokens)
        max_len = max(request_lengths) if request_lengths else 0

        # Create MLX arrays (lazy — only materialized when model forward runs)
        input_ids = None
        position_ids = None
        if all_tokens:
            try:
                import mlx.core as mx
                input_ids = mx.array(all_tokens, dtype=mx.int32)
                pos = []
                for length, slot in zip(request_lengths, batch.slots):
                    if slot.is_prefill:
                        pos.extend(range(length))
                    else:
                        # Decode step: position of the last generated token.
                        # generated_tokens[-1:] (1 token) is at position
                        # num_prompt_tokens + (N-1) where N = len(generated_tokens).
                        # Fallback for edge case: if generated_tokens is empty and
                        # num_prompt_tokens is 0, use total_tokens which accounts
                        # for prompt_tokens list length as a reliable fallback.
                        if slot.generated_tokens:
                            start = slot.num_prompt_tokens + len(slot.generated_tokens) - 1
                        else:
                            # Use the authoritative prompt count: prefer
                            # num_prompt_tokens, but fall back to the actual
                            # prompt_tokens list length if num_prompt_tokens
                            # was never set (e.g. from BatchComposer active_slots
                            # which only sets is_prefill and priority).
                            prompt_count = slot.num_prompt_tokens or len(slot.prompt_tokens)
                            start = prompt_count
                        pos.extend(range(start, start + length))
                position_ids = mx.array(pos, dtype=mx.int32)
            except ImportError:
                pass

        return cls(
            input_ids=input_ids,
            position_ids=position_ids,
            request_lengths=request_lengths,
            request_ids=request_ids,
            is_prefill_mask=is_prefill_mask,
            spec_draft_tokens=spec_draft_tokens,
            spec_draft_lengths=spec_draft_lengths,
            batch_size=len(request_lengths),
            total_tokens=total,
            max_seq_len=max_len,
        )

    @property
    def num_prefill(self) -> int:
        return sum(self.is_prefill_mask)

    @property
    def num_decode(self) -> int:
        return len(self.is_prefill_mask) - self.num_prefill

    @property
    def has_spec_drafts(self) -> bool:
        return any(n > 0 for n in self.spec_draft_lengths)


@dataclass
class BatchResult:
    """Output from a single forward pass.

    Contains per-request generated tokens, logits, and metadata.
    Maps back from ForwardBatch indices to request IDs.
    """

    request_ids: list[str] = field(default_factory=list)
    # Generated token IDs per request (list of lists for spec decode)
    generated_token_ids: list[list[int]] = field(default_factory=list)
    # Logits for the generated tokens (for logprobs)
    logits: Any = None  # mx.array or None
    # Per-request finish reasons
    finish_reasons: list[str | None] = field(default_factory=list)
    # Spec decode verification results
    spec_accepted_count: list[int] = field(default_factory=list)
    spec_rejected_count: list[int] = field(default_factory=list)
    # Timing
    forward_time_ms: float = 0.0
    sample_time_ms: float = 0.0
    total_time_ms: float = 0.0
    # Memory at time of forward
    memory_active_bytes: int = 0
    memory_peak_bytes: int = 0
    created_at: float = field(default_factory=time.monotonic)

    @property
    def batch_size(self) -> int:
        return len(self.request_ids)

    def get_per_request_results(self) -> dict[str, dict]:
        """Split into per-request result dicts."""
        results = {}
        for i, rid in enumerate(self.request_ids):
            results[rid] = {
                "token_ids": self.generated_token_ids[i] if i < len(self.generated_token_ids) else [],
                "finish_reason": self.finish_reasons[i] if i < len(self.finish_reasons) else None,
                "spec_accepted": self.spec_accepted_count[i] if i < len(self.spec_accepted_count) else 0,
                "spec_rejected": self.spec_rejected_count[i] if i < len(self.spec_rejected_count) else 0,
            }
        return results

    def get_stats(self) -> dict:
        return {
            "batch_size": self.batch_size,
            "forward_time_ms": round(self.forward_time_ms, 3),
            "sample_time_ms": round(self.sample_time_ms, 3),
            "total_time_ms": round(self.total_time_ms, 3),
            "memory_active_mb": round(self.memory_active_bytes / 1024 / 1024, 1),
            "memory_peak_mb": round(self.memory_peak_bytes / 1024 / 1024, 1),
            "spec_total_accepted": sum(self.spec_accepted_count),
            "spec_total_rejected": sum(self.spec_rejected_count),
        }


class BatchComposer:
    """Composes schedule batches from pending requests (vLLM Scheduler pattern).

    Takes pending requests, prioritizes them, and forms schedule batches
    respecting:
      - max_batch_size limits
      - prefill vs decode slot allocation
      - memory budgets
      - priority ordering
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_prefill_slots: int = 8,
        max_decode_slots: int = 32,
        prefill_chunk_size: int = 2048,
        priority_weight: float = 1.0,
        ttft_weight: float = 0.5,
    ) -> None:
        self._max_batch = max_batch_size
        self._max_prefill = max_prefill_slots
        self._max_decode = max_decode_slots
        self._prefill_chunk = prefill_chunk_size
        self._priority_weight = priority_weight
        self._ttft_weight = ttft_weight
        self._total_batches_composed = 0
        self._total_requests_scheduled = 0

    def compose(
        self,
        pending: list[RequestSlot],
        active_slots: list[RequestSlot] | None = None,
        memory_budget_tokens: int = 0,
    ) -> ScheduleBatch:
        """Compose a schedule batch from pending + active requests.

        Strategy:
        1. Active decode requests get priority (they hold KV cache)
        2. New prefill requests fill remaining slots
        3. Memory budget constrains total tokens

        Bug fixes applied:
        - Phase 1: sort active decode by priority (descending) so high-priority
          decode requests are preferred when decode_slots is limited.
        - Phase 2: token_budget deducted per-decode-slot is 1 (one new token
          per step), not the full history (prompt_tokens + generated_tokens).
          Without this, a few long-running decode requests exhaust the budget
          and block all new prefill requests.
        - Stats: only increment counters when the batch is non-empty.
        """
        batch = ScheduleBatch(
            max_batch_size=self._max_batch,
            max_prefill_batch=self._max_prefill,
            max_decode_batch=self._max_decode,
        )

        # Phase 1: Carry forward active decode requests (priority-sorted)
        decode_count = 0
        if active_slots:
            # Sort active decode by priority descending so that when
            # decode_slots is limited, higher-priority decode requests
            # are kept.  Without this sort, iteration order (dict insertion
            # order) determines which decode slots survive, which can evict
            # high-priority requests while keeping low-priority ones.
            decode_candidates = sorted(
                [s for s in active_slots if not s.is_prefill and not s.is_finished],
                key=lambda s: -s.priority,
            )
            for slot in decode_candidates:
                if batch.num_slots >= self._max_batch:
                    break
                if decode_count >= self._max_decode:
                    break
                batch.add_slot(slot)
                decode_count += 1

        # Phase 2: Add new prefill requests by priority
        new_requests = sorted(
            [s for s in pending if s.is_prefill],
            key=lambda s: self._schedule_score(s),
            reverse=True,
        )

        prefill_count = len(batch.prefill_slots)
        token_budget = memory_budget_tokens
        if token_budget > 0:
            # Deduct decode overhead: each decode slot only generates 1 new
            # token per step, not its full prompt + generated history.
            # The original code used batch.total_tokens which includes all
            # historical tokens for each decode slot — massively over-counting
            # memory usage and blocking all new prefill requests.
            token_budget -= decode_count  # 1 token per decode step

        for slot in new_requests:
            if batch.num_slots >= self._max_batch:
                break
            if prefill_count >= self._max_prefill:
                break
            if token_budget > 0 and (slot.num_prompt_tokens or len(slot.prompt_tokens)) > token_budget:
                continue

            batch.add_slot(slot)
            prefill_count += 1
            if token_budget > 0:
                token_budget -= slot.num_prompt_tokens or len(slot.prompt_tokens)

        if batch.num_slots > 0:
            self._total_batches_composed += 1
            self._total_requests_scheduled += batch.num_slots
        return batch

    def _schedule_score(self, slot: RequestSlot) -> float:
        """Compute scheduling priority score (higher = more urgent)."""
        score = slot.priority * self._priority_weight
        # Age bonus: older requests get slight priority boost
        if slot.arrival_time > 0:
            age = time.monotonic() - slot.arrival_time
            score += age * self._ttft_weight
        return score

    def get_stats(self) -> dict:
        return {
            "total_batches_composed": self._total_batches_composed,
            "total_requests_scheduled": self._total_requests_scheduled,
            "max_batch_size": self._max_batch,
            "max_prefill_slots": self._max_prefill,
            "max_decode_slots": self._max_decode,
        }
