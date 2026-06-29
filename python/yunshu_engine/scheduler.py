from __future__ import annotations

"""Yunshu Scheduler — continuous batching via mlx-lm BatchGenerator.

Features:
- mlx-lm BatchGenerator as backend
- Request lifecycle: waiting → running → finished
- Per-request sampler + SequenceStateMachine
- Per-request detokenizer (never pool — reset() leaks byte buffers)
- Deferred cache clearing (prevent IOKit kernel panics)
- Thread-safe abort via pending set
- Step counter + stats for monitoring

Architecture:
  EngineCore.step_loop() → Scheduler.step()
    → insert waiting requests into BatchGenerator
    → BatchGenerator.next() → process responses
    → distribute outputs to per-request collectors
    → deferred mx.clear_cache() when idle
"""

import copy
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from yunshu_kv.thinking_segment import ThinkingSegmentConfig, ThinkingSegmentSubstore

from .ngram_proposer import NgramConfig, NgramProposer
from .priority_queue import RequestPriorityQueue, make_waiting_queue
from .request import Request, RequestOutput, RequestStatus, SamplingParams
from .speculative_decoder import DraftResult, SpeculativeDecoder

logger = logging.getLogger(__name__)


# ── Batch-path SpecPrefill, Spec-Aware Scheduling, Batched Draft Collection ──


@dataclass
class BatchSpecPrefillConfig:
    """Configuration for batch-path SpecPrefill.

    Enables sparse prefill for long prompts in the batch/scheduler path,
    using a draft model's attention scores to skip unimportant tokens.
    Controlled via YUNSHU_BATCH_SPEC_PREFILL=1.
    """

    enabled: bool = False
    threshold: int = 8192  # Minimum prompt length to trigger
    keep_rate: float = 0.20  # Fraction of tokens to keep
    chunk_size: int = 32  # Chunk size for token selection
    draft_model: Any = None  # Draft model for attention scoring


@dataclass
class SpecBudget:
    """Result of spec-aware batch slot computation.

    Attributes:
        total_slots: Total available batch slots.
        decode_slots: Slots reserved for decode (running requests).
        spec_slots: Slots reserved for spec verification overhead.
        available_for_new: Slots available for new request insertion.
    """

    total_slots: int = 256
    decode_slots: int = 0
    spec_slots: int = 0
    available_for_new: int = 0


@dataclass
class DraftCollection:
    """Structured result from batched draft collection.

    Maps request_id → list of draft token IDs from all spec strategies.
    Used by the scheduler to batch-verify all drafts in a single forward pass.
    """

    drafts: dict[str, list[int]] = field(default_factory=dict)
    strategy_counts: dict[str, int] = field(default_factory=dict)
    total_draft_tokens: int = 0

    def add(
        self, request_id: str, tokens: list[int], strategy: str = "unknown"
    ) -> None:
        """Add draft tokens for a request."""
        if tokens:
            self.drafts[request_id] = tokens
            self.strategy_counts[strategy] = self.strategy_counts.get(
                strategy, 0
            ) + len(tokens)
            self.total_draft_tokens += len(tokens)

    def has_drafts(self) -> bool:
        return bool(self.drafts)

    def get_request_ids(self) -> list[str]:
        return list(self.drafts.keys())

    def merge(self, other: DraftCollection) -> None:
        """Merge another DraftCollection into this one (other takes priority)."""
        for rid, tokens in other.drafts.items():
            if rid not in self.drafts:
                self.drafts[rid] = tokens
        for strategy, count in other.strategy_counts.items():
            self.strategy_counts[strategy] = (
                self.strategy_counts.get(strategy, 0) + count
            )
        self.total_draft_tokens += other.total_draft_tokens


class BatchPathSpecPrefill:
    """SpecPrefill integration for the batch scheduler path.

    When a new request enters the scheduler with a long prompt, uses the
    draft model's attention scores to identify which prompt tokens to skip
    during prefill. This reduces TTFT for long prompts in batch mode.

    Pipeline (mirrors single-request _generate_fast path):
      1. score_tokens()  — draft model scores token importance
      2. select_chunks() — chunk-based top-K% selection
      3. sparse_prefill() — target prefill with selected tokens

    Enabled via YUNSHU_BATCH_SPEC_PREFILL=1.
    """

    def __init__(self, config: BatchSpecPrefillConfig | None = None) -> None:
        self.config = config or BatchSpecPrefillConfig()
        self._stats = {
            "prefills_attempted": 0,
            "prefills_succeeded": 0,
            "prefills_fallback": 0,
            "tokens_skipped_total": 0,
            "tokens_kept_total": 0,
        }

    def should_prefill(self, prompt_length: int) -> bool:
        """Check if SpecPrefill should be applied to a prompt."""
        return (
            self.config.enabled
            and self.config.draft_model is not None
            and prompt_length >= self.config.threshold
        )

    def compute_skippable_tokens(self, tokens: list[int]) -> list[int] | None:
        """Compute which tokens are skippable using draft model attention.

        Returns the list of selected (important) token indices, or None if
        the prompt is too short or scoring fails.
        """
        if not self.should_prefill(len(tokens)):
            return None

        self._stats["prefills_attempted"] += 1

        try:
            from .spec_prefill import score_tokens, select_chunks

            importance = score_tokens(
                self.config.draft_model,
                tokens,
            )
            selected = select_chunks(
                importance,
                keep_pct=self.config.keep_rate,
                chunk_size=self.config.chunk_size,
            )
            selected_list = selected.tolist()
            n_kept = len(selected_list)
            n_skipped = len(tokens) - n_kept

            self._stats["prefills_succeeded"] += 1
            self._stats["tokens_kept_total"] += n_kept
            self._stats["tokens_skipped_total"] += n_skipped

            logger.debug(
                f"Batch SpecPrefill: {n_kept}/{len(tokens)} tokens selected "
                f"({n_skipped} skipped)"
            )
            return selected_list

        except Exception as e:
            self._stats["prefills_fallback"] += 1
            logger.debug(f"Batch SpecPrefill scoring failed: {e}")
            return None

    def get_stats(self) -> dict:
        """Return SpecPrefill statistics."""
        stats = dict(self._stats)
        if stats["prefills_attempted"] > 0:
            stats["success_rate"] = round(
                stats["prefills_succeeded"] / stats["prefills_attempted"], 3
            )
        else:
            stats["success_rate"] = 0.0
        return stats


class SpecAwareBatchScheduler:
    """Spec-decode aware batch slot allocation.

    When speculative decoding is active (N-gram, cross-model, MTP, or Medusa),
    the scheduler must reserve batch slots for draft verification overhead.
    Without slot reservation, verification can starve normal decode or cause
    batch overflow.

    Integration with TBO: When both TBO and spec decode are enabled, draft
    generation overlaps with verification — the draft model runs on CPU while
    the target model verifies on GPU.
    """

    def __init__(
        self,
        max_num_seqs: int = 256,
        spec_overhead_per_request: float = 0.1,
        tbo_enabled: bool = False,
    ) -> None:
        self.max_num_seqs = max_num_seqs
        self.spec_overhead_per_request = spec_overhead_per_request
        self.tbo_enabled = tbo_enabled
        self._stats = {
            "budget_computations": 0,
            "spec_slots_reserved": 0,
            "total_slots_reserved": 0,
            "tbo_overlap_steps": 0,
        }

    def compute_spec_budget(
        self,
        num_running: int,
        spec_overhead: float | None = None,
    ) -> SpecBudget:
        """Calculate batch slot allocation considering spec decode overhead.

        Args:
            num_running: Number of currently running decode requests.
            spec_overhead: Per-request spec overhead (0.0–1.0). If None,
                uses the configured default.

        Returns:
            SpecBudget with slot allocation details.
        """
        if spec_overhead is None:
            spec_overhead = self.spec_overhead_per_request

        total = self.max_num_seqs
        # Clamp decode_slots to total — during preemption transitions,
        # num_running can briefly exceed max_num_seqs (e.g., before the
        # preempted request is removed from running). Without the clamp,
        # decode_slots + spec_slots exceeds total, and available becomes
        # negative (clamped to 0), but the inflated decode_slots prevents
        # new requests from being scheduled even after preemption frees slots.
        decode_slots = min(num_running, total)

        # Reserve slots proportional to spec verification overhead.
        # Each running request with spec decode active consumes extra slot
        # budget for draft verification. The overhead is the fraction of a
        # full slot each verification costs.
        spec_slots = 0
        if spec_overhead > 0 and decode_slots > 0:
            # TBO overlap: draft generation overlaps with verification,
            # reducing effective overhead by ~50%.
            effective_overhead = spec_overhead
            if self.tbo_enabled:
                effective_overhead *= 0.5
                self._stats["tbo_overlap_steps"] += 1
            # Use decode_slots (clamped) not raw num_running to avoid
            # spec_slots being proportional to an inflated running count.
            spec_slots = max(0, round(decode_slots * effective_overhead))

        # Ensure spec_slots doesn't push total reservation beyond capacity.
        # When decode_slots is near max_num_seqs, the proportional spec
        # reservation can exceed remaining capacity — cap it.
        spec_slots = min(spec_slots, total - decode_slots)
        available = max(0, total - decode_slots - spec_slots)

        self._stats["budget_computations"] += 1
        self._stats["spec_slots_reserved"] += spec_slots
        self._stats["total_slots_reserved"] += decode_slots + spec_slots

        return SpecBudget(
            total_slots=total,
            decode_slots=decode_slots,
            spec_slots=spec_slots,
            available_for_new=available,
        )

    def get_available_slots(
        self,
        num_running: int,
        spec_overhead: float | None = None,
    ) -> int:
        """Quick access to available slot count."""
        return self.compute_spec_budget(num_running, spec_overhead).available_for_new

    def get_stats(self) -> dict:
        """Return spec-aware scheduling statistics."""
        stats = dict(self._stats)
        stats["max_num_seqs"] = self.max_num_seqs
        stats["spec_overhead_per_request"] = self.spec_overhead_per_request
        stats["tbo_enabled"] = self.tbo_enabled
        if stats["budget_computations"] > 0:
            stats["avg_spec_slots"] = round(
                stats["spec_slots_reserved"] / stats["budget_computations"], 2
            )
        else:
            stats["avg_spec_slots"] = 0.0
        return stats


class BatchedDraftCollection:
    """Collect draft tokens from all spec strategies for all running requests.

    The key missing piece: currently drafts are verified per-request, not
    batched. This class collects drafts from N-gram, cross-model, MTP, and
    Medusa strategies for ALL running requests, returning them as a structured
    DraftCollection that the scheduler uses to batch-verify all drafts in a
    single forward pass.

    Usage in scheduler.step():
      1. After decode, call collect_all_drafts() for all running requests
      2. Store the DraftCollection for next-step verification
      3. On next step, verify all drafts together
    """

    def __init__(self) -> None:
        self._stats = {
            "collections": 0,
            "total_tokens_collected": 0,
            "strategy_breakdown": {},
        }

    def collect_all_drafts(
        self,
        running: dict[str, Request],
        spec_decoder: Any | None,
        mtp_decoder: Any | None,
        ngram_proposer: NgramProposer | None,
        pending_abort_ids: set[str],
        spec_drafts: dict[str, list[int]] | None = None,
    ) -> DraftCollection:
        """Collect draft tokens from all strategies for all running requests.

        Args:
            running: Map of request_id → Request for active requests.
            spec_decoder: Cross-model SpeculativeDecoder (or None).
            mtp_decoder: MTP decoder (or None).
            ngram_proposer: N-gram proposer (or None).
            pending_abort_ids: Set of request IDs pending abort.
            spec_drafts: Existing drafts (to avoid overwriting).

        Returns:
            DraftCollection with all collected drafts.
        """
        collection = DraftCollection()
        self._stats["collections"] += 1

        for rid, req in list(running.items()):
            # Skip aborted requests
            if rid in pending_abort_ids:
                continue
            # Skip requests that already have pending drafts
            if spec_drafts and rid in spec_drafts:
                collection.drafts[rid] = spec_drafts[rid]
                continue
            # Skip requests with no output tokens (can't seed a draft)
            if not req.output_token_ids:
                continue

            # Try N-gram first (zero GPU overhead)
            if ngram_proposer is not None:
                tokens = self._collect_ngram_draft(req, ngram_proposer)
                if tokens:
                    collection.add(rid, tokens, "ngram")
                    continue

            # Try MTP (self-speculative, low overhead)
            if mtp_decoder is not None:
                tokens = self._collect_mtp_draft(req, mtp_decoder)
                if tokens:
                    collection.add(rid, tokens, "mtp")
                    continue

            # Try cross-model spec decode (higher overhead)
            if spec_decoder is not None and isinstance(
                spec_decoder, SpeculativeDecoder
            ):
                tokens = self._collect_cross_model_draft(req, spec_decoder)
                if tokens:
                    collection.add(rid, tokens, "cross_model")
                    continue

        self._stats["total_tokens_collected"] += collection.total_draft_tokens
        for strategy, count in collection.strategy_counts.items():
            self._stats["strategy_breakdown"][strategy] = (
                self._stats["strategy_breakdown"].get(strategy, 0) + count
            )

        return collection

    def _collect_ngram_draft(
        self,
        req: Request,
        proposer: NgramProposer,
    ) -> list[int] | None:
        """Collect N-gram draft tokens for a request."""
        try:
            prompt_ids = req.prompt_token_ids or []
            all_ids = prompt_ids + list(req.output_token_ids)
            if len(all_ids) < proposer.config.min_n:
                return None
            return proposer.propose(all_ids)
        except Exception:
            logger.debug("spec proposer failed", exc_info=True)
            return None

    def _collect_mtp_draft(
        self,
        req: Request,
        mtp_decoder: Any,
    ) -> list[int] | None:
        """Collect MTP draft tokens for a request (lightweight, no GPU)."""
        try:
            # MTP in batch collection mode: return None to defer to per-request
            # _try_mtp_draft which has access to the model for the forward pass.
            # Batch collection for MTP is a placeholder — actual MTP requires
            # a model forward pass which is too expensive for batch collection.
            return None
        except Exception:
            logger.debug("MTP draft collection failed", exc_info=True)
            return None

    def _collect_cross_model_draft(
        self,
        req: Request,
        decoder: SpeculativeDecoder,
    ) -> list[int] | None:
        """Collect cross-model draft tokens (deferred to per-request path)."""
        # Cross-model drafting requires running the draft model which is
        # expensive. Defer to per-request _try_cross_model_draft.
        return None

    def get_stats(self) -> dict:
        """Return collection statistics."""
        stats = dict(self._stats)
        return stats


class AttentionScoreTracker:
    """Tracks cumulative attention scores per KV block for H2O-style eviction.

    H2O (Heavy-Hitter Oracle) eviction: instead of evicting
    the least-recently-used block, evict the block with the lowest cumulative
    attention score. This preserves blocks that the model "pays attention to".

    Since MLX does not expose per-head attention weights directly from
    BatchGenerator, this tracker uses a recency-based heuristic:
    recent blocks receive exponentially higher scores than older blocks.
    When real attention weights become available (e.g., via custom Metal
    kernels), update_scores() can accept them directly.

    The tracker is optional and disabled by default. Enable via
    SchedulerConfig.enable_attention_eviction = True.
    """

    def __init__(self, max_blocks_per_request: int = 256) -> None:
        self._scores: dict[str, dict[int, float]] = {}  # req_id -> {block_idx: score}
        self._max_blocks = max_blocks_per_request
        self._total_heuristic_updates: int = 0
        self._total_real_updates: int = 0

    def register_request(self, request_id: str) -> None:
        """Register a new request for attention score tracking."""
        self._scores[request_id] = {}

    def update_scores(self, request_id: str, block_scores: dict[int, float]) -> None:
        """Update cumulative attention scores for a request's KV blocks.

        Args:
            request_id: The tracked request.
            block_scores: Mapping of block index to incremental score.
        """
        if request_id not in self._scores:
            return
        scores = self._scores[request_id]
        for idx, score in block_scores.items():
            scores[idx] = scores.get(idx, 0.0) + score

    def update_heuristic(self, request_id: str, num_blocks: int) -> None:
        """Apply recency-based heuristic scores (exponential decay).

        Recent blocks (higher index) receive higher scores. This is a
        lightweight proxy for real attention weights.

        Args:
            request_id: The tracked request.
            num_blocks: Current total number of KV blocks for this request.
        """
        if request_id not in self._scores or num_blocks <= 0:
            return
        self._total_heuristic_updates += 1
        scores = self._scores[request_id]
        # Exponential decay: most recent block gets score 1.0, older blocks
        # decay by factor of 0.95 per block. This means block i (0-indexed
        # from oldest) gets score 0.95^(num_blocks - 1 - i).
        decay = 0.95
        for i in range(num_blocks):
            score = decay ** (num_blocks - 1 - i)
            # Exponential moving average: decay old score, add new score.
            # This gives recency-weighted scores instead of cumulative sums,
            # which inverted eviction order for long-running requests (older
            # blocks accumulated the highest scores over time).
            scores[i] = scores.get(i, 0.0) * decay + score
            # Cap per-block score to prevent unbounded growth
            if scores[i] > 1000.0:
                scores[i] = 1000.0

    def get_eviction_order(self, request_id: str) -> list[int]:
        """Return block indices sorted by ascending attention score (worst first).

        Blocks with the lowest cumulative score should be evicted first,
        as the model "pays least attention" to them.
        """
        scores = self._scores.get(request_id, {})
        if not scores:
            return []
        return sorted(scores.keys(), key=lambda i: scores[i])

    def remove_request(self, request_id: str) -> None:
        """Remove all tracking data for a finished/aborted request."""
        self._scores.pop(request_id, None)

    def get_stats(self) -> dict:
        """Return tracker statistics for monitoring."""
        return {
            "tracked_requests": len(self._scores),
            "total_blocks": sum(len(v) for v in self._scores.values()),
            "heuristic_updates": self._total_heuristic_updates,
            "real_updates": self._total_real_updates,
        }


class SchedulingPolicy(Enum):
    """Request scheduling policy (pluggable).

    To add a new policy:
    1. Add a new member here (e.g. MY_POLICY = auto()).
    2. Handle it in ``make_waiting_queue()`` (priority_queue.py) if the
       queue ordering needs to change.
    3. Add a branch in ``_schedule_waiting()`` below where the policy is
       checked (search for ``self.config.policy``).
    4. Add a test in ``tests/unit/test_scheduler.py``.
    """

    FCFS = auto()  # First-Come-First-Served
    PRIORITY = auto()  # Priority-based (higher priority = scheduled first)
    FAIR = (
        auto()
    )  # Round-robin across priority levels (prevents low-priority starvation)


@dataclass
class SchedulerConfig:
    """Scheduler tuning parameters."""

    model_name: str = ""
    completion_batch_size: int = 32
    prefill_batch_size: int = 8
    prefill_step_size: int = 2048
    max_kv_size: int | None = None
    deferred_clear_delay: int = 8
    cache_cleanup_interval: int = 512
    stream_interval: int = 1
    step_interval: float = 0.001
    policy: SchedulingPolicy = SchedulingPolicy.FCFS
    max_num_seqs: int = 256
    prefill_chunk_size: int = 2048
    request_timeout_seconds: float = 300  # 5 min timeout for waiting requests
    max_waiting_requests: int = (
        1024  # Backpressure: reject new requests when queue is full
    )
    memory_guard_enabled: bool = True  # Preflight memory check before scheduling
    memory_guard_soft_limit: float = (
        0.85  # Warn when active memory exceeds this fraction of total
    )
    # Sarathi-style hybrid chunked prefill (interleave prefill chunks with decode)
    hybrid_chunk_size: int = 512  # Tokens per prefill chunk when interleaving
    enable_hybrid_prefill: bool = False  # Enable chunked prefill+decode interleaving
    # Chunked prefill production hardening
    chunked_prefill_budget: int = 4  # Max chunks per scheduling round (fairness)
    chunked_prefill_timeout_seconds: float = (
        30.0  # Per-request prefill timeout (0 = no timeout)
    )
    chunked_prefill_abort_on_timeout: bool = (
        True  # Abort request on timeout (vs force-feed)
    )
    # Concurrent partial prefill control (GAP 1.3)
    max_num_partial_prefills: int = (
        1  # Max partial (chunked) prefills in-flight at once
    )
    max_long_partial_prefills: int = 1  # Max long partial prefills in-flight at once
    long_prefill_token_threshold: int = (
        4096  # Token count above which a prefill is "long"
    )
    # Request retraction (C14)
    enable_retraction: bool = (
        True  # Temporarily evict decode for prefill under pressure
    )
    retraction_memory_threshold: float = (
        0.90  # Retract when memory utilization exceeds this
    )
    retraction_max_count: int = 4  # Max decode requests to retract per step
    # Speculative decoding (Phase 4)
    enable_spec_decode: bool = False  # Enable speculative decoding
    draft_model: str = ""  # Draft model name or path (empty = auto-detect from target)
    spec_draft_length: int = 5  # Number of draft tokens per step (K)
    # N-gram speculative decoding (model-free, always available)
    ngram_spec_enabled: bool = False  # Enable N-gram speculative decoding in batch path
    ngram_spec_min_n: int = 1  # Min ngram length
    ngram_spec_max_n: int = 5  # Max ngram length
    ngram_spec_k: int = 5  # Draft tokens per step
    ngram_spec_mode: str = "lps"  # Proposer mode: lps, hashpool, lcg
    # Batch-path SpecPrefill (sparse prefill for long prompts)
    batch_spec_prefill_enabled: bool = False  # Enable via YUNSHU_BATCH_SPEC_PREFILL=1
    batch_spec_prefill_threshold: int = 8192  # Min prompt length to trigger
    batch_spec_prefill_keep_rate: float = 0.20  # Fraction of tokens to keep
    # Spec-aware batch scheduling
    spec_overhead_per_request: float = 0.1  # Slot overhead per spec-active request
    # SCHED-3: Starvation prevention aging
    aging_weight: float = (
        0.1  # Age bonus per second in waiting queue (higher = less starvation)
    )
    aging_enabled: bool = True  # Enable/disable aging in scheduling
    # H2O attention-score-based eviction
    enable_attention_eviction: bool = (
        False  # Enable attention score tracking for smarter KV eviction
    )
    attention_eviction_max_blocks: int = 256  # Max KV blocks tracked per request


class _LogitsProcessorSampler:
    """Wraps a sampler with logits processors (repetition/presence/frequency penalty).

    mlx-lm's make_sampler() does not accept repetition_penalty, presence_penalty,
    or frequency_penalty. Instead, make_logits_processors() returns processors that
    take (tokens: list[int], logits: mx.array) and return modified logits.

    This wrapper stores generated token IDs per-request and applies the processors
    before the base sampler. The BatchGenerator calls the sampler as sampler(logits),
    so we intercept and apply processors first.
    """

    def __init__(self, base_sampler, logits_processors, prompt_token_ids=None):
        self._base_sampler = base_sampler
        self._logits_processors = logits_processors
        self._tokens: list[int] = list(prompt_token_ids) if prompt_token_ids else []

    def __call__(self, logits):
        # Apply logits processors: each takes (tokens, logits) -> logits
        for proc in self._logits_processors:
            logits = proc(self._tokens, logits)
        token = self._base_sampler(logits)
        # Track token for subsequent calls
        try:
            tid = token.item() if hasattr(token, "item") else int(token)
            self._tokens.append(tid)
        except Exception:
            logger.debug("failed", exc_info=True)
        return token

    def reset(self):
        self._tokens.clear()


class Scheduler:
    """MLX-native continuous batching scheduler.

    Wraps mlx-lm's BatchGenerator with request lifecycle management.
    All GPU work runs on the shared MLX executor thread (not here directly).

    Key behaviors:
    - Deep-copy tokenizer to avoid Rust RefCell races
    - Per-request detokenizer (never pool)
    - Deferred cache clearing with 8-step delay
    - Thread-safe abort via pending set
    - ServerMetrics integration for dashboard
    - PrefillProgressTracker for live prefill progress
    """

    _DEFERRED_CLEAR_DELAY = 8

    def __init__(
        self, model: Any, tokenizer: Any, config: SchedulerConfig | None = None
    ):
        self.model = model
        self.tokenizer = copy.deepcopy(tokenizer)
        self.config = config or SchedulerConfig()
        self.model_id: str = config.model_name if config else ""

        # Request queues
        self.waiting: RequestPriorityQueue[Request] = make_waiting_queue(
            self.config.policy
        )
        self.running: dict[str, Request] = {}
        self.requests: dict[str, Request] = {}
        self.finished_ids: set[str] = set()

        # Thread-safe abort (CPython GIL guarantees set.add atomicity)
        self._pending_abort_ids: set[str] = set()

        # UIDs to remove from BatchGenerator (thinking budget overflow, etc.)
        self._uids_to_remove: list[int] = []

        # Requests that failed to insert (error outputs generated in step())
        self._failed_insert_ids: list[str] = []

        # BatchGenerator integration
        self._batch_gen = None
        self._uid_to_req: dict[int, str] = {}
        # uids whose prompt-prefix KV has already been saved
        # into the prefix cache, so we save each request's prefix exactly once
        # (during generation, while the sequence is still in the BatchGenerator).
        self._saved_prefix_uids: set[int] = set()

        # Per-request detokenizers (never pool)
        self._detokenizers: dict[str, Any] = {}

        # Sarathi-style chunked prefill: tracks partially-prefilled requests
        # Maps request_id → {'remaining_tokens': list[int], 'batch_uid': int | None}
        self._pending_prefill: dict[str, dict] = {}

        # Per-request thinking budget processors
        self._thinking_processors: dict[str, Any] = {}

        # KV prefix cache for batch-path insert_segments (C16)
        self._prefix_cache: Any = None

        # Cache-locality request reordering
        # Maps request_id → first KV block hash. Requests with the same hash
        # share a prefix and are scheduled consecutively for better cache locality.
        self._kv_prefix_hashes: dict[str, int] = {}

        # Deferred cache clearing
        self._step_counter: int = 0
        self._deferred_clear_at: int | None = None

        # Stats
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._num_requests: int = 0

        # External integrations (set by EngineCore)
        self._server_metrics: Any | None = None
        self._prefill_tracker: Any | None = None

        self._memory_monitor: Any | None = None

        # Thinking-segment KV substore for reasoning cache reuse
        self._thinking_store = ThinkingSegmentSubstore(ThinkingSegmentConfig())

        # Memory guard consecutive deferral counter — prevents spin-loop
        # where the scheduler repeatedly defers all waiting requests but
        # no progress is made (e.g., memory pressure from stale KV cache
        # that deferred clear hasn't reclaimed). After MAX_CONSECUTIVE_DEFERRALS
        # consecutive deferrals, reject the oldest waiting request instead of
        # deferring again, forcing the queue to drain.
        self._mem_guard_defer_count: int = 0
        self._MAX_CONSECUTIVE_DEFERRALS = 5

        # Speculative decoding (Phase 4: EAGLE-3 single-request path)
        self._spec_decoder: Any | None = None
        self._spec_head_info: Any | None = None  # SpecHeadInfo from detect_spec_heads()

        # MTP decoder for batch-path speculative decoding (self-speculative,
        # uses model's own prediction heads — no external draft model needed)
        self._mtp_decoder: Any | None = None

        # N-gram speculative decoding (model-free, available in batch path)
        self._ngram_proposer: NgramProposer | None = None
        if self.config.ngram_spec_enabled:
            # GPU-accelerated N-gram proposer (opt-in via YUNSHU_GPU_NGRAM=1)
            if os.environ.get("YUNSHU_GPU_NGRAM", "").strip() in ("1", "true", "yes"):
                try:
                    from .gpu_ngram import GPUNgramConfig, GPUNgramProposer

                    gpu_config = GPUNgramConfig(
                        min_n=self.config.ngram_spec_min_n,
                        max_n=self.config.ngram_spec_max_n,
                        k=self.config.ngram_spec_k,
                        max_model_len=self.config.max_kv_size or 32768,
                        gpu_fallback=True,
                    )
                    self._ngram_proposer = GPUNgramProposer(gpu_config)
                    logger.info(
                        "GPU-accelerated N-gram proposer enabled (YUNSHU_GPU_NGRAM=1)"
                    )
                except Exception:
                    logger.debug(
                        "GPU N-gram init failed, falling back to CPU", exc_info=True
                    )
                    self._ngram_proposer = NgramProposer(
                        NgramConfig(
                            min_n=self.config.ngram_spec_min_n,
                            max_n=self.config.ngram_spec_max_n,
                            k=self.config.ngram_spec_k,
                            mode=self.config.ngram_spec_mode,
                            max_model_len=self.config.max_kv_size or 32768,
                        )
                    )
            else:
                self._ngram_proposer = NgramProposer(
                    NgramConfig(
                        min_n=self.config.ngram_spec_min_n,
                        max_n=self.config.ngram_spec_max_n,
                        k=self.config.ngram_spec_k,
                        mode=self.config.ngram_spec_mode,
                        max_model_len=self.config.max_kv_size or 32768,
                    )
                )

        # Speculative decoding — batch-path draft/verify state
        # Maps request_id → list[int] of draft token IDs from the spec decoder.
        # Drafts are generated after a decode step and verified against the
        # target model's output on the next step (verify-after).
        self._spec_drafts: dict[str, list[int]] = {}
        # Bug 2 fix: track the position in output_token_ids where drafts were proposed,
        # so verification compares against the correct token range.
        self._spec_draft_start_pos: dict[str, int] = {}

        # Per-request spec decode statistics
        self._spec_stats: dict[
            str, dict[str, int]
        ] = {}  # req_id → {proposals, accepted, rejected}

        # Aggregate spec decode counters for get_stats()
        self._spec_total_proposals: int = 0
        self._spec_total_accepted: int = 0
        self._spec_total_rejected: int = 0

        # Pre-draft cache snapshots for spec decode rollback.
        # Keyed by request_id. Snapshot is taken BEFORE draft generation
        # so _verify_spec_drafts can restore to the correct state on rejection.
        self._spec_draft_cache_snapshots: dict[str, list] = {}

        # Per-request thinking state tracking for segment store
        # Maps request_id → dict with:
        # 'in_thinking': bool — currently in reasoning state
        # 'thinking_start_idx': int | None — output_token_ids index where thinking began
        # 'was_in_thinking': bool — previous step's thinking state (for transition detection)
        self._thinking_state: dict[str, dict] = {}

        # mRoPE batch delta manager
        from .mrope import BatchRopeDeltaManager

        self._rope_delta_mgr = BatchRopeDeltaManager()
        self._last_batch_rope_deltas: list[tuple[int, float]] = []

        # Encoder-decoder cache
        from .encoder_cache import EncoderCacheManager

        self._encoder_cache = EncoderCacheManager()

        # Hybrid KV cache for Mamba/hybrid models (layer-type-aware routing)
        # Set by EngineCore when model has mixed attention + SSM layers.
        self._hybrid_kv: Any | None = None

        # ITL tracking (C2/ITL-1: inter-token latency per request)
        self._last_token_time: dict[str, float] = {}
        self._itl_samples: dict[str, list[float]] = {}

        # H2O attention-score-based eviction
        self._attention_score_tracker: AttentionScoreTracker | None = None
        if self.config.enable_attention_eviction:
            self._attention_score_tracker = AttentionScoreTracker(
                max_blocks_per_request=self.config.attention_eviction_max_blocks,
            )
            logger.info("Attention-score-based eviction (H2O) enabled")

        # Chunked prefill production counters
        self._chunked_prefill_chunks_processed: int = 0
        self._chunked_prefill_fairness: dict[str, int] = {}  # req_id -> chunks served
        self._chunked_prefill_enqueued_at: dict[
            str, float
        ] = {}  # req_id -> time.monotonic()
        self._chunked_prefill_budget_used: int = (
            0  # Chunks consumed this scheduling round
        )
        self._chunked_prefill_failed_ids: list[
            str
        ] = []  # Requests that failed during chunked prefill
        # Chunked prefill progress outputs — synthetic RequestOutputs
        # carrying (processed, total) progress, emitted each step during chunked prefill.
        self._prefill_progress_outputs: list = []

        # Concurrent partial prefill control (GAP 1.3)
        self._active_partial_prefills: int = 0  # Count of in-flight partial prefills

        # Batch-path SpecPrefill (attention-based sparse prefill for long prompts)
        import os as _os

        self._batch_spec_prefill: BatchPathSpecPrefill | None = None
        if self.config.batch_spec_prefill_enabled or _os.environ.get(
            "YUNSHU_BATCH_SPEC_PREFILL", ""
        ).strip() in ("1", "true", "yes"):
            self._batch_spec_prefill = BatchPathSpecPrefill(
                BatchSpecPrefillConfig(
                    enabled=True,
                    threshold=self.config.batch_spec_prefill_threshold
                    or int(
                        _os.environ.get("YUNSHU_BATCH_SPEC_PREFILL_THRESHOLD", "8192")
                    ),
                    keep_rate=self.config.batch_spec_prefill_keep_rate
                    or float(
                        _os.environ.get("YUNSHU_BATCH_SPEC_PREFILL_KEEP_RATE", "0.20")
                    ),
                )
            )
            logger.info(
                f"Batch SpecPrefill enabled: threshold={self.config.batch_spec_prefill_threshold}, "
                f"keep_rate={self.config.batch_spec_prefill_keep_rate}"
            )

        # Spec-aware batch scheduler (slot allocation with spec overhead).
        # use the effective decode cap (min of max_num_seqs and
        # completion_batch_size), not the looser max_num_seqs, so spec slot
        # budgeting is computed against the real concurrent-decode capacity.
        self._spec_aware_scheduler = SpecAwareBatchScheduler(
            max_num_seqs=min(
                self.config.max_num_seqs, self.config.completion_batch_size
            ),
            spec_overhead_per_request=self.config.spec_overhead_per_request,
        )

        # Batched draft collection (collect drafts from all strategies for all running)
        self._draft_collector = BatchedDraftCollection()

        # FAIR policy: rotate starting priority level each scheduling step so
        # low-priority requests are not starved when available_slots is small.
        self._fair_rr_offset: int = 0
        # Preemption cascade cooldown: requests reinserted within this many
        # monotonic seconds are immune from preemption (prevents immediate
        # re-preemption that causes cascade feedback loops).
        self._preemption_cooldown_seconds: float = 0.5

        # Batch composer (ScheduleBatch → ForwardBatch)
        from .forward_batch import BatchComposer

        self._batch_composer = BatchComposer(
            max_batch_size=self.config.max_num_seqs,
            max_prefill_slots=self.config.prefill_batch_size
            if hasattr(self.config, "prefill_batch_size")
            else 8,
            max_decode_slots=self.config.max_num_seqs,
        )

        # Chunked prefill optimizer (semantic chunk boundary selection)
        if self.config.enable_hybrid_prefill:
            try:
                from .kv_optimizations import ChunkedPrefillOptimizer

                self._chunked_prefill_optimizer = ChunkedPrefillOptimizer()
                logger.info("ChunkedPrefillOptimizer wired (hybrid prefill)")
            except Exception:
                logger.debug("ChunkedPrefillOptimizer init skipped", exc_info=True)
                self._chunked_prefill_optimizer = None
        else:
            self._chunked_prefill_optimizer = None

    @property
    def _effective_max_seqs(self) -> int:
        """Real concurrent-decode admission cap.

        mlx-lm's BatchGenerator only DECODES ``completion_batch_size`` sequences at
        once — its ``_next()`` early-returns once the active batch reaches that size and
        the rest sit in mlx-lm's internal ``_unprocessed_sequences`` FIFO deque. Admitting
        up to ``max_num_seqs`` (default 256, 8× larger) therefore (a) silently defeats our
        PRIORITY/FAIR/aging policy — mlx-lm's blind FIFO, not our waiting-queue ordering,
        decides which of the surplus actually decode — and (b) prefills and holds the KV
        of all admitted sequences while only ``completion_batch_size`` decode, the OOM
        surface on a small Mac. So the true admission limit is the smaller of the two; the
        surplus stays in OUR waiting queue where policy/aging apply.
        """
        return min(self.config.max_num_seqs, self.config.completion_batch_size)

    def _init_batch_generator(self) -> None:
        """Create BatchGenerator on first use (lazy init)."""
        if self._batch_gen is not None:
            return

        from mlx_lm.generate import BatchGenerator, generation_stream
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=1.0)
        # mlx-lm's BatchGenerator chunks a long prompt NATIVELY by
        # prefill_step_size — it processes prefill_step_size tokens per next() per
        # sequence (interleaving with decode of other sequences) and only DECODES once
        # a sequence collapses to its final 1-token segment. That is correct + spurious-
        # token-free. So we honor the intended chunk size here and DROP Yunshu's buggy
        # manual cross-step chunking below (it fed segments=[[chunk]] per step, making
        # each chunk a complete "prompt" that decoded a spurious token AND corrupted the
        # carried KV → confidently-wrong output, verified via teacher-forcing).
        _eff_step = self.config.prefill_step_size
        # Honor the intended chunk granularity so mlx-lm's native chunking matches what
        # the manual chunking used to do (hybrid/Sarathi → hybrid_chunk_size, else
        # prefill_chunk_size).
        _target_chunk = (
            self.config.hybrid_chunk_size
            if self.config.enable_hybrid_prefill
            else getattr(self.config, "prefill_chunk_size", 0)
        )
        if _target_chunk and _target_chunk > 0:
            _eff_step = (
                min(_eff_step, _target_chunk) if _eff_step > 0 else _target_chunk
            )
        self._batch_gen = BatchGenerator(
            self.model,
            max_tokens=self.config.completion_batch_size,
            sampler=sampler,
            prefill_batch_size=self.config.prefill_batch_size,
            completion_batch_size=self.config.completion_batch_size,
            prefill_step_size=_eff_step,
            max_kv_size=self.config.max_kv_size,
            stream=generation_stream,
        )
        logger.info("BatchGenerator initialized")

    def shutdown(self) -> None:
        """Shutdown scheduler and release BatchGenerator resources.

        Gracefully drains any pending chunked prefills by marking them as
        errors before resetting, so clients receive termination signals.
        """
        # Gracefully drain pending chunked prefills — mark as errors
        if self._pending_prefill:
            n_pending = len(self._pending_prefill)
            for req_id in list(self._pending_prefill.keys()):
                req = self.running.get(req_id) or self.requests.get(req_id)
                if req is not None:
                    req.set_finished(RequestStatus.FINISHED_ERROR, reason="shutdown")
            logger.info(f"Shutdown: draining {n_pending} pending chunked prefills")
        self.deep_reset()
        if self._batch_gen is not None:
            if hasattr(self._batch_gen, "close"):
                try:
                    self._batch_gen.close()
                except Exception:
                    logger.debug("failed", exc_info=True)
            self._batch_gen = None
        logger.info("Scheduler shutdown complete")

    def set_server_metrics(self, metrics: Any) -> None:
        self._server_metrics = metrics

    def set_prefill_tracker(self, tracker: Any) -> None:
        self._prefill_tracker = tracker

    def set_memory_monitor(self, monitor: Any) -> None:
        """Set memory monitor for external preflight checks."""
        self._memory_monitor = monitor

    def set_prefix_cache(self, cache: Any) -> None:
        """Set KV prefix cache for batch-path cache hits (C16: insert_segments support)."""
        self._prefix_cache = cache

    def _save_active_prefixes(self) -> None:
        """Save each active request's prompt-prefix KV into the prefix cache once.

        The engine loop matched prefixes (radix 95%) but never
        skipped prefill because the KVPrefixCache that drives the insert-time
        skip (``insert_segments(caches=[cached_kv])``) was empty — it was only
        populated on preemption. A FINISHED request's sequence is auto-dropped
        from the BatchGenerator, so we must extract its KV WHILE it is still
        generating. For each request that has produced >=1 token (prefill done,
        still active) and isn't saved yet, pull its cache via
        ``BatchGenerator.extract_cache`` and ``_prefix_cache.add`` it (add() trims
        the generated tail back to prompt length). Best-effort and per-request
        once — never raises into the step loop.
        """
        if (
            self._prefix_cache is None
            or self._batch_gen is None
            or not self._uid_to_req
        ):
            return
        for uid, req_id in list(self._uid_to_req.items()):
            self._save_one_prefix(uid, req_id)

    def _save_one_prefix(self, uid, req_id) -> None:
        """Save ONE request's prompt-prefix KV into the prefix cache (once), while
        its KV is still in the BatchGenerator. Called both from _save_active_prefixes
        (for long, still-generating donors) AND at the FINISH point in
        _process_responses BEFORE the uid is popped from _uid_to_req — otherwise a
        SHORT request finishes, its uid is dropped from _uid_to_req, and the
        end-of-step _save_active_prefixes can't see it → its prefix is never cached
        (the original fix called _save_active_prefixes in the decode sub-loop,
        but that still iterates _uid_to_req which _process_responses had already pruned
        → ineffective for finished requests; this per-uid save at the finish point is
        the correct hook)."""
        pc = self._prefix_cache
        bg = self._batch_gen
        if pc is None or bg is None or uid in self._saved_prefix_uids:
            return
        req = self.running.get(req_id)
        if req is None:
            return
        # Need prefill done (>=1 generated token) and a worthwhile prefix.
        if (
            req.num_output_tokens < 1
            or req.num_prompt_tokens < 32
            or not req.prompt_token_ids
        ):
            return
        # CRITICAL: only save requests that did a FULL prefill. A request that
        # warm-started from an external cached prefix keeps the reused KV in the shared
        # cached_kv ref, NOT in its own BatchGenerator cache — extract_cache would then
        # return a cache SHORTER than the prompt, and storing that creates a corrupt entry
        # (claims more tokens than it has KV for) that makes the next reuse fail to insert.
        # gate on the EXACT external-reuse flag set at insert time, NOT on
        # req.cached_tokens — the paged scheduler sets cached_tokens from the radix
        # accounting match, which is >0 even when the KVPrefixCache missed and a full
        # prefill ran, so the old gate permanently skipped saving those prefixes.
        if getattr(req, "_external_kv_reuse", False):
            self._saved_prefix_uids.add(uid)
            return
        try:
            import mlx.core as mx

            extracted = bg.extract_cache([uid])
            ct = extracted.get(uid) if extracted else None
            cache_data = ct[0] if isinstance(ct, (tuple, list)) and ct else None
            if cache_data:
                pc.add(mx.array(req.prompt_token_ids), cache_data)
            # Mark saved even on empty extract so we don't retry every step.
            self._saved_prefix_uids.add(uid)
        except Exception:
            logger.debug("prefix save failed for uid %s", uid, exc_info=True)
            self._saved_prefix_uids.add(uid)

    def set_hybrid_kv_cache(self, hybrid_kv: Any) -> None:
        """Set HybridKVCache for layer-type-aware KV management.

        When the model has mixed attention + SSM layers (Mamba, Jamba, etc.),
        this allows the scheduler to route allocate/free to the correct pool.
        """
        self._hybrid_kv = hybrid_kv

    def set_kv_prefix_hash(self, request_id: str, prefix_hash: int) -> None:
        """Store a request's KV prefix hash for cache-locality reordering.

        Called by EngineCore after computing the prefix hash for a new request.
        The hash is derived from the first complete KV block of the prompt,
        which approximates the system prompt / conversation prefix.
        """
        self._kv_prefix_hashes[request_id] = prefix_hash

    def _reorder_by_cache_locality(self, requests: list) -> list:
        """Sort requests by KV prefix hash for better cache utilization.

        Groups requests sharing the same KV prefix hash (i.e., same system
        prompt / conversation prefix) so they are inserted consecutively into
        the BatchGenerator. This maximizes KV cache block locality during
        prefill and decode, reducing cache thrashing.

        IMPORTANT: Groups are ordered by the highest effective priority (i.e.
        the first element's priority, since *requests* already arrives sorted
        by effective priority from SCHED-3). Within each group the original
        priority order is preserved. This prevents a high-priority aged request
        with a unique prefix from being pushed behind low-priority requests
        that share a common prefix.

        Args:
            requests: List of Request objects to sort.

        Returns:
            Reordered list. No-op when len <= 1. Stable sort preserves
            insertion order within each prefix group.
        """
        if len(requests) <= 1:
            return requests

        # Group by KV prefix hash
        prefix_groups: dict[int, list] = {}
        no_prefix: list = []

        for req in requests:
            rid = req.request_id
            prefix_hash = self._kv_prefix_hashes.get(rid)
            if prefix_hash is not None:
                prefix_groups.setdefault(prefix_hash, []).append(req)
            else:
                no_prefix.append(req)

        # Emit groups ordered by highest effective priority within each group.
        # *requests* arrives pre-sorted by effective priority (SCHED-3), so the
        # minimum index in a group corresponds to the highest-priority request.
        # We use the minimum original index as the sort key for groups so that
        # a group containing a high-priority request is emitted first.
        group_min_index: dict[int, int] = {}
        for idx, req in enumerate(requests):
            prefix_hash = self._kv_prefix_hashes.get(req.request_id)
            if prefix_hash is not None:
                prev = group_min_index.get(prefix_hash)
                if prev is None or idx < prev:
                    group_min_index[prefix_hash] = idx

        sorted_group_keys = sorted(
            prefix_groups.keys(),
            key=lambda h: group_min_index.get(h, 0),
        )

        # No-prefix requests keep their original relative order; they come
        # after all grouped requests (lower cache locality benefit).
        no_prefix_min_index = len(requests)  # sentinel — always after groups
        for idx, req in enumerate(requests):
            if self._kv_prefix_hashes.get(req.request_id) is None:
                no_prefix_min_index = idx
                break

        result: list = []
        if no_prefix_min_index < len(requests):
            # Interleave: emit groups and no-prefix in priority order
            # Merge sorted groups with no_prefix based on their min index
            all_segments: list[tuple[int, list]] = []
            for h in sorted_group_keys:
                all_segments.append((group_min_index[h], prefix_groups[h]))
            all_segments.append((no_prefix_min_index, no_prefix))
            all_segments.sort(key=lambda seg: seg[0])
            for _, seg_requests in all_segments:
                result.extend(seg_requests)
        else:
            # No no-prefix requests — just emit groups in priority order
            for h in sorted_group_keys:
                result.extend(prefix_groups[h])

        return result

    def add_request(self, request: Request) -> None:
        """Add request to waiting queue (called from event loop thread)."""
        if len(self.waiting) >= self.config.max_waiting_requests:
            request.set_finished(RequestStatus.FINISHED_ERROR, reason="queue_full")
            logger.warning(
                f"Rejecting request {request.request_id}: waiting queue full "
                f"({len(self.waiting)}/{self.config.max_waiting_requests})"
            )
            return
        request.status = RequestStatus.WAITING
        self.requests[request.request_id] = request
        import time as _time

        # SCHED-3: Record submit time for priority aging.
        # _submit_time is a proper field on Request (default 0.0).
        request._submit_time = _time.monotonic()
        self.waiting.push(request, priority=request.sampling_params.priority)

    def abort_request(self, request_id: str) -> bool:
        """Thread-safe abort (deferred to next step)."""
        self._pending_abort_ids.add(request_id)
        return request_id in self.requests

    def has_requests(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    def _has_active_requests(self) -> bool:
        return bool(self.running)

    def step(self) -> Any:
        """Run one scheduler step (called on MLX executor thread).

        Returns SchedulerOutput with response list.

        mlx-lm BatchGenerator API (0.31.3):
          next() → (prompt_responses, gen_responses) — handles prefill
          next_generated() → gen_responses — decode step, one token per request
        """
        self._init_batch_generator()

        # 0. Reset per-round budget counter
        self._chunked_prefill_budget_used = 0
        self._prefill_progress_outputs = []

        # 1. Process deferred aborts
        self._process_aborts()

        # 1b. Remove UIDs from BatchGenerator (thinking budget overflow, etc.)
        if self._uids_to_remove and self._batch_gen is not None:
            self._batch_gen.remove(self._uids_to_remove)
            self._uids_to_remove.clear()

        # 2. Insert waiting requests
        self._schedule_waiting()

        # 2b. Generate error outputs for requests that failed to insert.
        # These never reach the BatchGenerator, so they won't produce output
        # in step 3+. Without synthetic error outputs, EngineCore would never
        # call _finalize_request for them, leaking per-request resources.
        outputs = []
        if self._failed_insert_ids:
            for fail_id in self._failed_insert_ids:
                fail_req = self.requests.get(fail_id)
                actual_reason = getattr(fail_req, "finish_reason", None) or "error"
                outputs.append(
                    RequestOutput(
                        request_id=fail_id,
                        finished=True,
                        finish_reason=actual_reason,
                        error=f"Request {fail_id} failed to insert into batch generator",
                        prompt_tokens=getattr(fail_req, "num_prompt_tokens", 0)
                        if fail_req
                        else 0,
                        completion_tokens=0,
                    )
                )
            self._failed_insert_ids.clear()

        # 2c. Generate error outputs for requests that failed during chunked prefill.
        # These requests were partially prefilled but a chunk insertion failed or
        # timed out. They need error outputs so EngineCore can finalize them.
        if self._chunked_prefill_failed_ids:
            for fail_id in self._chunked_prefill_failed_ids:
                fail_req = self.requests.get(fail_id)
                outputs.append(
                    RequestOutput(
                        request_id=fail_id,
                        finished=True,
                        finish_reason=getattr(fail_req, "finish_reason", None)
                        or "error",
                        error=f"Request {fail_id} failed during chunked prefill",
                        prompt_tokens=getattr(fail_req, "num_prompt_tokens", 0)
                        if fail_req
                        else 0,
                        completion_tokens=0,
                    )
                )
            self._chunked_prefill_failed_ids.clear()

        if self._batch_gen is None:
            return SchedulerOutput(outputs=outputs)

        # 3. Run one BatchGenerator step (prefill + first decode)
        # Note: outputs may already contain error outputs from failed inserts (step 2b).
        if not self._has_active_requests() and not self._pending_prefill:
            return SchedulerOutput(outputs=outputs)
        try:
            prompt_responses, gen_responses = self._batch_gen.next()

            # 3a. Retrieve batch RoPE deltas for decode (mRoPE multimodal support)
            # Provides per-request mRoPE deltas aligned to UID order. Currently
            # stored for future use in multimodal batch decode; text-only requests
            # return 0.0 deltas.
            if self.running:
                try:
                    _uids = [
                        uid
                        for uid, rid in self._uid_to_req.items()
                        if rid in self.running
                    ]
                    if _uids:
                        _rope_deltas = self.get_batch_rope_deltas(_uids)
                        self._last_batch_rope_deltas = list(
                            zip(_uids, _rope_deltas, strict=False)
                        )
                except Exception:
                    logger.debug("batch rope deltas collection failed", exc_info=True)

            # 4. Process prompt responses (prefill completion)
            if prompt_responses:
                self._process_prefill_responses(prompt_responses)

            # 5. Process initial generation responses
            if gen_responses:
                outputs.extend(self._process_responses(gen_responses))

            # 6. Run additional decode steps to get more tokens per step
            for _ in range(self.config.stream_interval):
                if not self._has_active_requests():
                    break
                # Remove finished UIDs from BatchGenerator immediately so
                # the next decode step doesn't waste GPU on dead requests. (Their
                # prompt prefixes were already captured at the finish point in
                # _process_responses via _save_one_prefix, before the uid pop.)
                if self._uids_to_remove and self._batch_gen is not None:
                    self._batch_gen.remove(self._uids_to_remove)
                    self._uids_to_remove.clear()
                try:
                    gen_responses = self._batch_gen.next_generated()
                except StopIteration:
                    break
                except IndexError:
                    break
                if gen_responses:
                    new_outputs = self._process_responses(gen_responses)
                    outputs.extend(new_outputs)
                else:
                    break

            # 6a. Sarathi-style hybrid prefill: interleave remaining prefill
            # chunks with decode steps for better tail latency.
            # When enable_hybrid_prefill=True and there are pending partial
            # prefills, feed one chunk per iteration then decode, repeating
            # until all chunks are consumed or a limit is reached.
            if self.config.enable_hybrid_prefill and self._pending_prefill:
                outputs = self._hybrid_prefill_step(outputs)

            # 6b. Speculative decoding: verify drafts, then generate new ones
            # (verify-after — verify pending drafts against
            # target model output, then draft K tokens for next step)
            spec_decoder_active = self.config.enable_spec_decode and isinstance(
                self._spec_decoder, SpeculativeDecoder
            )
            mtp_active = (
                self.config.enable_spec_decode and self._mtp_decoder is not None
            )
            ngram_active = self._ngram_proposer is not None
            if spec_decoder_active or mtp_active or ngram_active:
                self._verify_spec_drafts(outputs)
                # Batched draft collection: collect drafts from all strategies
                # for all running requests in one pass, then merge into _spec_drafts.
                batch_drafts = self.collect_batch_drafts()
                # Count only GENUINELY-NEW proposals. collect_batch_drafts re-returns
                # drafts that are still pending from a prior step; a request that produced
                # no output this step keeps its pending draft and would be re-counted every
                # step → inflated proposals → understated acceptance rate. Count the same
                # set we actually store below (rid not already in _spec_drafts).
                if batch_drafts.drafts:
                    self._spec_total_proposals += sum(
                        len(t)
                        for rid, t in batch_drafts.drafts.items()
                        if rid not in self._spec_drafts
                    )
                for rid, tokens in batch_drafts.drafts.items():
                    if rid not in self._spec_drafts:
                        self._spec_drafts[rid] = tokens
                        # Bug 2 fix: record position where drafts start
                        req = self.running.get(rid)
                        if req is not None:
                            self._spec_draft_start_pos[rid] = len(
                                req.output_token_ids or []
                            )
                # Generate drafts for still-active requests (per-request fallback
                # for strategies not covered by batch collection, e.g. MTP/cross-model)
                for req_id in list(self.running.keys()):
                    req = self.running.get(req_id)
                    if req is not None and req.output_token_ids:
                        # Skip finished requests — they will be cleaned up by
                        # _cleanup_finished.  Generating drafts for them wastes
                        # GPU time (cross-model/MTP do forward passes) and the
                        # drafts will never be verified.
                        if RequestStatus.is_finished(req.status):
                            continue
                        if req_id not in self._spec_drafts:
                            self._try_spec_decode_draft(req)
        except Exception as e:
            logger.error(f"BatchGenerator step error: {e}", exc_info=True)
            from .exceptions import is_cache_corruption_error

            if is_cache_corruption_error(e):
                logger.warning("Cache corruption detected — resetting BatchGenerator")
                # deep_reset() clears running/waiting/requests but the old
                # code emitted NO outputs for them → every in-flight request's
                # finished event never fired → clients hang to request_timeout and
                # per-request resources leak (the whole batch at once). The
                # non-corruption branch already emits an error output per request;
                # mirror it here. Capture IDs BEFORE the reset wipes them.
                _lost = list(self.requests.keys())
                self.deep_reset()
                for _rid in _lost:
                    outputs.append(
                        RequestOutput(
                            request_id=_rid,
                            finished=True,
                            finish_reason="error",
                            error=f"Cache corruption reset: {e}",
                            prompt_tokens=0,
                            completion_tokens=0,
                        )
                    )
            else:
                # Non-corruption error: fail all running requests to prevent
                # them from being stuck in RUNNING forever.
                logger.warning(
                    "BatchGenerator step failed — failing %d running requests",
                    len(self.running),
                )
                failed = self.fail_all_requests()
                for _rid in failed:
                    outputs.append(
                        RequestOutput(
                            request_id=_rid,
                            finished=True,
                            finish_reason="error",
                            error=f"BatchGenerator step failed: {e}",
                            prompt_tokens=0,
                            completion_tokens=0,
                        )
                    )
            return SchedulerOutput(outputs=outputs)

        # 7. Step counter
        self._step_counter += 1

        # 8. Cleanup finished FIRST — removes finished requests from
        # self.running so that _maybe_clear_cache (below) sees the
        # up-to-date running count.  Without this reordering, all
        # requests could be finished but still in self.running,
        # preventing cache reclamation for one full step.
        # save each active request's prompt-prefix KV into the
        # prefix cache BEFORE _cleanup_finished — a finished sequence is
        # auto-dropped from the BatchGenerator, so the only window to extract its
        # KV is while it is still generating. This is the lever that lets the
        # next same-prefix request skip prefill.
        self._save_active_prefixes()
        self._cleanup_finished()

        # 7a. Deferred cache clearing (runs after cleanup so
        # not self.running is accurate)
        self._maybe_clear_cache()

        # 7b. Periodic memory pressure eviction (C12)
        if self._step_counter % 64 == 0 and self._memory_monitor is not None:
            self._maybe_evict_kv_cache()

        # 7d. Periodic encoder-decoder cache eviction
        # Evict expired encoder hidden-state entries to reclaim memory.
        if self._step_counter % 64 == 0:
            self._encoder_cache.evict_all_expired()

        # 9. Append prefill progress outputs
        # These synthetic RequestOutputs carry (processed, total) progress
        # during chunked prefill, enabling client-side progress bars.
        if self._prefill_progress_outputs:
            outputs.extend(self._prefill_progress_outputs)

        return SchedulerOutput(outputs=outputs)

    def _schedule_waiting(self) -> None:
        """Move waiting requests into BatchGenerator.

        Supports FCFS (default) and PRIORITY scheduling policies.
        PRIORITY scheduling sorts by request priority (higher = first).

        Request preemption: when max_num_seqs is reached
        and policy is PRIORITY, preempts the lowest-priority running
        request to make room for a higher-priority waiting request.
        Under FCFS, no preemption occurs (new requests wait).

        When enable_hybrid_prefill is True and there are active decode
        requests, inserts only hybrid_chunk_size tokens per step
        (Sarathi-style chunked prefill), interleaving prefill chunks
        with decode steps for better tail latency.
        """
        # First, process any pending partial prefills from previous steps
        self._process_pending_prefill()

        if not self.waiting or self._batch_gen is None:
            return

        now = time.monotonic()
        timeout = self.config.request_timeout_seconds
        to_insert = []
        while self.waiting:
            req = self.waiting.pop()
            if req.request_id in self._pending_abort_ids:
                self._pending_abort_ids.discard(req.request_id)
                req.set_finished(RequestStatus.FINISHED_ABORTED, reason="abort")
                self.finished_ids.add(req.request_id)
                self._failed_insert_ids.append(req.request_id)
                continue
            submit = req._submit_time if req._submit_time > 0 else now
            if timeout > 0 and (now - submit) > timeout:
                req.set_finished(RequestStatus.FINISHED_TIMEOUT, reason="timeout")
                self.finished_ids.add(req.request_id)
                # Track timed-out request so step() generates an error output
                # for EngineCore to finalize (otherwise resources leak).
                self._failed_insert_ids.append(req.request_id)
                logger.warning(
                    f"Request {req.request_id} timed out after {now - submit:.0f}s in waiting queue"
                )
                continue
            to_insert.append(req)

        # Heap order is by raw priority; we re-sort below with aging.

        # SCHED-3: Apply aging to prevent starvation of low-priority requests.
        #
        # Each request has a _submit_time (set when entering the
        # waiting queue).  The effective priority is:
        # effective_priority = raw_priority + (now - _submit_time) * aging_weight
        #
        # A request waiting 10s with aging_weight=0.1 gets +1.0 boost, enough
        # to overtake a request with priority 1 higher that just arrived.
        # This prevents indefinite starvation under sustained high-priority load.
        #
        # Since ALL items are popped from the heap before this sort, the aging
        # is applied to the full waiting queue contents (not just the visible
        # top).  Overflow requests pushed back via push_front retain their
        # _submit_time so aging accumulates correctly across scheduling rounds.
        if (
            self.config.aging_enabled
            and self.config.policy == SchedulingPolicy.PRIORITY
            and len(to_insert) > 1
        ):
            aging_weight = self.config.aging_weight
            _aged_insert = []
            for _req in to_insert:
                _submit = _req._submit_time if _req._submit_time > 0 else now
                _age = max(0.0, now - _submit)
                _effective_priority = (
                    _req.sampling_params.priority + _age * aging_weight
                )
                _aged_insert.append((_effective_priority, _req))
            _aged_insert.sort(key=lambda x: -x[0])  # Higher effective priority first
            to_insert = [_req for _, _req in _aged_insert]

        # Preempted-request priority: requests that were preempted (have
        # num_preemptions > 0) should always be scheduled before new requests
        # at the same effective priority level.  Placing this AFTER the aging
        # sort ensures preempted requests are not demoted by new requests that
        # accumulated aging bonus while the preempted request's _submit_time
        # was reset to now() during preemption.
        if to_insert:
            _preempted = [r for r in to_insert if getattr(r, "num_preemptions", 0) > 0]
            _new = [r for r in to_insert if getattr(r, "num_preemptions", 0) == 0]
            if _preempted and _new:
                to_insert = _preempted + _new

        # FAIR policy: round-robin across priority levels.
        #
        # Instead of serving all high-priority requests first (which can starve
        # low-priority ones indefinitely), serve one request from each priority
        # level in turn.  Higher-priority levels get more slots proportional to
        # how many requests they have, but every level gets at least one slot
        # per round (if it has requests waiting).
        #
        # This is simpler than aging and more predictable: no tuning of
        # aging_weight needed.  It guarantees bounded waiting time for any
        # priority level with at least one request.
        if self.config.policy == SchedulingPolicy.FAIR and len(to_insert) > 1:
            # Group requests by priority level
            buckets: dict[int, list] = {}
            for _req in to_insert:
                p = _req.sampling_params.priority if _req.sampling_params else 0
                buckets.setdefault(p, []).append(_req)
            # Sort priority levels descending (high priority goes first)
            sorted_priorities = sorted(buckets.keys(), reverse=True)
            # Sort each bucket by arrival_time (FIFO within priority level)
            # then rotate within each bucket so the same request is not always
            # first.  Without rotation, when available_slots < bucket size,
            # the first request by arrival_time is always scheduled while
            # later requests starve.
            for p in sorted_priorities:
                buckets[p].sort(key=lambda r: r.arrival_time)
                if len(buckets[p]) > 1:
                    offset = self._fair_rr_offset % len(buckets[p])
                    buckets[p] = buckets[p][offset:] + buckets[p][:offset]
            # Rotate starting priority level each step so that when
            # available_slots is small, low-priority levels eventually get
            # the first slot.  Without rotation, the highest-priority level
            # always gets the first slot and low-priority requests starve.
            if len(sorted_priorities) > 1 and self._fair_rr_offset > 0:
                offset = self._fair_rr_offset % len(sorted_priorities)
                sorted_priorities = (
                    sorted_priorities[offset:] + sorted_priorities[:offset]
                )
            self._fair_rr_offset += 1
            # Round-robin: take one from each priority level in turn
            round_robin = []
            while any(buckets[p] for p in sorted_priorities):
                for p in sorted_priorities:
                    if buckets[p]:
                        round_robin.append(buckets[p].pop(0))
            to_insert = round_robin

        # Cache-locality reordering: sort to_insert by KV prefix hash so
        # requests sharing the same system prompt / conversation prefix are
        # inserted into the BatchGenerator consecutively. This improves KV
        # cache block locality during prefill and decode.
        # Skip under FAIR policy to preserve round-robin ordering guarantees.
        if self.config.policy != SchedulingPolicy.FAIR:
            to_insert = self._reorder_by_cache_locality(to_insert)

        # Respect max_num_seqs limit — with preemption under PRIORITY policy
        active_count = len(self.running)

        # Spec-aware slot allocation: when speculative decoding consumes batch
        # capacity, reserve slots for draft verification overhead. N-gram drafting
        # is CPU-only pattern matching verified in the SAME forward pass — it costs
        # ZERO batch slots, so it must NOT trigger the ~10% slot reservation (which
        # would needlessly cap concurrency ~10% below capacity whenever it's enabled).
        # Only cross-model spec and MTP add a drafting/verification batch cost.
        has_spec = (
            self.config.enable_spec_decode
            and isinstance(self._spec_decoder, SpeculativeDecoder)
        ) or self._mtp_decoder is not None
        if has_spec and self._spec_aware_scheduler is not None:
            budget = self._spec_aware_scheduler.compute_spec_budget(active_count)
            available_slots = budget.available_for_new
        else:
            available_slots = max(0, self._effective_max_seqs - active_count)

        # Batch-path SpecPrefill: compute skippable tokens for long prompts
        # before insertion, reducing prefill time.
        if self._batch_spec_prefill is not None and to_insert:
            to_insert = self._apply_batch_spec_prefill(to_insert)

        if len(to_insert) > available_slots and self.config.policy in (
            SchedulingPolicy.PRIORITY,
            SchedulingPolicy.FAIR,
        ):
            # evict lowest-priority running requests
            # to make room for higher-priority waiting requests.
            # Cap per-step preemptions to prevent cascade: a burst of 200
            # waiting requests should not evict 198 running requests at once,
            # which would thrash KV cache and cause latency spikes for all
            # preempted requests needing re-prefill.
            to_preempt = min(
                len(to_insert) - available_slots,
                self._MAX_PREEMPTIONS_PER_STEP,
            )
            preempted = self._preempt_lowest_priority(to_preempt)
            if preempted > 0:
                # Recompute available_slots from scratch instead of naively
                # adding the raw preempted count.  When spec decode is active,
                # the spec overhead depends on the (now reduced) running count,
                # so the naive addition overestimates slots and can cause batch
                # overflow.
                new_active_count = len(self.running)
                if has_spec and self._spec_aware_scheduler is not None:
                    budget = self._spec_aware_scheduler.compute_spec_budget(
                        new_active_count
                    )
                    available_slots = budget.available_for_new
                else:
                    available_slots = max(
                        0, self._effective_max_seqs - new_active_count
                    )
                logger.info(
                    f"Preempted {preempted} running requests for {len(to_insert)} waiting "
                    f"(priority policy, available_slots={available_slots})"
                )

        if len(to_insert) > available_slots:
            # C14: Memory-pressure retraction
            # Under pressure, temporarily retract decode requests to make room for prefill
            if self.config.enable_retraction and self._memory_monitor is not None:
                try:
                    info = self._memory_monitor.get_memory_info()
                    if (
                        info.utilization_pct
                        >= self.config.retraction_memory_threshold * 100
                    ):
                        retracted = self._retract_decode_requests(
                            min(
                                self.config.retraction_max_count,
                                len(to_insert) - available_slots,
                            )
                        )
                        if retracted > 0:
                            # Recompute available_slots instead of naive addition
                            # (same spec-overhead fix as preemption path above).
                            new_active_count = len(self.running)
                            if has_spec and self._spec_aware_scheduler is not None:
                                budget = self._spec_aware_scheduler.compute_spec_budget(
                                    new_active_count
                                )
                                available_slots = budget.available_for_new
                            else:
                                available_slots = max(
                                    0, self._effective_max_seqs - new_active_count
                                )
                            logger.info(
                                f"Retracted {retracted} decode requests under memory pressure "
                                f"(util={info.utilization_pct:.1f}%, available_slots={available_slots})"
                            )
                except Exception:
                    logger.debug("retraction check failed", exc_info=True)

        if len(to_insert) > available_slots:
            overflow = to_insert[available_slots:]
            to_insert = to_insert[:available_slots]
            # Put overflow back at front of waiting queue
            for req in reversed(overflow):
                self.waiting.push_front(req, priority=req.sampling_params.priority)

        # Generation memory guard: defer scheduling under memory pressure
        # Use len(self.running) instead of the stale active_count captured
        # before preemption/retraction — preemption reduces the running count,
        # but the memory guard should check against the current state.
        # NOTE: The guard runs even when current_running_count == 0.  Under
        # memory pressure with no active decode (e.g., model weights + stale
        # KV cache from just-finished burst), inserting all waiting requests
        # at once risks OOM before the deferred cache clear fires.
        current_running_count = len(self.running)
        if self.config.memory_guard_enabled and to_insert:
            try:
                import mlx.core as mx

                active_mem = mx.get_active_memory()
                from .utils.hardware import get_hardware_info

                hw = get_hardware_info()
                total_mem = hw.total_memory_bytes
                soft_limit = int(total_mem * self.config.memory_guard_soft_limit)
                if active_mem > soft_limit:
                    self._mem_guard_defer_count += 1
                    # Spin-loop prevention: after N consecutive deferrals with no
                    # progress, reject the oldest waiting request to force the
                    # queue to drain. Without this, the scheduler can loop
                    # indefinitely deferring all requests while memory remains
                    # above the soft limit (e.g., stale KV cache that the deferred
                    # clear hasn't reclaimed yet).
                    if self._mem_guard_defer_count >= self._MAX_CONSECUTIVE_DEFERRALS:
                        victim = self.waiting.pop_lowest_priority()
                        if victim is not None:
                            self._failed_insert_ids.append(victim.request_id)
                            victim.set_finished(
                                RequestStatus.FINISHED_ERROR, reason="memory_limit"
                            )
                            logger.warning(
                                "Memory guard: rejecting request %s after %d deferrals "
                                "(%s active > %s soft limit)",
                                victim.request_id,
                                self._mem_guard_defer_count,
                                f"{active_mem / 1024**3:.1f}GB",
                                f"{soft_limit / 1024**3:.1f}GB",
                            )
                            self._mem_guard_defer_count = 0
                        # the in-hand to_insert requests were popped off
                        # `waiting` (line ~1456); the old code cleared them WITHOUT
                        # re-queuing or failing them → they vanished (no finished
                        # output ever emitted → client hangs to request_timeout +
                        # leaked per-request resources). Only the freshly-popped
                        # victim is rejected; push the rest back to retry next step
                        # (matches the deferral branch).
                        for req in reversed(to_insert):
                            self.waiting.push_front(
                                req, priority=req.sampling_params.priority
                            )
                        to_insert = []
                    else:
                        logger.debug(
                            f"Memory guard: deferring {len(to_insert)} requests "
                            f"({active_mem / 1024**3:.1f}GB active > {soft_limit / 1024**3:.1f}GB soft limit)"
                        )
                        for req in reversed(to_insert):
                            self.waiting.push_front(
                                req, priority=req.sampling_params.priority
                            )
                        to_insert = []
                    # When no requests are running, nothing will trigger memory
                    # release via request completion. Force an immediate cache
                    # clear so the next step can make progress.
                    if current_running_count == 0:
                        try:
                            mx.synchronize()
                            mx.clear_cache()
                            logger.debug(
                                "Memory guard: forced immediate cache clear "
                                "(no running requests to trigger deferred clear)"
                            )
                        except Exception:
                            logger.debug("forced cache clear failed", exc_info=True)
                else:
                    self._mem_guard_defer_count = 0
            except Exception:
                logger.debug("memory guard check failed in scheduling", exc_info=True)

        # Track batch composition via BatchComposer
        if to_insert:
            from .forward_batch import RequestSlot

            pending_slots = [
                RequestSlot(
                    request_id=req.request_id,
                    prompt_tokens=req.prompt_token_ids or [],
                    max_tokens=req.sampling_params.max_tokens
                    if req.sampling_params
                    else 512,
                    priority=req.sampling_params.priority if req.sampling_params else 0,
                    is_prefill=True,
                    num_prompt_tokens=len(req.prompt_token_ids or []),
                    cached_tokens=getattr(req, "cached_tokens", 0),
                    arrival_time=req._submit_time if req._submit_time > 0 else now,
                )
                for req in to_insert
            ]
            # Populate num_prompt_tokens and generated_tokens for active decode
            # slots so BatchComposer can compute accurate total_tokens and
            # ForwardBatch.from_schedule_batch can compute correct position IDs.
            # Without these fields, total_tokens returns 0 for all decode slots
            # (num_prompt_tokens=0 and generated_tokens=[]), causing the memory
            # budget check in compose() to underestimate batch token usage.
            active_slots = [
                RequestSlot(
                    request_id=rid,
                    prompt_tokens=r.prompt_token_ids or [],
                    max_tokens=r.sampling_params.max_tokens
                    if r.sampling_params
                    else 512,
                    is_prefill=False,
                    priority=r.sampling_params.priority if r.sampling_params else 0,
                    num_prompt_tokens=getattr(r, "num_prompt_tokens", 0)
                    or len(r.prompt_token_ids or []),
                    generated_tokens=list(r.output_token_ids)
                    if getattr(r, "output_token_ids", None)
                    else [],
                )
                for rid, r in self.running.items()
            ]
            self._batch_composer.compose(pending_slots, active_slots)

        # ── GAP 1.3: Short-prompt fast path ──
        # When partial prefills are in-flight, short prompts (< threshold) should
        # jump ahead of long partial prefills. Reorder so short requests are
        # scheduled first, avoiding head-of-line blocking by long prefills.
        if self._active_partial_prefills > 0 and len(to_insert) > 1:
            _threshold = self.config.long_prefill_token_threshold

            def _short_prompt_priority(r: Request) -> int:
                """0 = short (schedule first), 1 = long."""
                n = len(r.prompt_token_ids) if r.prompt_token_ids else 0
                return 1 if n > _threshold else 0

            to_insert.sort(key=_short_prompt_priority)

        for req in to_insert:
            try:
                # Re-check abort status — an abort may have arrived after the
                # pre-filter loop but before we reached this point.
                if req.request_id in self._pending_abort_ids:
                    self._pending_abort_ids.discard(req.request_id)
                    req.set_finished(RequestStatus.FINISHED_ABORTED, reason="abort")
                    self._failed_insert_ids.append(req.request_id)
                    continue

                sp = req.sampling_params

                # Validate max_tokens: 0 or negative means no generation needed.
                # Immediately finish the request to avoid inserting into BatchGenerator.
                if sp.max_tokens is not None and sp.max_tokens <= 0:
                    req.set_finished(RequestStatus.FINISHED_STOPPED, reason="length")
                    self.finished_ids.add(req.request_id)
                    self._failed_insert_ids.append(req.request_id)
                    logger.debug(
                        f"Request {req.request_id} skipped: max_tokens={sp.max_tokens}"
                    )
                    continue

                sampler = self._make_sampler(sp, req.prompt_token_ids)
                sm = self._make_state_machine(sp.stop, sp.stop_token_ids)

                # ── Thinking-segment KV lookup before prefill ──
                # Check for reusable thinking KV segments from prior turns in
                # the same conversation. If found, attach cached KV data so the
                # engine can potentially skip re-thinking identical steps.
                if req.prompt_token_ids:
                    try:
                        # Use request_id as conversation identifier; callers may
                        # set a stable conversation_id via request metadata.
                        conv_id = (
                            getattr(req, "conversation_id", None) or req.request_id
                        )
                        conv_segments = self._thinking_store.get_conversation_segments(
                            conv_id
                        )
                        if conv_segments:
                            # Attach the most recently accessed segment for
                            # potential KV reuse (prefill optimisation).
                            best = max(conv_segments, key=lambda s: s.last_accessed)
                            req.prompt_cache = best.kv_data
                            req.cached_tokens = (
                                getattr(req, "cached_tokens", 0) + best.num_tokens
                            )
                            logger.debug(
                                f"Thinking KV reuse: {conv_id} → segment "
                                f"{best.step_hash} ({best.num_tokens} tokens)"
                            )
                    except Exception as e:
                        logger.debug(
                            f"Thinking KV lookup failed for {req.request_id}: {e}"
                        )

                # ── Chunked prefill: split long prompts across steps ──
                # Two modes:
                # 1. Sarathi-style (enable_hybrid_prefill=True): Always chunk
                # when decode requests are running, use hybrid_chunk_size.
                # 2. Standard chunked (SCHED-2): When prompt exceeds
                # prefill_chunk_size, chunk to avoid monopolising the batch.
                # This interleaves prefill chunks with decode even without
                # the full Sarathi mode, reducing head-of-line blocking.
                #
                # GAP 1.3: Concurrent partial prefill control caps how many
                # chunked prefills can be in-flight at once. Short prompts
                # (< long_prefill_token_threshold) can jump ahead of long
                # partial prefills that are monopolizing the prefill budget.
                tokens_to_insert = req.prompt_token_ids

                # ── Continue-resume after preemption ──
                # A preempted request kept its generated tokens (output not wiped). Fold
                # them into the re-prefill context so it CONTINUES rather than regenerates
                # (prevents the duplicate stream / truncation). The prefix cache still
                # matches the prompt prefix below, so typically only the generated tokens
                # are actually re-prefilled. Keyed on num_preemptions>0 so fresh requests
                # are unaffected (their output_token_ids is empty anyway).
                if getattr(req, "num_preemptions", 0) > 0 and req.output_token_ids:
                    tokens_to_insert = list(req.prompt_token_ids) + list(
                        req.output_token_ids
                    )

                # ── Prefill resume after preemption ──
                # When a request is preempted mid-prefill, num_computed_tokens
                # records how many prefix tokens are cached (via prefix cache or
                # block-level preemption).  Skip those tokens so we resume from
                # the saved position instead of re-prefilling from scratch.
                #
                # Guard: only apply when num_computed_tokens > 0 AND the request
                # was actually preempted (num_preemptions > 0).  A fresh request
                # with num_computed_tokens == 0 should prefill normally.
                #
                # If saved _prefill_progress has a KV cache reference, use
                # insert_segments to warm-start from the cached KV state.
                prefill_resume_offset = 0
                prefill_resume_kv = None
                if (
                    getattr(req, "num_computed_tokens", 0) > 0
                    and getattr(req, "num_preemptions", 0) > 0
                    and len(tokens_to_insert) > req.num_computed_tokens
                ):
                    prefill_resume_offset = req.num_computed_tokens
                    tokens_to_insert = req.prompt_token_ids[prefill_resume_offset:]
                    # Check if _preempt_request saved a KV cache for resumption.
                    # This carries the partial KV from the in-flight chunked
                    # prefill at the time of preemption.
                    saved_progress = getattr(req, "_prefill_progress", None)
                    if (
                        saved_progress is not None
                        and saved_progress.get("kv_cache") is not None
                    ):
                        prefill_resume_kv = saved_progress["kv_cache"]
                    # Also try the prefix cache for a KV hit on the already-computed prefix.
                    if prefill_resume_kv is None and self._prefix_cache is not None:
                        try:
                            import mlx.core as mx

                            ids_arr = mx.array(
                                req.prompt_token_ids[:prefill_resume_offset]
                            )
                            cached_kv, _, matched = self._prefix_cache.get(
                                ids_arr, exact_refeed_trim=False
                            )
                            if (
                                cached_kv is not None
                                and matched >= prefill_resume_offset
                            ):
                                prefill_resume_kv = cached_kv
                        except Exception:
                            logger.debug(
                                "prefix cache lookup for prefill resume failed",
                                exc_info=True,
                            )
                    logger.info(
                        f"Resuming prefill for preempted request {req.request_id}: "
                        f"skipping {prefill_resume_offset} cached tokens, "
                        f"{len(tokens_to_insert)} tokens remaining"
                    )
                    # Clear the saved progress to avoid stale state on subsequent steps
                    if hasattr(req, "_prefill_progress"):
                        del req._prefill_progress

                effective_chunk_size = 0
                should_chunk = False

                # Sarathi/hybrid: the BatchGenerator now chunks NATIVELY at
                # hybrid_chunk_size (prefill_step_size set at construction), which already
                # interleaves prefill with decode of running requests — the exact Sarathi
                # goal — and decodes only at the final 1-token segment. The old manual
                # chunking here had the same spurious-token + KV-corruption bug as the
                # standard path, so we no longer set should_chunk; the whole prompt is
                # inserted and mlx-lm chunks + interleaves it correctly. (_hybrid_prefill_
                # step then has no _pending_prefill entries to drive.)
                # SCHED-2: long prompts are now chunked NATIVELY by the
                # BatchGenerator (prefill_step_size = prefill_chunk_size, set at
                # construction) — it processes prefill_step_size tokens per next() per
                # sequence, interleaving with decode, and decodes only at the final
                # 1-token segment. Yunshu's old MANUAL cross-step chunking here fed
                # segments=[[chunk]] per step, so each chunk became a complete "prompt"
                # that decoded a spurious token AND corrupted the carried KV
                # (confidently-wrong output, proven by teacher-forcing). So we no longer
                # set should_chunk for the standard path; the whole prompt is inserted
                # and the BatchGenerator chunks + interleaves it correctly.

                # ── GAP 1.3: Enforce concurrent partial prefill caps ──
                if should_chunk:
                    num_tokens = len(tokens_to_insert)
                    is_long = num_tokens > self.config.long_prefill_token_threshold

                    if (
                        self._active_partial_prefills
                        >= self.config.max_num_partial_prefills
                    ):
                        # Total cap reached — defer this request
                        logger.debug(
                            f"Partial prefill cap reached ({self._active_partial_prefills}/"
                            f"{self.config.max_num_partial_prefills}), deferring {req.request_id}"
                        )
                        self.waiting.push(req, priority=req.sampling_params.priority)
                        continue

                    if (
                        is_long
                        and self._active_partial_prefills
                        >= self.config.max_long_partial_prefills
                    ):
                        # Long prefill cap reached — defer this long request
                        logger.debug(
                            f"Long partial prefill cap reached ({self._active_partial_prefills}/"
                            f"{self.config.max_long_partial_prefills}), deferring long request {req.request_id}"
                        )
                        self.waiting.push(req, priority=req.sampling_params.priority)
                        continue

                if should_chunk and effective_chunk_size > 0:
                    self._active_partial_prefills += 1
                    chunk = tokens_to_insert[:effective_chunk_size]
                    remaining = tokens_to_insert[effective_chunk_size:]
                    # When resuming from preemption, tokens_to_insert is already
                    # trimmed to the tail.  The full prompt context for
                    # insert_segments must include the resume prefix so KV cache
                    # positions align correctly.
                    all_prompt_for_chunks = (
                        req.prompt_token_ids
                        if prefill_resume_offset > 0
                        else tokens_to_insert
                    )
                    # Store remaining tokens for subsequent steps
                    self._pending_prefill[req.request_id] = {
                        "remaining_tokens": remaining,
                        "batch_uid": None,
                        # Track chunking mode for _process_pending_prefill
                        "chunk_size": effective_chunk_size,
                        "total_prompt_len": len(tokens_to_insert),
                        "offset": prefill_resume_offset + effective_chunk_size,
                        "kv_cache": prefill_resume_kv,
                        "all_prompt_tokens": all_prompt_for_chunks,
                    }
                    # Chunked prefill production tracking: fairness + timeout
                    self._chunked_prefill_fairness[req.request_id] = 0
                    self._chunked_prefill_enqueued_at[req.request_id] = time.monotonic()
                    tokens_to_insert = chunk

                # C16: Try KV prefix cache hit for batch-path acceleration
                # Must run BEFORE should_chunk adds to _pending_prefill, otherwise
                # the first chunk of a chunked prefill never benefits from caching.
                #
                # Prefill resume: if we already have a KV cache from preemption
                # (prefill_resume_kv), use it as the starting KV state and skip
                # the prefix cache lookup — the prefix is already in the resume KV.
                cached_kv = prefill_resume_kv
                remaining_tokens = tokens_to_insert
                if cached_kv is not None:
                    # Already have resume KV — tokens_to_insert is already trimmed
                    req.cached_tokens = prefill_resume_offset
                elif self._prefix_cache is not None:
                    try:
                        import mlx.core as mx

                        ids_arr = mx.array(tokens_to_insert)
                        cached_kv, _, matched = self._prefix_cache.get(
                            ids_arr, exact_refeed_trim=False
                        )
                        # a FULL-prompt match (matched ==
                        # every token) leaves 0 tokens to prefill, and the
                        # BatchGenerator needs >=1 token to start decoding →
                        # "failed to insert". The valuable case (shared system
                        # prompt + DIFFERENT user msg) is a PARTIAL match and is
                        # unaffected. For an exact full match (identical request,
                        # which the response cache handles) fall back to a normal
                        # prefill so output stays correct.
                        if cached_kv is not None and matched >= len(tokens_to_insert):
                            cached_kv = None
                        if cached_kv is not None and matched > 0:
                            remaining_tokens = tokens_to_insert[matched:]
                            req.cached_tokens = matched
                            if matched > 32:
                                logger.info(
                                    f"Batch prefix cache hit: {matched}/{len(tokens_to_insert)} tokens "
                                    f"for {req.request_id}"
                                )
                    except Exception:
                        logger.debug(
                            "prefix cache lookup failed in batch path", exc_info=True
                        )

                # Inflight prefix sharing: if the completed
                # prefix cache didn't have a match, check if another request
                # is currently being prefilled with the same prefix.  If so,
                # reuse its partial KV cache so we skip the shared prefill.
                #
                # This handles the case where Request A starts prefilling and
                # Request B arrives with the same system prompt.  B finds A's
                # in-flight KV via the tracker and starts from the shared
                # prefix boundary, avoiding redundant prefill.
                if cached_kv is None and req.prompt_token_ids:
                    try:
                        from .inflight_prefix_sharing import get_inflight_tracker

                        _tracker = get_inflight_tracker()
                        _inflight_entry = _tracker.find_prefix(
                            req.prompt_token_ids,
                            getattr(self, "model_id", "") or "",
                        )
                        if (
                            _inflight_entry is not None
                            and _inflight_entry.kv_cache_ref is not None
                        ):
                            cached_kv = _inflight_entry.kv_cache_ref
                            shared_len = min(
                                len(_inflight_entry.token_ids),
                                len(req.prompt_token_ids),
                            )
                            remaining_tokens = req.prompt_token_ids[shared_len:]
                            req.cached_tokens = shared_len
                            logger.debug(
                                "inflight prefix reuse: %d tokens from req=%s for %s",
                                shared_len,
                                _inflight_entry.request_id[:12],
                                req.request_id[:12],
                            )
                    except Exception:
                        logger.debug(
                            "inflight prefix lookup failed in batch path", exc_info=True
                        )

                # Check encoder cache for encoder-decoder models.
                # If the request has a cached encoder hidden state (from a prior
                # request with the same encoder input), attach it so the decoder
                # can skip re-encoding.
                if hasattr(req, "encoder_request_id"):
                    cached_encoder = self._encoder_cache.get(req.encoder_request_id)
                    if cached_encoder is not None:
                        req.cached_encoder_output = cached_encoder
                        logger.debug(
                            "Encoder cache hit for %s — reusing encoder output",
                            req.request_id,
                        )

                # When resuming prefill after preemption, all_tokens_for_segments
                # must be the FULL prompt (not the trimmed tail) so the KV cache
                # positions align correctly.  For non-resume paths, it equals
                # tokens_to_insert (which is the full prompt in that case).
                all_tokens_for_segments = (
                    req.prompt_token_ids
                    if prefill_resume_offset > 0
                    else tokens_to_insert
                )

                # on continue-resume the already-generated tokens
                # count against the original max_tokens, so insert with the REMAINING
                # budget (else the request would generate a full max_tokens MORE on top of
                # what it already produced). Fresh requests have empty output → unchanged.
                _eff_max = sp.max_tokens
                if _eff_max is not None and req.output_token_ids:
                    _eff_max = max(1, _eff_max - len(req.output_token_ids))

                # record whether this request actually warm-started from an
                # EXTERNAL cached KV (insert_segments(caches=...)). _save_one_prefix must
                # gate on THIS, not on req.cached_tokens: paged_scheduler sets cached_tokens
                # from the RADIX accounting match (num_matched_tokens), which diverges from
                # the KVPrefixCache — when radix matches but the prefix cache MISSES, the
                # request does a FULL BatchGenerator prefill yet cached_tokens stays >0, so
                # the old gate skipped saving a genuinely-complete prefill KV → that prefix
                # was never cached (permanent missed reuse). The flag is exact: True only
                # when external KV was actually adopted (extract_cache would be short).
                req._external_kv_reuse = cached_kv is not None
                if cached_kv is not None and len(remaining_tokens) > 0:
                    # Use insert_segments with cached KV state
                    uids = self._batch_gen.insert_segments(
                        segments=[[remaining_tokens]],
                        max_tokens=[_eff_max],
                        caches=[cached_kv],
                        all_tokens=[all_tokens_for_segments],
                        samplers=[sampler],
                        state_machines=[sm],
                    )
                elif cached_kv is not None and len(remaining_tokens) == 0:
                    # Full prefix cache hit — no tokens to prefill.  Use
                    # insert_segments with the cached KV and an empty segment
                    # so the BatchGenerator initializes the decode state from
                    # the cached KV without re-prefilling.  Without this branch,
                    # the else path calls insert() with the full prompt, wasting
                    # the prefix cache hit entirely.
                    uids = self._batch_gen.insert_segments(
                        segments=[[]],  # no remaining tokens to prefill
                        max_tokens=[_eff_max],
                        caches=[cached_kv],
                        all_tokens=[all_tokens_for_segments],
                        samplers=[sampler],
                        state_machines=[sm],
                    )
                else:
                    uids = self._batch_gen.insert(
                        prompts=[tokens_to_insert],
                        max_tokens=[_eff_max],
                        samplers=[sampler],
                        state_machines=[sm],
                    )

                if not uids:
                    logger.error(
                        f"BatchGenerator.insert returned empty UIDs for {req.request_id}"
                    )
                    req.set_finished(
                        RequestStatus.FINISHED_ERROR, reason="insert_failed"
                    )
                    if should_chunk:
                        self._pending_prefill.pop(req.request_id, None)
                        self._active_partial_prefills = max(
                            0, self._active_partial_prefills - 1
                        )
                    # Must track as failed insert so step() generates a synthetic
                    # error output and EngineCore calls _finalize_request.
                    # Without this, per-request resources (output_queue, done_event)
                    # leak because no output is ever produced for this request.
                    self._failed_insert_ids.append(req.request_id)
                    continue
                req.batch_uid = uids[0]
                req.status = RequestStatus.RUNNING
                req.prefill_start = time.monotonic()
                self.running[req.request_id] = req
                self._uid_to_req[uids[0]] = req.request_id

                # H2O: Register request for attention score tracking
                if self._attention_score_tracker is not None:
                    self._attention_score_tracker.register_request(req.request_id)

                # Register mRoPE delta for batch decode
                if getattr(req, "rope_deltas", 0.0) != 0.0:
                    self._rope_delta_mgr.register(uids[0], req.rope_deltas)

                # Create fresh detokenizer (never pool)
                req.detokenizer = self._create_detokenizer()
                self._detokenizers[req.request_id] = req.detokenizer

                # Create thinking budget processor if configured
                sp = req.sampling_params
                if sp.thinking_budget is not None or sp.reasoning_effort is not None:
                    from .thinking_budget import (
                        ThinkingBudgetConfig,
                        ThinkingBudgetProcessor,
                        parse_thinking_budget,
                    )

                    config = parse_thinking_budget(
                        {
                            "thinking_budget": sp.thinking_budget,
                            "reasoning_effort": sp.reasoning_effort,
                        }
                    )
                    if config is not None:
                        self._thinking_processors[req.request_id] = (
                            ThinkingBudgetProcessor(config)
                        )
                elif getattr(sp, "enable_thinking", False) or getattr(
                    req, "enable_thinking", False
                ):
                    # Auto-detected thinking mode with default budget
                    from .thinking_budget import (
                        ThinkingBudgetConfig,
                        ThinkingBudgetProcessor,
                        detect_needs_think_prefix,
                    )

                    if detect_needs_think_prefix(
                        req.prompt_token_ids or [], self.tokenizer
                    ):
                        self._thinking_processors[req.request_id] = (
                            ThinkingBudgetProcessor(
                                ThinkingBudgetConfig(max_thinking_tokens=8192)
                            )
                        )

                self._total_prompt_tokens += req.num_prompt_tokens
                # Only count unique requests — skip re-inserted preempted requests
                # which were already counted during their first insertion.
                if getattr(req, "num_preemptions", 0) == 0:
                    self._num_requests += 1

                # Track prefill progress
                if self._prefill_tracker is not None:
                    self._prefill_tracker.update(
                        req.request_id,
                        0,
                        req.num_prompt_tokens,
                        self.model_id,
                    )

                # ── Speculative decoding head detection (Phase 4) ──
                # On first request, check if model has spec heads (EAGLE/MTP/Medusa)
                # and create a SpeculativeDecoder if detected.
                if self._spec_decoder is None and self._spec_head_info is None:
                    self._try_init_spec_decoder()

            except Exception as e:
                logger.error(
                    f"Failed to insert request {req.request_id}: {e}", exc_info=True
                )
                req.set_finished(RequestStatus.FINISHED_ERROR, reason="error")
                # Undo _total_prompt_tokens increment — the request never ran
                self._total_prompt_tokens = max(
                    0, self._total_prompt_tokens - getattr(req, "num_prompt_tokens", 0)
                )
                # Signal completion so callers don't hang
                self._uid_to_req.pop(getattr(req, "batch_uid", None), None)
                self.finished_ids.add(req.request_id)
                # Track failed insert so step() generates an error output for
                # EngineCore to finalize (otherwise resources leak).
                self._failed_insert_ids.append(req.request_id)
                # Clean up partial prefill tracking if we incremented the counter
                # but never added to self.running (which _cleanup_finished skips).
                if should_chunk:
                    self._pending_prefill.pop(req.request_id, None)
                    self._active_partial_prefills = max(
                        0, self._active_partial_prefills - 1
                    )

    _MAX_PREEMPTIONS_PER_REQUEST = 3
    _MAX_PREEMPTIONS_PER_STEP = 8  # Cap per-step preemptions to prevent cascade

    def _preempt_lowest_priority(self, count: int) -> int:
        """Preempt the lowest-priority running requests.

        Under PRIORITY policy, finds the running requests with the lowest
        priority (ties broken by arrival_time, newest first) and preempts
        them. Preempted requests are placed back at the front of the
        waiting queue with their KV state freed.

        Caps individual requests at _MAX_PREEMPTIONS_PER_REQUEST to prevent
        livelock where the same request is preempted and re-inserted every step.

        Single-sort optimization: builds a sorted candidate list once instead
        of scanning all running requests count times (O(n log n) vs O(n*count)).

        Args:
            count: Number of requests to preempt.

        Returns:
            Number of requests actually preempted.
        """
        if not self.running or count <= 0:
            return 0

        # Build sorted candidate list: prefill first, then decode.
        # Within each group: lowest priority first, highest KV block usage
        # first (thinking-budget requests consume more KV and free more
        # memory when preempted), newest arrival first.
        def _sort_key(rid: str) -> tuple:
            req = self.running[rid]
            is_decode = 1 if req.output_token_ids else 0
            kv_usage = -(req.num_prompt_tokens + req.num_output_tokens)
            # Use effective priority (raw + aging) to match _schedule_waiting
            # ordering. Using raw priority causes priority inversion where
            # an aged-in request is immediately preempted due to low raw priority.
            raw_pri = getattr(req.sampling_params, "priority", 0) or 0
            _age = max(
                0.0,
                time.monotonic()
                - (req._submit_time if req._submit_time > 0 else req.arrival_time),
            )
            effective_pri = raw_pri + _age * self.config.aging_weight
            return (is_decode, effective_pri, kv_usage, -req.arrival_time)

        now = time.monotonic()
        eligible = [
            rid
            for rid in self.running
            if self.running[rid].num_preemptions < self._MAX_PREEMPTIONS_PER_REQUEST
            and rid not in self._pending_abort_ids
            # Preemption cascade cooldown: skip requests that were just
            # preempted and reinserted within the cooldown window.  Without
            # this, a request can be preempted → reinserted → immediately
            # preempted again in a feedback loop (cascade).
            and (now - getattr(self.running[rid], "_last_preempt_time", 0.0))
            >= self._preemption_cooldown_seconds
        ]
        if not eligible:
            return 0

        eligible.sort(key=_sort_key)
        to_preempt = eligible[:count]

        actual_preempted = 0
        for victim_id in to_preempt:
            victim = self.running.pop(victim_id, None)
            if victim is not None:
                self._preempt_request(victim)
                actual_preempted += 1

        return actual_preempted

    def _preempt_request(self, request: Request) -> None:
        """Preempt a running request and return it to the waiting queue.

        Block-level preemption with partial recomputation (SCHED-1):
        1. Extract KV cache from BatchGenerator before removal
        2. Save the prompt prefix portion to the KV prefix cache
        3. Preserve cached prefix tokens (RadixTree maintains these)
        4. Only reset computed tokens beyond the cached prefix
        5. Set status to PREEMPTED
        6. Put back at front of waiting queue for re-scheduling

        When the request is re-scheduled, the prefix cache will be checked
        and only the uncached tail needs re-prefilling, significantly
        reducing re-prefill overhead compared to whole-request preemption.

        Safety: The caller already popped the request from ``self.running``.
        If any step below fails, we still push the request to ``self.waiting``
        so it is never orphaned (neither running nor waiting).
        """
        uid = request.batch_uid

        try:
            # SCHED-1: Extract KV cache before removal so we can save the prefix.
            # BatchGenerator.remove(uids, return_prompt_caches=True) returns
            # {uid: (cache_list, tokens_list)} for generation-stage requests,
            # or {uid: (cache_list, tokens)} for prompt-stage requests.
            extracted_caches: dict = {}
            if uid is not None and self._batch_gen is not None:
                try:
                    extracted_caches = self._batch_gen.remove(
                        [uid], return_prompt_caches=True
                    )
                except Exception as e:
                    # Fallback: remove without cache extraction if API differs
                    logger.debug(
                        f"Failed to extract cache for preempted UID {uid}: {e}"
                    )
                    try:
                        self._batch_gen.remove([uid])
                    except Exception as e2:
                        logger.debug(f"Failed to remove preempted UID {uid}: {e2}")

            # SCHED-1: Save the prompt prefix portion of the KV cache to the
            # prefix cache.  Only the prompt tokens (not generated tokens) are
            # saved because the prefix cache keys off prompt token sequences.
            #
            # Sliding window guard: for models with sliding window attention,
            # the KV cache may have evicted early prompt tokens.  If the
            # total token count (prompt + output) exceeds the sliding window,
            # early prompt KV blocks have been evicted and saving the full
            # prompt KV would store stale/corrupted data.
            # Skip saving in that case.
            saved_prefix = 0
            _sw_window = getattr(self.model, "_yunshu_swa_window", None)
            _prompt_outside_window = (
                isinstance(_sw_window, (int, float))
                and request.num_output_tokens + request.num_prompt_tokens > _sw_window
            )
            if (
                self._prefix_cache is not None
                and uid in extracted_caches
                and request.prompt_token_ids
                and not _prompt_outside_window
                # CRITICAL: do NOT save a request that REUSED a cached
                # prefix (cached_tokens > 0). Its BatchGenerator cache only covers
                # prompt_len - cached_tokens tokens, but we'd store it under the
                # FULL prompt key → a corrupt entry that claims more tokens than it
                # has KV for → the next exact hit returns matched==full but the
                # cache is short → wrong output / "failed to insert". Mirrors the
                # same guard in _save_one_prefix. (cached_tokens is reset to 0 only
                # later in this method, so it's still valid here.)
                and not getattr(request, "cached_tokens", 0)
            ):
                try:
                    import mlx.core as mx

                    cache_and_tokens = extracted_caches[uid]
                    if cache_and_tokens is not None:
                        cache_data = cache_and_tokens[0]
                        cache_and_tokens[1]
                        # Only save if we got valid cache data and the request
                        # has generated enough tokens to make caching worthwhile
                        if cache_data and request.num_prompt_tokens >= 32:
                            prompt_ids = mx.array(request.prompt_token_ids)
                            self._prefix_cache.add(prompt_ids, cache_data)
                            saved_prefix = len(request.prompt_token_ids)
                            logger.info(
                                f"Saved KV prefix cache for preempted request "
                                f"{request.request_id}: {saved_prefix} prompt tokens"
                            )
                except Exception:
                    logger.debug(
                        "Failed to save KV prefix during preemption", exc_info=True
                    )

            self._uid_to_req.pop(uid, None)
            # Unregister mRoPE delta for the old UID to prevent stale delta
            # entries from leaking in the batch delta manager (the request will
            # get a new UID on re-insertion).
            if uid is not None:
                self._rope_delta_mgr.unregister(uid)
            self._detokenizers.pop(request.request_id, None)
            self._thinking_processors.pop(request.request_id, None)
            self._thinking_state.pop(request.request_id, None)
            # Save chunked prefill progress before discarding, so the request
            # can resume (not restart) when re-scheduled after preemption.
            _pending_state = self._pending_prefill.get(request.request_id)
            if _pending_state is not None:
                request._prefill_progress = {
                    "kv_cache": _pending_state.get("kv_cache"),
                    "offset": _pending_state.get("offset", 0),
                }
            self._pop_pending_prefill(request.request_id)
            self._cleanup_spec_state(request.request_id)
            # Clean up ITL tracking — stale _last_token_time causes a massive
            # ITL spike on the first token after re-insertion (the delta spans
            # the entire preemption + re-prefill period).  Without this cleanup,
            # preemption corrupts ITL p99 histograms and ServerMetrics.
            self._last_token_time.pop(request.request_id, None)
            self._itl_samples.pop(request.request_id, None)

            # H2O: Log attention-based eviction order for debugging.
            if self._attention_score_tracker is not None:
                eviction_order = self._attention_score_tracker.get_eviction_order(
                    request.request_id
                )
                if eviction_order:
                    logger.debug(
                        f"H2O eviction order for preempted {request.request_id}: "
                        f"first={eviction_order[0]}, last={eviction_order[-1]}, "
                        f"total_blocks={len(eviction_order)}"
                    )
                self._attention_score_tracker.remove_request(request.request_id)

            # Cleanup chunked prefill production tracking
            self._chunked_prefill_fairness.pop(request.request_id, None)
            self._chunked_prefill_enqueued_at.pop(request.request_id, None)

            # Block-level preemption: check how many prefix tokens are cached.
            # If SCHED-1 saved the prefix above, this will find it immediately.
            # Otherwise, fall back to checking existing prefix cache entries.
            cached_prefix = max(saved_prefix, 0)
            if (
                cached_prefix == 0
                and self._prefix_cache is not None
                and request.prompt_token_ids
            ):
                try:
                    import mlx.core as mx

                    ids_arr = mx.array(request.prompt_token_ids)
                    _, _, matched = self._prefix_cache.get(
                        ids_arr, exact_refeed_trim=False
                    )
                    if matched > 0:
                        cached_prefix = matched
                except Exception:
                    logger.debug("failed", exc_info=True)

            request.status = RequestStatus.PREEMPTED
            # Undo prompt token count — re-insertion will re-add it.
            # Without this, each preemption cycle double-counts prompt tokens.
            self._total_prompt_tokens = max(
                0, self._total_prompt_tokens - request.num_prompt_tokens
            )
            # Reset submit time so timeout doesn't count time spent preempted
            request._submit_time = time.monotonic()
            # Record last preemption time for cascade cooldown — a request
            # that was just preempted and reinserted should not be immediately
            # preempted again in the same or next step (causes cascade feedback).
            request._last_preempt_time = time.monotonic()
            # Preserve cached prefix tokens — only reset beyond cache boundary
            prompt_len = getattr(request, "num_prompt_tokens", 0) or len(
                request.prompt_token_ids
            )
            request.num_computed_tokens = min(
                cached_prefix, request.num_computed_tokens, prompt_len
            )
            request.batch_uid = None
            # BUG FIX: Always increment num_preemptions for both priority
            # preemption and retraction.  Previously, retraction left
            # num_preemptions at 0, causing _schedule_waiting to
            # double-count the request in self._num_requests on re-insertion
            # (line: "if getattr(req, 'num_preemptions', 0) == 0").
            # The _MAX_PREEMPTIONS_PER_REQUEST cap in _preempt_lowest_priority
            # now covers both priority preemption and retraction, which is
            # desirable — a request evicted 3 times for any reason should be
            # protected from further livelock.
            request.num_preemptions += 1
            # CONTINUE-not-restart. Do NOT wipe the generated
            # tokens. _schedule_waiting folds output_token_ids into the re-prefill context
            # (keyed on num_preemptions>0), so the request resumes from where it stopped
            # instead of regenerating. Wiping + regenerating made a STREAMING client
            # receive the pre-preempt text and THEN a second generation (duplicated /
            # divergent stream — S3), and truncated the answer below max_tokens because
            # the (kept) token budget no longer matched the (wiped) output (BUG-3). The
            # detokenizer was popped above and is recreated fresh on resume, so only NEW
            # tokens are detokenized; req.output_text keeps appending seamlessly.
            # BUG FIX: Reset cached_tokens so re-insertion recalculates it.
            # Without this, stale cached_tokens from the original prefill
            # persists and gets reported in RequestOutput.cached_tokens,
            # over-reporting cache hits to the client / billing.
            request.cached_tokens = 0
            # BUG FIX: Clear stale finish_reason from the original lifecycle.
            # If the request was previously finished (e.g. thinking budget
            # overflow) and then preempted, the stale finish_reason would be
            # reported in the error output if re-insertion fails (line 1119).
            request.finish_reason = None

            self.waiting.push_front(request, priority=request.sampling_params.priority)

            prefix_info = (
                f", cached_prefix={cached_prefix}" if cached_prefix > 0 else ""
            )
            logger.info(
                f"Preempted request {request.request_id} "
                f"(preemptions={request.num_preemptions}, "
                f"output_tokens={request.num_output_tokens}{prefix_info})"
            )
        except Exception:
            # Safety net: if anything above failed, the request has already
            # been popped from self.running by the caller.  We MUST push it
            # to self.waiting so it is never orphaned.
            logger.exception(
                f"Unexpected error during preemption of {request.request_id}, "
                f"returning request to waiting queue"
            )
            request.status = RequestStatus.PREEMPTED
            request.batch_uid = None
            request.cached_tokens = 0
            request.finish_reason = None
            self.waiting.push_front(request, priority=request.sampling_params.priority)

    def _retract_decode_requests(self, count: int) -> int:
        """Temporarily retract decode requests under memory pressure (C14).

        Swap out decode requests (which have lower per-token
        memory cost than prefill) to make room for new prefill requests.
        Retracted requests are placed at the front of the waiting queue and
        will be re-inserted with their existing KV state (via prefix cache).

        Retraction now increments num_preemptions (same as priority preemption),
        which prevents _num_requests double-counting on re-insertion and makes
        the _MAX_PREEMPTIONS_PER_REQUEST cap cover both eviction types — a
        request evicted 3 times for any reason should be protected from livelock.

        Args:
            count: Maximum number of requests to retract.

        Returns:
            Number of requests retracted.
        """
        retracted = 0
        # Sort running requests by output tokens (longest = most memory, evict first).
        # Under PRIORITY policy, use priority as secondary criterion: among equal
        # output tokens, evict lowest-priority requests first.
        if getattr(self.config, "policy", None) == SchedulingPolicy.PRIORITY:
            candidates = sorted(
                [r for r in self.running.values() if r.batch_uid is not None],
                key=lambda r: (
                    -r.num_output_tokens,
                    r.sampling_params.priority if r.sampling_params else 0,
                ),
            )
        else:
            candidates = sorted(
                [r for r in self.running.values() if r.batch_uid is not None],
                key=lambda r: r.num_output_tokens,
                reverse=True,
            )

        _now = time.monotonic()
        for victim in candidates:
            if retracted >= count:
                break
            if (
                getattr(victim, "num_preemptions", 0)
                >= self._MAX_PREEMPTIONS_PER_REQUEST
            ):
                continue
            # Cascade cooldown: same as _preempt_lowest_priority
            if (
                _now - getattr(victim, "_last_preempt_time", 0.0)
            ) < self._preemption_cooldown_seconds:
                continue
            self.running.pop(victim.request_id, None)
            self._preempt_request(victim)
            retracted += 1

        return retracted

    def _pop_pending_prefill(self, req_id: str) -> None:
        """Remove a request from _pending_prefill and decrement the active counter.

        Helper for GAP 1.3: ensures _active_partial_prefills stays in sync
        whenever a partial prefill is removed outside of _process_pending_prefill's
        normal completion/errored cleanup path.
        """
        if self._pending_prefill.pop(req_id, None) is not None:
            self._active_partial_prefills = max(0, self._active_partial_prefills - 1)

    def _process_pending_prefill(self) -> None:
        """Process pending partial prefill chunks with production hardening.

        ⚠️ CURRENTLY UNREACHABLE: `_pending_prefill` is
        populated only when `should_chunk` is True, but `should_chunk` is hardwired
        False (see ~line 1892 — the old manual chunking corrupted KV; mlx-lm now
        chunks natively via prefill_step_size). So this method and everything it
        drives (`_hybrid_prefill_step`, the chunked-prefill fairness/budget/timeout
        accounting) never execute. Retained as dormant code; do NOT re-flag as a
        bug — it is intentionally dead until/unless a corruption-free manual
        chunking path is reintroduced.

        Handles two chunked-prefill modes:

        1. Sarathi-style hybrid (enable_hybrid_prefill=True): Process
           pending prefill chunks interleaved with decode steps. Prevents
           long prefills from starving generation.

        2. Standard chunked prefill (SCHED-2): When a prompt exceeded
           prefill_chunk_size, continue feeding chunks on each step.
           Also interleaves with decode to avoid head-of-line blocking.
           Chunk size is tracked per-request in the pending_prefill state dict.

        Production hardening:
        - Fairness: configurable budget (chunked_prefill_budget) caps chunks
          per scheduling round; decode always gets at least 1 slot.
          Requests with fewer chunks served are prioritized.
        - Timeout: if a chunked prefill takes longer than
          chunked_prefill_timeout_seconds, it is either aborted (default)
          or force-fed depending on chunked_prefill_abort_on_timeout.
        - Cleanup: ensures _pending_prefill, _chunked_prefill_fairness,
          and _chunked_prefill_enqueued_at are cleaned for aborted, missing,
          or failed requests. Failed requests get FINISHED_ERROR status.
        - Progress: reports chunk-level progress via PrefillProgressTracker.
        - Error handling: if a single chunk fails to insert, the entire
          request is aborted with FINISHED_ERROR — no partially prefilled
          requests are left in the scheduler.

        When hybrid prefill is off and no standard chunking is active,
        all pending chunks are fed at once for maximum throughput.
        """
        # Always clean orphaned entries (entries in tracking dicts but not
        # in _pending_prefill). This can happen when a request finishes or
        # is aborted between steps.
        if not self._pending_prefill:
            if self._chunked_prefill_enqueued_at or self._chunked_prefill_fairness:
                orphan_ids = set(self._chunked_prefill_enqueued_at.keys()) | set(
                    self._chunked_prefill_fairness.keys()
                )
                for rid in orphan_ids:
                    self._chunked_prefill_enqueued_at.pop(rid, None)
                    self._chunked_prefill_fairness.pop(rid, None)
            return

        _now = time.monotonic()
        completed_ids: set[str] = set()
        errored_ids: set[str] = set()
        chunks_fed = 0
        budget = self.config.chunked_prefill_budget
        timeout_seconds = self.config.chunked_prefill_timeout_seconds
        abort_on_timeout = self.config.chunked_prefill_abort_on_timeout

        # ── Timeout handling ──
        # Check for chunked prefills that have been pending too long.
        timed_out_ids: list[str] = []
        for req_id, enqueued_at in list(self._chunked_prefill_enqueued_at.items()):
            if req_id not in self._pending_prefill:
                # Already completed or cleaned up elsewhere
                self._chunked_prefill_enqueued_at.pop(req_id, None)
                self._chunked_prefill_fairness.pop(req_id, None)
                continue
            if timeout_seconds > 0 and _now - enqueued_at > timeout_seconds:
                timed_out_ids.append(req_id)

        for req_id in timed_out_ids:
            state = self._pending_prefill.get(req_id)
            if state is None:
                continue
            pending_duration = _now - self._chunked_prefill_enqueued_at.get(
                req_id, _now
            )

            if abort_on_timeout:
                # Abort the request — return error to the client
                req = self.running.get(req_id)
                # Always decrement partial prefill counter — the request had
                # one entry in _pending_prefill regardless of whether it's still
                # in self.running.  Without this, a request removed from running
                # by a concurrent _cleanup_finished leaks the counter permanently.
                self._pop_pending_prefill(req_id)
                if req is not None:
                    req.set_finished(
                        RequestStatus.FINISHED_ERROR, reason="prefill_timeout"
                    )
                    # Remove from running immediately to free the slot —
                    # _cleanup_finished won't run until end-of-step and
                    # this dead request would waste a preemption slot.
                    self.running.pop(req_id, None)
                    uid = getattr(req, "batch_uid", None)
                    self._uid_to_req.pop(uid, None)
                    # Remove from BatchGenerator if it was inserted
                    if uid is not None and self._batch_gen is not None:
                        try:
                            self._batch_gen.remove([uid])
                        except Exception:
                            logger.debug(
                                "batch gen remove for timeout abort failed",
                                exc_info=True,
                            )
                    # Decrement _total_prompt_tokens — the counter was incremented
                    # when the request was first inserted into the batch (line 1836)
                    # and this abort means the prompt tokens are "wasted" (the request
                    # will never produce output).  Without this, _total_prompt_tokens
                    # grows monotonically even as requests fail, inflating metrics.
                    self._total_prompt_tokens = max(
                        0,
                        self._total_prompt_tokens
                        - getattr(req, "num_prompt_tokens", 0),
                    )
                    # Clean up per-request state.  _pop_pending_prefill was already
                    # called above (outside the if block) so the bottom loop's guard
                    # will correctly skip the second decrement.
                    for cleanup_dict in (
                        self._detokenizers,
                        self._thinking_processors,
                        self._thinking_state,
                        self._chunked_prefill_fairness,
                        self._chunked_prefill_enqueued_at,
                    ):
                        cleanup_dict.pop(req_id, None)
                    if uid is not None:
                        self._rope_delta_mgr.unregister(uid)
                    if self._prefill_tracker is not None:
                        self._prefill_tracker.remove(req_id)
                    logger.warning(
                        f"Chunked prefill timeout for {req_id}: aborting "
                        f"(pending for {pending_duration:.1f}s, "
                        f"{len(state.get('remaining_tokens', []))} tokens remaining)"
                    )
                errored_ids.add(req_id)
            else:
                # Force-feed: dump all remaining tokens in one shot
                remaining = state.get("remaining_tokens", [])
                force_fed = False
                if remaining and self._batch_gen is not None:
                    req = self.running.get(req_id)
                    if req is not None and req_id not in self._pending_abort_ids:
                        logger.warning(
                            f"Chunked prefill timeout for {req_id}: "
                            f"force-feeding {len(remaining)} remaining tokens "
                            f"(pending for {pending_duration:.1f}s)"
                        )
                        try:
                            sp = req.sampling_params
                            sampler = self._make_sampler(
                                sp, state.get("all_prompt_tokens")
                            )
                            sm = self._make_state_machine(sp.stop, sp.stop_token_ids)
                            # Use insert_segments for KV continuity if cache available
                            prev_kv = state.get("kv_cache")
                            all_tokens = state.get("all_prompt_tokens")
                            if prev_kv is not None and all_tokens is not None:
                                processed_count = state.get("offset", 0)
                                uids = self._batch_gen.insert_segments(
                                    segments=[[remaining]],
                                    max_tokens=[sp.max_tokens],
                                    caches=[prev_kv],
                                    all_tokens=[
                                        all_tokens[:processed_count] + remaining
                                    ],
                                    samplers=[sampler],
                                    state_machines=[sm],
                                )
                            else:
                                uids = self._batch_gen.insert(
                                    prompts=[remaining],
                                    max_tokens=[sp.max_tokens],
                                    samplers=[sampler],
                                    state_machines=[sm],
                                )
                            # Guard: BatchGenerator may return empty UIDs (e.g., batch full).
                            # Without this guard, uids[0] raises IndexError, which is caught
                            # by the outer except but leaves the old UID in BatchGenerator,
                            # leaking GPU memory (the old chunk's KV cache stays allocated).
                            if not uids:
                                logger.error(
                                    "BatchGenerator.insert returned empty UIDs for "
                                    "force-feed of timed-out chunked prefill %s",
                                    req_id,
                                )
                                req.set_finished(
                                    RequestStatus.FINISHED_ERROR, reason="insert_failed"
                                )
                                errored_ids.add(req_id)
                                # Remove old UID from BatchGenerator to prevent KV leak
                                old_uid_ff = getattr(req, "batch_uid", None)
                                if (
                                    old_uid_ff is not None
                                    and self._batch_gen is not None
                                ):
                                    try:
                                        self._batch_gen.remove([old_uid_ff])
                                    except Exception:
                                        logger.debug(
                                            "Failed to remove old UID %s during force-feed failure",
                                            old_uid_ff,
                                            exc_info=True,
                                        )
                                continue
                            # Update UID tracking (force-feed creates a new UID)
                            old_uid = getattr(req, "batch_uid", None)
                            if old_uid is not None and old_uid != uids[0]:
                                self._uid_to_req.pop(old_uid, None)
                                try:
                                    self._batch_gen.remove([old_uid])
                                except Exception:
                                    logger.warning(
                                        "Failed to remove old UID %s during "
                                        "force-feed — BatchGenerator slot may leak",
                                        old_uid,
                                        exc_info=True,
                                    )
                            req.batch_uid = uids[0]
                            self._uid_to_req[uids[0]] = req_id
                            self._chunked_prefill_chunks_processed += 1
                            chunks_fed += 1
                            force_fed = True
                            state["remaining_tokens"] = []
                        except Exception as e:
                            logger.error(
                                f"Failed to force-feed timed-out chunked prefill for {req_id}: {e}",
                                exc_info=True,
                            )
                            req = self.running.get(req_id)
                            if req is not None:
                                req.set_finished(
                                    RequestStatus.FINISHED_ERROR, reason="prefill_error"
                                )
                                self._total_prompt_tokens = max(
                                    0,
                                    self._total_prompt_tokens
                                    - getattr(req, "num_prompt_tokens", 0),
                                )
                                failed_uid_ff = getattr(req, "batch_uid", None)
                                self._uid_to_req.pop(failed_uid_ff, None)
                                if (
                                    failed_uid_ff is not None
                                    and self._batch_gen is not None
                                ):
                                    try:
                                        self._batch_gen.remove([failed_uid_ff])
                                    except Exception:
                                        logger.debug(
                                            "Failed to remove UID during force-feed error",
                                            exc_info=True,
                                        )
                            errored_ids.add(req_id)
                # Only mark as completed if force-feed succeeded or there are no remaining tokens.
                # If force-feed failed (but not errored — e.g., req was None or batch_gen missing),
                # mark as errored to prevent the request from being silently dropped.
                if force_fed or not remaining:
                    completed_ids.add(req_id)
                else:
                    # Force-feed failed without exception (e.g., req was None
                    # or batch_gen missing) — mark as error to prevent silent
                    # drop. The request would otherwise be orphaned forever.
                    _ff_req = self.running.get(req_id)
                    # CRITICAL: is_finished is a METHOD
                    # (request.py:230, no @property); `not <bound_method>`
                    # is always False → FINISHED_ERROR branch never ran on
                    # force-feed failures. Call it.
                    if _ff_req is not None and not _ff_req.is_finished():
                        _ff_req.set_finished(
                            RequestStatus.FINISHED_ERROR,
                            reason="prefill_force_feed_failed",
                        )
                    self._total_prompt_tokens = max(
                        0,
                        self._total_prompt_tokens
                        - getattr(_ff_req, "num_prompt_tokens", 0),
                    )
                    errored_ids.add(req_id)

        # ── Fairness: sort pending requests by chunks served (ascending) ──
        # Requests that have received fewer chunks are processed first,
        # preventing a single long prompt from starving others.
        pending_items = list(self._pending_prefill.items())
        if len(pending_items) > 1:

            def _fairness_key(item: tuple[str, dict]) -> int:
                req_id = item[0]
                return self._chunked_prefill_fairness.get(req_id, 0)

            pending_items.sort(key=_fairness_key)

        for req_id, state in pending_items:
            if req_id in completed_ids or req_id in errored_ids:
                continue

            remaining = state["remaining_tokens"]
            if not remaining:
                completed_ids.add(req_id)
                continue

            # Check if the request was aborted
            if req_id in self._pending_abort_ids:
                completed_ids.add(req_id)
                continue

            req = self.running.get(req_id)
            if req is None:
                completed_ids.add(req_id)
                continue

            # Skip actual insertion if BatchGenerator is not ready
            if self._batch_gen is None:
                continue

            # ── Fairness budget: cap total chunks per scheduling round ──
            # When decode requests are running, respect the budget to ensure
            # decode always gets at least 1 slot. The budget is per-round, so
            # decode requests are never starved across scheduling cycles.
            is_sarathi = self.config.enable_hybrid_prefill
            is_standard_chunked = state.get("chunk_size") is not None

            if (
                (is_sarathi or is_standard_chunked)
                and self._has_active_requests()
                and chunks_fed >= budget
            ):
                # Budget exhausted — remaining chunks deferred to next round.
                logger.debug(
                    f"Chunked prefill budget exhausted ({chunks_fed}/{budget}), "
                    f"deferring {len(self._pending_prefill)} pending prefills"
                )
                break

            # Determine chunk size:
            # - SCHED-2 standard: use the per-request chunk_size from state
            # - Sarathi hybrid: use hybrid_chunk_size from config
            # - Legacy fallback: hybrid_chunk_size
            _cs = state.get("chunk_size")
            chunk_size = _cs if _cs is not None else self.config.hybrid_chunk_size

            # Use semantic chunk boundaries when optimizer is available
            if (
                self._chunked_prefill_optimizer is not None
                and len(remaining) > chunk_size
            ):
                try:
                    chunks = self._chunked_prefill_optimizer.compute_optimal_chunks(
                        remaining,
                        chunk_size,
                        max_chunks=1,
                    )
                    if chunks:
                        semantic_end = chunks[0].end_token
                        if semantic_end > 0 and semantic_end < len(remaining):
                            chunk_size = max(semantic_end, chunk_size // 2)
                except Exception:
                    logger.debug("semantic chunking fallback", exc_info=True)
            chunk = remaining[:chunk_size]
            state["remaining_tokens"] = remaining[chunk_size:]

            # SCHED-2: update offset tracker for progress reporting
            if "offset" in state:
                state["offset"] += len(chunk)

            try:
                sp = req.sampling_params
                sampler = self._make_sampler(sp, state.get("all_prompt_tokens"))
                sm = self._make_state_machine(sp.stop, sp.stop_token_ids)

                # KV continuity: if we have a cached KV from a previous chunk,
                # use insert_segments instead of insert to carry forward the
                # KV cache. This prevents the model from seeing each chunk as
                # an isolated prompt without context.
                prev_kv = state.get("kv_cache")
                all_tokens = state.get("all_prompt_tokens")
                if prev_kv is not None and all_tokens is not None:
                    # offset was updated above to include this chunk's length,
                    # so all_tokens[:offset] covers everything through this chunk.
                    processed_count = state.get("offset", len(chunk))
                    uids = self._batch_gen.insert_segments(
                        segments=[[chunk]],
                        max_tokens=[sp.max_tokens],
                        caches=[prev_kv],
                        all_tokens=[all_tokens[:processed_count]],
                        samplers=[sampler],
                        state_machines=[sm],
                    )
                else:
                    uids = self._batch_gen.insert(
                        prompts=[chunk],
                        max_tokens=[sp.max_tokens],
                        samplers=[sampler],
                        state_machines=[sm],
                    )

                # Guard: BatchGenerator may return empty UIDs (e.g., batch full).
                # Without this guard, uids[0] raises IndexError, which is caught
                # by the outer except but leaves the old UID in BatchGenerator.
                if not uids:
                    logger.error(
                        "BatchGenerator.insert returned empty UIDs for chunked "
                        "prefill of %s (chunk %d tokens)",
                        req_id,
                        len(chunk),
                    )
                    req.set_finished(
                        RequestStatus.FINISHED_ERROR, reason="insert_failed"
                    )
                    self._uid_to_req.pop(getattr(req, "batch_uid", None), None)
                    errored_ids.add(req_id)
                    continue

                # Update tracking: each insert() returns a new UID.
                # Remove the old UID from both the scheduler mapping AND
                # the BatchGenerator itself.  If we only remove from
                # _uid_to_req, the old UID stays in BatchGenerator and
                # causes silent token loss (the old chunk's forward pass
                # produces output for a UID nobody tracks).
                #
                # KV continuity: extract the KV cache from the old chunk
                # before removing it, so the next chunk can use insert_segments
                # to carry forward the accumulated KV state.
                old_uid = getattr(req, "batch_uid", None)
                if old_uid is not None and old_uid != uids[0]:
                    self._uid_to_req.pop(old_uid, None)
                    try:
                        extracted = self._batch_gen.remove(
                            [old_uid], return_prompt_caches=True
                        )
                        if old_uid in extracted and extracted[old_uid] is not None:
                            state["kv_cache"] = extracted[old_uid][0]
                    except Exception:
                        logger.warning(
                            "Failed to remove old chunked UID %s from BatchGenerator "
                            "— latent slot leak possible",
                            old_uid,
                            exc_info=True,
                        )
                req.batch_uid = uids[0]
                self._uid_to_req[uids[0]] = req_id
                chunks_fed += 1
                self._chunked_prefill_chunks_processed += 1
                self._chunked_prefill_budget_used += 1
                self._chunked_prefill_fairness[req_id] = (
                    self._chunked_prefill_fairness.get(req_id, 0) + 1
                )

                total_prompt = state.get("total_prompt_len", 0)
                offset = state.get("offset", len(chunk))

                # ── Progress tracking: report via PrefillProgressTracker ──
                if self._prefill_tracker is not None and total_prompt > 0:
                    self._prefill_tracker.update(
                        req_id,
                        offset,
                        total_prompt,
                        self.model_id,
                    )

                # ── Emit prefill progress output ──
                # Create a synthetic RequestOutput with prefill_progress so the
                # client can show a progress bar during long chunked prefills.
                # Only emitted when there are remaining tokens (not on the final chunk,
                # since the final chunk will produce normal generation output).
                if total_prompt > 0 and state["remaining_tokens"]:
                    self._prefill_progress_outputs.append(
                        RequestOutput(
                            request_id=req_id,
                            finished=False,
                            prompt_tokens=offset,
                            completion_tokens=0,
                            prefill_progress=(offset, total_prompt),
                        )
                    )

                if not state["remaining_tokens"]:
                    completed_ids.add(req_id)
                    # Remove from progress tracker — prefill complete
                    if self._prefill_tracker is not None:
                        self._prefill_tracker.remove(req_id)
                    logger.debug(
                        f"Chunked prefill complete for {req_id}: "
                        f"final chunk {len(chunk)} tokens "
                        f"(total={total_prompt})"
                    )
                else:
                    logger.debug(
                        f"Chunked prefill step for {req_id}: "
                        f"{len(chunk)} tokens at offset {offset}, "
                        f"{len(state['remaining_tokens'])} remaining "
                        f"of {total_prompt}"
                    )
            except Exception as e:
                logger.error(
                    f"Failed to process chunked prefill for {req_id}: {e}",
                    exc_info=True,
                )
                # ── Error handling: abort entire request on chunk failure ──
                req.set_finished(RequestStatus.FINISHED_ERROR, reason="prefill_error")
                failed_uid = getattr(req, "batch_uid", None)
                self._uid_to_req.pop(failed_uid, None)
                # Decrement _total_prompt_tokens — the counter was incremented
                # when the request was first inserted (line 1836).  This abort
                # means the prompt tokens are wasted and should not be counted.
                self._total_prompt_tokens = max(
                    0, self._total_prompt_tokens - getattr(req, "num_prompt_tokens", 0)
                )
                # Remove the failed UID from BatchGenerator to prevent GPU
                # memory leak (the old chunk's KV cache stays allocated
                # until explicitly removed).
                if failed_uid is not None and self._batch_gen is not None:
                    try:
                        self._batch_gen.remove([failed_uid])
                    except Exception:
                        logger.debug(
                            "Failed to remove failed-chunk UID %s from BatchGenerator",
                            failed_uid,
                            exc_info=True,
                        )
                errored_ids.add(req_id)

        # ── Cleanup completed and errored requests ──
        # GAP 1.3: Decrement active partial prefill counter for each completed/errored request.
        for rid in completed_ids:
            popped = self._pending_prefill.pop(rid, None)
            self._chunked_prefill_fairness.pop(rid, None)
            self._chunked_prefill_enqueued_at.pop(rid, None)
            if popped is not None:
                self._active_partial_prefills = max(
                    0, self._active_partial_prefills - 1
                )

        for rid in errored_ids:
            popped = self._pending_prefill.pop(rid, None)
            self._chunked_prefill_fairness.pop(rid, None)
            self._chunked_prefill_enqueued_at.pop(rid, None)
            self._chunked_prefill_failed_ids.append(rid)
            # Only decrement if this entry was still in _pending_prefill.
            # The timeout abort path already decremented via _pop_pending_prefill.
            # popped will be None here, so we correctly skip the second decrement.
            if popped is not None:
                self._active_partial_prefills = max(
                    0, self._active_partial_prefills - 1
                )

        # ── Prometheus observation ──
        try:
            from yunshu_gateway.middleware.prometheus_exporter import (
                get_prometheus_metrics,
            )

            pm = get_prometheus_metrics()
            pm.set_gauge(
                "chunked_prefill_active_chunks", float(len(self._pending_prefill))
            )
            pm.set_counter(
                "chunked_prefill_total_chunks_processed",
                self._chunked_prefill_chunks_processed,
            )
            pm.set_gauge(
                "chunked_prefill_budget_used", float(self._chunked_prefill_budget_used)
            )
            pm.set_gauge(
                "chunked_prefill_budget_limit",
                float(self.config.chunked_prefill_budget),
            )
        except Exception:
            pass  # Prometheus not available in unit tests

    def _hybrid_prefill_step(self, outputs: list) -> list:
        """Sarathi-style hybrid chunked prefill interleaving.

        When hybrid prefill is enabled and there are pending partial prefills,
        this method interleaves prefill chunk insertion with decode steps.
        The pattern is:

            for each pending prefill chunk:
                1. Insert next prefill chunk into BatchGenerator
                2. Run next() → prefill the chunk + decode all running requests
                3. Process and collect decode outputs
                4. Repeat until no more pending chunks or limit reached

        This prevents a long prefill from starving running decode requests
        by ensuring that after each chunk, all active requests get a decode
        token. The trade-off is slightly lower prefill throughput for
        significantly better decode latency under load.

        Args:
            outputs: Accumulated outputs from the current step so far.

        Returns:
            Updated outputs list with decode outputs from interleaving.
        """
        # Safety limit: don't spend more than N hybrid iterations per step
        # to avoid starving the event loop.
        max_hybrid_iters = 32
        iters = 0

        while self._pending_prefill and iters < max_hybrid_iters:
            iters += 1

            # Feed exactly one pending prefill chunk
            self._process_pending_prefill()

            # If nothing was inserted (all remaining chunks already consumed),
            # break out.
            if not self._pending_prefill and not self._has_active_requests():
                break

            try:
                # next() prefills the chunk we just inserted AND decodes
                # all active requests (including the one being prefilled
                # if this is its final chunk).
                prompt_responses, gen_responses = self._batch_gen.next()

                if prompt_responses:
                    self._process_prefill_responses(prompt_responses)

                if gen_responses:
                    new_outputs = self._process_responses(gen_responses)
                    outputs.extend(new_outputs)

                # Also do any configured extra decode steps
                for _ in range(self.config.stream_interval):
                    if not self._has_active_requests():
                        break
                    try:
                        gen_responses = self._batch_gen.next_generated()
                    except StopIteration:
                        break
                    if gen_responses:
                        new_outputs = self._process_responses(gen_responses)
                        outputs.extend(new_outputs)
                    else:
                        break

            except Exception as e:
                logger.error(f"Hybrid prefill step error: {e}", exc_info=True)
                from .exceptions import is_cache_corruption_error

                if is_cache_corruption_error(e):
                    logger.warning(
                        "Cache corruption in hybrid prefill — resetting BatchGenerator"
                    )
                    self.deep_reset()
                break

        return outputs

    def _process_responses(self, responses: list) -> list[RequestOutput]:
        """Distribute GenerationBatch.Response to per-request outputs.

        Includes ITL tracking and progressive KV quantization.
        Deduplicates responses by UID to prevent double-processing if
        the BatchGenerator returns duplicate Response UIDs.
        """
        _now = time.perf_counter()
        outputs = []
        processed_uids: set[str] = set()
        for resp in responses:
            uid = resp.uid
            if uid in processed_uids:
                logger.debug("Skipping duplicate response UID: %s", uid)
                continue
            processed_uids.add(uid)
            req_id = self._uid_to_req.get(uid)
            if req_id is None:
                continue

            req = self.running.get(req_id)
            if req is None:
                # Request was aborted or preempted between steps — clean up
                # the stale UID mapping to prevent repeated lookups.
                self._uid_to_req.pop(uid, None)
                continue

            is_stop = resp.finish_reason == "stop"
            is_finished = resp.finish_reason is not None

            # Inflight prefix sharing: update the tracker with the KV cache
            # ref from the first response.  The entry was registered in
            # engine_core.add_request() with kv_cache_ref=None.  After the
            # first forward pass, resp.prompt_cache contains the full prompt
            # KV, which concurrent requests can now reuse.
            if (
                not is_stop
                and hasattr(resp, "prompt_cache")
                and resp.prompt_cache is not None
            ):
                try:
                    from .inflight_prefix_sharing import get_inflight_tracker

                    _tracker = get_inflight_tracker()
                    _entry = _tracker._entries.get(req_id)
                    if _entry is not None and _entry.kv_cache_ref is None:
                        _entry.kv_cache_ref = resp.prompt_cache
                        logger.debug(
                            "inflight prefix KV updated for req=%s (%d tokens)",
                            req_id[:12],
                            len(_entry.token_ids),
                        )
                except Exception:
                    logger.debug("inflight KV update failed", exc_info=True)

            token_text = ""
            new_token_ids = []

            if not is_stop:
                req.append_token(resp.token)
                new_token_ids.append(resp.token)

                detok = self._detokenizers.get(req_id)
                if detok is not None:
                    detok.add_token(resp.token)
                    token_text = detok.last_segment
                else:
                    token_text = self.tokenizer.decode([resp.token])

                req.output_text += token_text

                # H2O: Update attention scores with heuristic after each decode token.
                # The heuristic gives higher scores to more recent KV blocks,
                # approximating which blocks the model "pays attention to".
                if self._attention_score_tracker is not None:
                    total_tokens = len(req.prompt_token_ids or []) + len(
                        req.output_token_ids
                    )
                    # Estimate block count: each block holds ~256 tokens (typical page size)
                    num_blocks = max(1, total_tokens // 256)
                    self._attention_score_tracker.update_heuristic(req_id, num_blocks)

                # ITL tracking: record inter-token latency per request
                last_tok = self._last_token_time.get(req_id)
                if last_tok is not None:
                    itl = _now - last_tok
                    if 0 < itl < 10:  # filter outliers
                        itl_list = self._itl_samples.get(req_id)
                        if itl_list is None:
                            self._itl_samples[req_id] = itl_list = []
                        itl_list.append(itl)
                self._last_token_time[req_id] = _now
            elif is_finished:
                pass

            logprobs = None
            if hasattr(resp, "logprobs") and resp.logprobs is not None:
                try:
                    logprobs = resp.logprobs
                except Exception:
                    logger.debug("logprobs extraction failed", exc_info=True)

            current_state = getattr(resp, "current_state", "normal") or "normal"
            finish_reason = resp.finish_reason

            # ── Thinking-segment KV tracking ──
            is_thinking = current_state == "reasoning"
            ts = self._thinking_state.get(req_id)
            if ts is None:
                ts = {
                    "in_thinking": False,
                    "thinking_start_idx": None,
                    "was_in_thinking": False,
                    "total_reasoning_tokens": 0,
                }
                self._thinking_state[req_id] = ts

            # Detect thinking-start transition (normal → reasoning)
            if is_thinking and not ts["in_thinking"]:
                # Only record thinking start if we actually appended a token.
                # When is_stop=True, no token was appended, so there's nothing
                # to mark as the start of a thinking segment.
                if not is_stop:
                    ts["thinking_start_idx"] = len(req.output_token_ids) - 1
                    ts["in_thinking"] = True

            # Detect thinking-end transition (reasoning → normal):
            # store the completed thinking segment in the KV substore.
            if (
                ts["in_thinking"]
                and not is_thinking
                and ts["thinking_start_idx"] is not None
            ):
                thinking_end_idx = len(req.output_token_ids)
                thinking_tokens = list(
                    req.output_token_ids[ts["thinking_start_idx"] : thinking_end_idx]
                )
                # Accumulate reasoning tokens from this completed segment
                ts["total_reasoning_tokens"] += len(thinking_tokens)
                context_tokens = (
                    list(req.prompt_token_ids) if req.prompt_token_ids else []
                )
                # Extract KV data from response if available, else None
                kv_data = getattr(resp, "prompt_cache", None)
                try:
                    step_hash = self._thinking_store.store(
                        conversation_id=req_id,
                        thinking_tokens=thinking_tokens,
                        context_tokens=context_tokens,
                        kv_data=kv_data,
                    )
                    if step_hash is not None:
                        logger.debug(
                            f"Thinking segment stored for {req_id}: "
                            f"{len(thinking_tokens)} tokens, hash={step_hash}"
                        )
                except Exception as e:
                    logger.debug(f"Thinking segment store failed for {req_id}: {e}")
                ts["in_thinking"] = False
                ts["thinking_start_idx"] = None

            ts["was_in_thinking"] = is_thinking

            # Thinking budget enforcement
            thinking_proc = self._thinking_processors.get(req_id)
            if thinking_proc is not None and not is_finished:
                budget_result = thinking_proc.process_token(current_state)
                if budget_result["force_stop"]:
                    finish_reason = "stop"
                    is_finished = True
                    # Schedule BatchGenerator UID removal on next step
                    if uid not in self._uids_to_remove:
                        self._uids_to_remove.append(uid)

            output = RequestOutput(
                request_id=req_id,
                new_token_ids=new_token_ids,
                new_text=token_text,
                output_token_ids=list(req.output_token_ids),
                output_text=req.output_text,
                finished=is_finished,
                finish_reason=finish_reason,
                prompt_tokens=req.num_prompt_tokens,
                completion_tokens=req.num_output_tokens,
                logprobs=logprobs,
                current_state=current_state,
                reasoning_tokens=(
                    (ts.get("total_reasoning_tokens", 0) or 0)
                    + (
                        len(req.output_token_ids) - ts["thinking_start_idx"]
                        if ts
                        and ts.get("thinking_start_idx") is not None
                        and ts.get("in_thinking")
                        else 0
                    )
                ),
                cached_tokens=getattr(req, "cached_tokens", 0),
            )
            outputs.append(output)

            if finish_reason:
                # Finalize detokenizer
                detok = self._detokenizers.pop(req_id, None)
                self._thinking_processors.pop(req_id, None)
                self._thinking_state.pop(req_id, None)
                # Cleanup speculative decoding state
                self._cleanup_spec_state(req_id)
                # Clean up chunked prefill state (request may finish while
                # still in the middle of chunked prefill, e.g. thinking budget overflow)
                self._pop_pending_prefill(req_id)
                # H2O: Remove attention score tracking for finished request
                if self._attention_score_tracker is not None:
                    self._attention_score_tracker.remove_request(req_id)
                # Cleanup chunked prefill production tracking
                self._chunked_prefill_fairness.pop(req_id, None)
                self._chunked_prefill_enqueued_at.pop(req_id, None)
                # Unregister mRoPE delta
                if uid is not None:
                    self._rope_delta_mgr.unregister(uid)
                if detok is not None:
                    try:
                        detok.finalize()
                        final = detok.last_segment
                        if final:
                            req.output_text += final
                            output.new_text += final
                            output.output_text = req.output_text
                    except Exception:
                        logger.debug("detokenizer step failed", exc_info=True)

                # GUARANTEED string-level stop truncation. The
                # token-based SequenceStateMachine only catches stops at token
                # boundaries it registered (bare + space-prefixed), so a stop
                # after some other char ("(END") still leaks. Truncate the final
                # output at the first occurrence of any stop string so it never
                # appears in the response (OpenAI semantics).
                try:
                    _stops = getattr(req.sampling_params, "stop", None) or []
                    if _stops and req.output_text:
                        _cut = len(req.output_text)
                        for _s in _stops:
                            if _s:
                                _i = req.output_text.find(_s)
                                if _i != -1:
                                    _cut = min(_cut, _i)
                        if _cut < len(req.output_text):
                            req.output_text = req.output_text[:_cut]
                            output.output_text = req.output_text
                except Exception:
                    logger.debug("stop-truncation failed", exc_info=True)

                # Update request state
                status_map = {
                    "stop": RequestStatus.FINISHED_STOPPED,
                    "length": RequestStatus.FINISHED_LENGTH,
                }
                req.set_finished(
                    status_map.get(finish_reason, RequestStatus.FINISHED_STOPPED),
                    reason=finish_reason,
                )
                # Capture this (just-finished) request's prompt-prefix KV NOW —
                # before its uid is popped from _uid_to_req and its KV dropped from
                # the BatchGenerator — so even SHORT requests' prefixes get cached
                # for the next same-prefix request (deterministic prefix reuse).
                self._save_one_prefix(uid, req_id)
                self._uid_to_req.pop(uid, None)
                # Schedule removal from BatchGenerator so subsequent decode
                # steps within the same step() call don't waste GPU forward
                # passes on a finished request.
                if uid not in self._uids_to_remove:
                    self._uids_to_remove.append(uid)

                # Deferred cache clearing
                self._deferred_clear_at = (
                    self._step_counter + self._DEFERRED_CLEAR_DELAY
                )

                self._total_completion_tokens += req.num_output_tokens

                # Record in ServerMetrics
                if self._server_metrics is not None:
                    self._server_metrics.record_request_complete(
                        prompt_tokens=req.num_prompt_tokens,
                        completion_tokens=req.num_output_tokens,
                        cached_tokens=req.cached_tokens,
                        prefill_duration=req.prefill_duration,
                        generation_duration=req.generation_duration,
                        model_id=self.model_id,
                    )
                    # Record ITL samples to histogram (ITL-1)
                    itl_list = self._itl_samples.pop(req_id, [])
                    if itl_list and hasattr(self._server_metrics, "record_itl"):
                        for sample in itl_list:
                            self._server_metrics.record_itl(sample)

                # Cleanup ITL tracking state
                self._last_token_time.pop(req_id, None)
                self._itl_samples.pop(req_id, None)

        return outputs

    def _process_prefill_responses(self, responses: list) -> None:
        """Handle prompt processing completion."""
        for resp in responses:
            uid = resp.uid
            req_id = self._uid_to_req.get(uid)
            if req_id:
                req = self.running.get(req_id)
                if req is None:
                    continue
                if req.status in (RequestStatus.WAITING, RequestStatus.PREFILLING):
                    req.status = RequestStatus.RUNNING
                if hasattr(resp, "end_of_prompt") and resp.end_of_prompt:
                    if req.status in (RequestStatus.WAITING, RequestStatus.PREFILLING):
                        req.status = RequestStatus.RUNNING

                    # Mark prefill complete
                    req.prefill_end = time.monotonic()
                    req.generation_start = time.monotonic()

                    # Update prefill tracker (auto-removes on complete)
                    if self._prefill_tracker is not None:
                        self._prefill_tracker.update(
                            req.request_id,
                            req.num_prompt_tokens,
                            req.num_prompt_tokens,
                            self.model_id,
                        )

                    # Cache encoder outputs for encoder-decoder models.
                    # If the response carries encoder hidden states (e.g. from a
                    # Whisper/T5-style encoder-decoder model), store them in the
                    # encoder cache for potential reuse in subsequent requests.
                    encoder_output = getattr(resp, "encoder_outputs", None)
                    if encoder_output is not None:
                        self._encoder_cache.put(req_id, encoder_output)

    def _process_aborts(self) -> None:
        """Process deferred abort requests."""
        if not self._pending_abort_ids:
            return

        abort_uids = []
        for req_id in list(self._pending_abort_ids):
            req = self.running.get(req_id)
            if req and req.batch_uid is not None:
                abort_uids.append(req.batch_uid)

        if abort_uids and self._batch_gen:
            self._batch_gen.remove(abort_uids)
        # Dedup: prevent the step preamble's _uids_to_remove from removing
        # these UIDs a second time (double removal from BatchGenerator).
        if self._uids_to_remove:
            abort_set = set(abort_uids)
            self._uids_to_remove = [
                u for u in self._uids_to_remove if u not in abort_set
            ]

        for req_id in list(self._pending_abort_ids):
            req = self.requests.get(req_id)
            if req:
                req.set_finished(RequestStatus.FINISHED_ABORTED, reason="abort")
                uid = getattr(req, "batch_uid", None)
                self._uid_to_req.pop(uid, None)
                if uid is not None:
                    self._rope_delta_mgr.unregister(uid)
                if req_id in self.running:
                    self._total_prompt_tokens = max(
                        0,
                        self._total_prompt_tokens
                        - getattr(req, "num_prompt_tokens", 0),
                    )
                    self._total_completion_tokens += getattr(
                        req, "num_output_tokens", 0
                    )
            self.running.pop(req_id, None)
            self._detokenizers.pop(req_id, None)
            self._thinking_processors.pop(req_id, None)
            self._thinking_state.pop(req_id, None)
            self._pop_pending_prefill(req_id)
            self._cleanup_spec_state(req_id)
            # H2O: cleanup attention score tracking
            if self._attention_score_tracker is not None:
                self._attention_score_tracker.remove_request(req_id)
            # Chunked prefill production tracking cleanup
            self._chunked_prefill_fairness.pop(req_id, None)
            self._chunked_prefill_enqueued_at.pop(req_id, None)

        # Bug 3 fix: save abort IDs before clearing so we can also sweep
        # the waiting queue for aborted requests that haven't entered running yet.
        _aborted_ids_snapshot = set(self._pending_abort_ids)
        self._pending_abort_ids.clear()

        # Bug 3 fix: also remove aborted requests from the waiting queue.
        # If a request was aborted while still waiting (not yet in running),
        # _process_aborts would miss it because only self.running is checked
        # above. On the next _schedule_waiting, the cleared _pending_abort_ids
        # means the request gets scheduled anyway. Fix: sweep the waiting queue.
        if self.waiting and _aborted_ids_snapshot:
            try:
                remaining = []
                with self.waiting._lock:
                    for entry in self.waiting._heap:
                        item = self.waiting._extract_item(entry)
                        if (
                            hasattr(item, "request_id")
                            and item.request_id in _aborted_ids_snapshot
                        ):
                            item.set_finished(
                                RequestStatus.FINISHED_ABORTED, reason="abort"
                            )
                            self._failed_insert_ids.append(item.request_id)
                            # Clean up pending prefill entry if this request
                            # was chunked and waiting for its next chunk.
                            self._pop_pending_prefill(item.request_id)
                        else:
                            remaining.append(entry)
                    if len(remaining) != len(self.waiting._heap):
                        self.waiting._heap = remaining
                        import heapq

                        heapq.heapify(self.waiting._heap)
            except Exception:
                logger.debug("waiting queue abort sweep failed", exc_info=True)

    def _maybe_clear_cache(self) -> None:
        """Deferred Metal cache cleanup."""
        should_clear = False
        if (
            self.config.cache_cleanup_interval > 0
            and self._step_counter % self.config.cache_cleanup_interval == 0
        ):
            should_clear = True
        if (
            self._deferred_clear_at is not None
            and self._step_counter >= self._deferred_clear_at
        ):
            should_clear = True
            self._deferred_clear_at = None

        if should_clear and not self.running:
            try:
                import mlx.core as mx

                mx.synchronize()
                mx.clear_cache()
            except Exception:
                logger.debug("failed", exc_info=True)

    def _maybe_evict_kv_cache(self) -> None:
        """Proactive memory pressure eviction (C12).

        Called periodically from step(). When active memory exceeds
        the configured threshold, evicts the oldest cached KV blocks.
        """
        try:
            import mlx.core as mx

            active_mem = mx.get_active_memory()
            from .utils.hardware import get_hardware_info

            hw = get_hardware_info()
            total_mem = hw.total_memory_bytes
            if total_mem <= 0:
                return
            usage = active_mem / total_mem
            threshold = self.config.memory_guard_soft_limit
            if (
                usage >= threshold
                and hasattr(self, "_prefix_cache")
                and self._prefix_cache is not None
            ):
                # Protect prefix cache entries referenced by running requests.
                # Evicting a prefix still in active use corrupts the KV state of
                # decode-phase requests that share that prefix.
                _active_prefix_tokens = set()
                for _r in self.running.values():
                    _pt = getattr(_r, "prompt_token_ids", None)
                    if _pt:
                        _active_prefix_tokens.add(tuple(_pt))

                # Set a temporary block_evict_checker that skips entries whose
                # prompt tokens match an active request, preventing corruption.
                _original_checker = getattr(
                    self._prefix_cache, "_block_evict_checker", None
                )

                def _active_request_checker(block_hash):
                    # Return True if the block is safe to evict.
                    _cached = getattr(self._prefix_cache, "_block_hashes", None)
                    if not _cached:
                        return True
                    for _idx, _bhs in enumerate(_cached):
                        if block_hash in _bhs:
                            _prompts = getattr(self._prefix_cache, "_prompts", None)
                            if _prompts and _idx < len(_prompts):
                                import numpy as np

                                try:
                                    _cached_tuple = tuple(
                                        int(t)
                                        for t in np.array(_prompts[_idx]).flatten()
                                    )
                                    return _cached_tuple not in _active_prefix_tokens
                                except Exception:
                                    # fail-CLOSED, not
                                    # fail-open. Prior `pass` + fallthrough to
                                    # `return True` silently marked
                                    # an active KV block as evictable on any
                                    # conversion error → in-flight request KV
                                    # corruption. Return False to KEEP the
                                    # block rather than evict it.
                                    return False
                            break
                    return True

                self._prefix_cache._block_evict_checker = _active_request_checker
                try:
                    evicted = self._prefix_cache.evict_under_pressure(
                        threshold_pct=threshold * 100
                    )
                    if evicted > 0:
                        logger.info(
                            f"Memory pressure eviction: {evicted} KV blocks freed "
                            f"(usage {usage:.1%})"
                        )
                finally:
                    self._prefix_cache._block_evict_checker = _original_checker
        except Exception:
            logger.debug("failed", exc_info=True)

    def _cleanup_finished(self) -> None:
        """Remove finished requests from running dict."""
        uids_to_remove = []
        for req_id in list(self.running.keys()):
            req = self.running[req_id]
            if RequestStatus.is_finished(req.status):
                self.running.pop(req_id, None)
                self.finished_ids.add(req_id)
                # Clean up UID mapping to prevent stale lookups
                uid = getattr(req, "batch_uid", None)
                if uid is not None:
                    self._uid_to_req.pop(uid, None)
                    uids_to_remove.append(uid)
                    self._saved_prefix_uids.discard(uid)
                self._kv_prefix_hashes.pop(req_id, None)
                # Clean up chunked prefill state for finished/aborted requests.
                # Without this, _pending_prefill leaks when a request finishes
                # while still in the middle of chunked prefill.
                self._pop_pending_prefill(req_id)
                # Evict encoder cache entry for finished request.
                # The encoder output is no longer needed once the decoder
                # has completed generation.
                self._encoder_cache.evict(req_id)
                # H2O: cleanup attention score tracking for finished request
                if self._attention_score_tracker is not None:
                    self._attention_score_tracker.remove_request(req_id)
                # Chunked prefill production tracking cleanup
                self._chunked_prefill_fairness.pop(req_id, None)
                self._chunked_prefill_enqueued_at.pop(req_id, None)
                # Clean per-request state that may not have been cleaned by
                # _process_responses (e.g. chunked prefill timeout paths).
                self._detokenizers.pop(req_id, None)
                self._thinking_processors.pop(req_id, None)
                self._thinking_state.pop(req_id, None)
                self._spec_drafts.pop(req_id, None)
                self._spec_draft_start_pos.pop(req_id, None)
                self._spec_stats.pop(req_id, None)
                self._spec_draft_cache_snapshots.pop(req_id, None)
        # Free BatchGenerator's internal resources (KV cache, attention state)
        # for finished requests. Without this, GPU memory leaks indefinitely.
        if (
            uids_to_remove
            and hasattr(self, "_batch_gen")
            and self._batch_gen is not None
        ):
            try:
                self._batch_gen.remove(uids_to_remove)
            except Exception:
                logger.debug(
                    "batch_gen.remove failed in cleanup_finished", exc_info=True
                )

        # Clear _uids_to_remove to prevent double-removal at the start of the
        # next step().  _process_responses appends finished UIDs to
        # _uids_to_remove so that the inner decode loop (step 6) skips them,
        # but _cleanup_finished removes those same UIDs from the BatchGenerator
        # right here.  Without this clear, the next step's preamble would try
        # to remove them a second time.
        if uids_to_remove and self._uids_to_remove:
            removed_set = set(uids_to_remove)
            self._uids_to_remove = [
                u for u in self._uids_to_remove if u not in removed_set
            ]

    def _create_detokenizer(self):
        """Create a per-request detokenizer instance.

        CRITICAL: was `return self.tokenizer.detokenizer` —
        that's a SINGLETON @property on mlx-lm's tokenizer wrapper. All
        concurrent requests shared the same detokenizer; one request's
        `.reset()` wiped another's byte buffer mid-stream, corrupting
        streaming output. CLAUDE.md explicitly warns: "never pool".

        Now instantiate a fresh detokenizer via the tokenizer's
        detokenizer_class when available; fall back to the singleton +
        reset() only when no class hook exists (best-effort legacy path).
        """
        if self.tokenizer is None:
            return None
        # a fresh `detok_cls(tokenizer)` REBUILDS the per-vocab
        # token map every request — ~129ms on Qwen's 151k vocab, which profiling
        # showed was 32% of decode wall-time under concurrency. The token map +
        # byte decoder are IMMUTABLE and identical across requests; only the
        # streaming state (offset/_unflushed/text/tokens) is per-request. So build
        # ONE template (paying the rebuild once), then per request shallow-copy it
        # (shares the immutable maps by reference — O(1), not O(vocab)) and reset()
        # for fresh state. Each request still gets its OWN object → no shared-state
        # race; only read-only structures are shared.
        import copy as _copy

        # The class is exposed as `_detokenizer_class` (what the `.detokenizer`
        # property instantiates); the public `detokenizer_class` attr is usually
        # None, which is why the old code fell through to the rebuild-every-time
        # `.detokenizer` property.
        detok_cls = getattr(self.tokenizer, "_detokenizer_class", None) or getattr(
            self.tokenizer, "detokenizer_class", None
        )
        tmpl = getattr(self, "_detok_template", None)
        if tmpl is None and detok_cls is not None:
            try:
                tmpl = detok_cls(self.tokenizer)
                self._detok_template = tmpl
            except Exception:
                tmpl = None
        if tmpl is not None:
            try:
                new = _copy.copy(tmpl)
                new.reset()
                return new
            except Exception:
                logger.debug("detok template copy failed; rebuilding", exc_info=True)
                if detok_cls is not None:
                    try:
                        return detok_cls(self.tokenizer)
                    except Exception:
                        pass
        # Fallback: singleton + reset (legacy path; multi-request races possible)
        detok = self.tokenizer.detokenizer
        detok.reset()
        return detok

    def _make_sampler(
        self, sp: SamplingParams, prompt_token_ids: list[int] | None = None
    ):
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        # the engine-loop is the concurrency path, so it is exactly where
        # mlx-lm's PRNG trap bites hardest. `make_sampler(temp>0)` routes through
        # `categorical_sampling`, which is wrapped with
        # `@mx.compile(inputs=mx.random.state, outputs=mx.random.state)`: the compile
        # cache traps the FIRST call's PRNG state, so every later temp>0 request —
        # and concurrent ones in the same batch — collapse to correlated/identical
        # token streams, and `seed` becomes a global no-op (last writer wins). Route
        # temp>0 through `_build_temp_sampler` (a numpy/Gumbel sampler with an
        # explicit per-request RNG key — no shared compile cache), so each running
        # sequence samples independently. Greedy (temp==0) has no RNG, so it stays
        # on mlx-lm's argmax `make_sampler`. This mirrors the fix on
        # the default fast paths (batched_engine), completing the sibling sweep into
        # the concurrency engine-loop.
        if sp.temperature is not None and sp.temperature > 1e-6:
            from .batched_engine import _build_temp_sampler

            base_sampler = _build_temp_sampler(
                temperature=sp.temperature,
                top_p=sp.top_p,
                top_k=sp.top_k,
                min_p=sp.min_p,
                seed=sp.seed,
                xtc_probability=getattr(sp, "xtc_probability", 0.0),
                xtc_threshold=getattr(sp, "xtc_threshold", 0.0),
            )
        else:
            base_sampler = make_sampler(
                temp=sp.temperature,
                top_p=sp.top_p,
                top_k=sp.top_k,
                min_p=sp.min_p,
                xtc_probability=getattr(sp, "xtc_probability", 0.0),
                xtc_threshold=getattr(sp, "xtc_threshold", 0.0),
            )

        # Build logits processors. repetition_penalty (20-token window) + logit_bias come
        # from mlx-lm and match the fast path. Frequency/presence penalty does
        # NOT — mlx-lm's make_*_penalty uses a 20-token sliding window, and this engine's
        # _LogitsProcessorSampler._tokens is seeded with the FULL prompt, so the engine
        # loop penalized the prompt tail and only a 20-token window. The fast path counts
        # over the GENERATED completion ONLY, across the full history. Replace the two
        # penalties with the same incremental generated-only closure (n_prompt = prompt
        # length here, since tokens = prompt + generated), leaving rep-penalty/logit_bias.
        logits_processors = make_logits_processors(
            repetition_penalty=sp.repetition_penalty
            if sp.repetition_penalty != 1.0
            else None,
            logit_bias=getattr(sp, "logit_bias", None),
        )
        _fp = sp.frequency_penalty if sp.frequency_penalty != 0.0 else 0.0
        _pp = sp.presence_penalty if sp.presence_penalty != 0.0 else 0.0
        if _fp != 0.0 or _pp != 0.0:
            if logits_processors is None:
                logits_processors = []
            _fp_state: dict[str, object] = {"counts": {}, "last_len": -1}
            _fp_nprompt = len(prompt_token_ids or [])

            def _freq_pres_penalty(
                tokens, logits, fp=_fp, pp=_pp, n_prompt=_fp_nprompt, _st=_fp_state
            ):
                counts: dict[int, int] = _st["counts"]  # type: ignore[assignment]
                last_len = int(_st["last_len"])  # type: ignore[arg-type]
                cur_len = len(tokens)
                if cur_len <= n_prompt:
                    _st["last_len"] = cur_len
                    return logits
                if cur_len < last_len or last_len < n_prompt:
                    counts = {}
                    for t in tokens[n_prompt:]:
                        counts[int(t)] = counts.get(int(t), 0) + 1
                    _st["counts"] = counts
                else:
                    start = max(last_len, n_prompt)
                    for t in tokens[start:]:
                        counts[int(t)] = counts.get(int(t), 0) + 1
                _st["last_len"] = cur_len
                for tid, cnt in counts.items():
                    if fp != 0.0:
                        logits[..., tid] = logits[..., tid] - fp * cnt
                    if pp != 0.0 and cnt > 0:
                        logits[..., tid] = logits[..., tid] - pp
                return logits

            logits_processors.append(_freq_pres_penalty)

        # SAMP-2: Append user-provided custom logits processors
        custom_procs = getattr(sp, "logits_processors", None)
        if custom_procs:
            if logits_processors is None:
                logits_processors = []
            logits_processors.extend(custom_procs)

        # wire min_tokens / ignore_eos / suppress_tokens. These are
        # declared on SamplingParams and honored on the fast path
        # (batched_engine._generate_fast) but were silently DROPPED here in the
        # opt-in engine loop. The processors receive _tokens = prompt + generated,
        # so the generated count is len(tokens) - n_prompt.
        _eos_ids = (
            list(self.tokenizer.eos_token_ids)
            if hasattr(self.tokenizer, "eos_token_ids")
            else []
        )
        _n_prompt = len(prompt_token_ids or [])

        def _ban(logits, ids):
            vocab = logits.shape[-1]
            for tid in ids:
                if 0 <= tid < vocab:
                    logits[..., tid] = -float("inf")
            return logits

        _supp = getattr(sp, "suppress_tokens", None)
        if _supp:
            _sup_ids = [int(t) for t in _supp]
            if logits_processors is None:
                logits_processors = []
            logits_processors.append(lambda _t, lg, ids=_sup_ids: _ban(lg, ids))
        if getattr(sp, "ignore_eos", False) and _eos_ids:
            if logits_processors is None:
                logits_processors = []
            logits_processors.append(lambda _t, lg, ids=_eos_ids: _ban(lg, ids))
        _min_tok = int(getattr(sp, "min_tokens", 0) or 0)
        # Skip min_tokens EOS-masking when constrained — the constraint governs
        # termination and masking EOS at its DONE state yields invalid output
        # (same reasoning as the fast-path min_tokens guard).
        if (
            _min_tok > 0
            and _eos_ids
            and getattr(sp, "json_schema", None) is None
            and getattr(sp, "grammar", None) is None
        ):
            if logits_processors is None:
                logits_processors = []

            def _min_tokens_proc(
                toks, lg, eos=_eos_ids, floor=_min_tok, nprompt=_n_prompt
            ):
                if (len(toks) - nprompt) < floor:
                    return _ban(lg, eos)
                return lg

            logits_processors.append(_min_tokens_proc)

        if logits_processors:
            # Wrap sampler to apply logits processors before sampling.
            # Logits processors take (tokens, logits) and return modified logits.
            # We store generated tokens per-request via _generation_tokens.
            sampler = _LogitsProcessorSampler(
                base_sampler, logits_processors, prompt_token_ids
            )
        else:
            sampler = base_sampler

        # JSON schema / grammar constrained generation
        json_schema = getattr(sp, "json_schema", None)
        grammar = getattr(sp, "grammar", None)
        if json_schema is not None or grammar is not None:
            from .grammar_constraint import ConstraintFactory
            from .json_schema import ConstrainedSampler, JsonSchemaConstraint

            # grammar field takes priority for non-JSON types (regex, choice, cfg)
            if grammar is not None and isinstance(grammar, dict):
                gtype = grammar.get("type")
                if gtype in ("regex", "choice", "cfg"):
                    try:
                        if gtype == "regex":
                            gpayload = grammar.get("pattern", "")
                        elif gtype == "choice":
                            gpayload = grammar.get("choices", [])
                        elif gtype == "cfg":
                            gpayload = grammar.get("grammar", "")
                        else:
                            gpayload = None

                        constraint = ConstraintFactory.create(
                            gtype, gpayload, self.tokenizer
                        )
                        return ConstrainedSampler(sampler, constraint, self.tokenizer)
                    except Exception:
                        logger.debug(
                            "grammar constraint setup failed, falling back",
                            exc_info=True,
                        )
                elif gtype == "json":
                    schema = grammar.get("schema")
                    constraint = JsonSchemaConstraint(schema)
                    return ConstrainedSampler(sampler, constraint, self.tokenizer)

            # Fallback to json_schema field
            if json_schema is not None:
                from .json_schema import make_constrained_sampler

                schema = json_schema if isinstance(json_schema, dict) else None
                constrained_sampler = make_constrained_sampler(
                    base_sampler=sampler if logits_processors else base_sampler,
                    schema=schema,
                    tokenizer=self.tokenizer,
                    mode="json_schema" if schema else "json_object",
                )
                return constrained_sampler

        return sampler

    def _make_state_machine(
        self, stop: list[str] | None = None, stop_token_ids: list[int] | None = None
    ):
        from mlx_lm.generate import SequenceStateMachine

        eos_ids = (
            list(self.tokenizer.eos_token_ids)
            if hasattr(self.tokenizer, "eos_token_ids")
            else []
        )
        common_stops = [((t,), None) for t in eos_ids]
        for w in stop or []:
            if not w:
                continue  # Skip empty stop strings — they cause immediate stop
            # the model emits SPACE-PREFIXED tokens mid-stream
            # (" C", " banana"), so encode(w) of a BARE stop ("C", "banana")
            # produces a token sequence the model never generates → the state
            # machine silently never fires and the stop text leaks. Register both
            # the bare and the space-prefixed tokenizations so a stop that lands
            # after whitespace (the common case) is caught.
            _seen: set = set()
            for variant in (w, " " + w):
                t = tuple(self.tokenizer.encode(variant, add_special_tokens=False))
                if t and t not in _seen:
                    _seen.add(t)
                    common_stops.append((t, None))
        # Add raw stop token IDs (e.g., from stop_token_ids parameter)
        for tid in stop_token_ids or []:
            if ((tid,), None) not in common_stops:
                common_stops.append(((tid,), None))

        transitions = {"normal": list(common_stops)}

        if getattr(self.tokenizer, "has_thinking", False):
            try:
                ts = self.tokenizer.think_start_tokens
                te = self.tokenizer.think_end_tokens
                transitions["normal"].append((ts, "reasoning"))
                transitions["reasoning"] = [(te, "normal")]
                transitions["reasoning"].extend(common_stops)
            except (AttributeError, TypeError):
                pass

        return SequenceStateMachine(transitions, initial="normal")

    def fail_all_requests(self) -> list[str]:
        """Fail all active requests (running + waiting queue) for error recovery."""
        failed = list(self.running.keys())
        for req_id in failed:
            req = self.running.get(req_id)
            if req:
                req.set_finished(RequestStatus.FINISHED_ERROR, reason="error")
                self._total_prompt_tokens = max(
                    0, self._total_prompt_tokens - getattr(req, "num_prompt_tokens", 0)
                )
        self.running.clear()

        # Also fail waiting queue requests — they have collectors that need sentinels
        while self.waiting:
            req = self.waiting.pop()
            if req is not None:
                req.set_finished(RequestStatus.FINISHED_ERROR, reason="error")
                failed.append(req.request_id)

        # Remove the failed sequences from the BatchGenerator BEFORE dropping the
        # uid map — otherwise their uids stay live inside _batch_gen and the next
        # _batch_gen.next() keeps running forward passes for these dead sequences
        # (wasted GPU + KV held) until a later corruption reset clears it.
        if self._batch_gen is not None and self._uid_to_req:
            try:
                self._batch_gen.remove(list(self._uid_to_req.keys()))
            except Exception:
                logger.debug(
                    "batch_gen.remove failed in fail_all_requests", exc_info=True
                )
        # Also drop these uids from saved-prefix tracking so the set doesn't grow unbounded.
        if hasattr(self, "_saved_prefix_uids") and self._uid_to_req:
            for _u in self._uid_to_req:
                self._saved_prefix_uids.discard(_u)
        self._uid_to_req.clear()
        # Clean up per-request state dicts to prevent stale entries
        for rid in failed:
            self._detokenizers.pop(rid, None)
            self._thinking_processors.pop(rid, None)
            self._thinking_state.pop(rid, None)
            self._pending_prefill.pop(rid, None)
            self._spec_drafts.pop(rid, None)
            self._spec_draft_start_pos.pop(rid, None)
            self._spec_stats.pop(rid, None)
            self._spec_draft_cache_snapshots.pop(rid, None)
            self._last_token_time.pop(rid, None)
            self._itl_samples.pop(rid, None)
            self._chunked_prefill_fairness.pop(rid, None)
            self._chunked_prefill_enqueued_at.pop(rid, None)
            if self._attention_score_tracker is not None:
                self._attention_score_tracker.remove_request(rid)
        self._active_partial_prefills = 0
        return failed

    def remove_finished_request(self, request_id: str) -> None:
        req = self.requests.pop(request_id, None)
        self.finished_ids.discard(request_id)
        self._kv_prefix_hashes.pop(request_id, None)
        if req is not None:
            req.release_resources()

    def _try_init_spec_decoder(self) -> None:
        """Try to initialize speculative decoding by detecting spec heads in the model.

        Scans the model config for EAGLE-3 / MTP / Medusa / MLPSpeculator patterns.
        If detected and config.enable_spec_decode is True, creates a SpeculativeDecoder.
        The decoder is used for single-request speculative decoding in the serving path.
        """
        from .speculative_decoder import detect_spec_heads

        # Get model config
        model_config = {}
        config_obj = getattr(self.model, "config", None) or getattr(
            self.model, "args", None
        )
        if config_obj is not None:
            if hasattr(config_obj, "to_dict"):
                model_config = config_obj.to_dict()
            elif hasattr(config_obj, "__dict__"):
                model_config = {
                    k: v
                    for k, v in config_obj.__dict__.items()
                    if not k.startswith("_")
                }

        head_info = detect_spec_heads(model_config)
        self._spec_head_info = head_info

        if head_info.head_type == "none":
            logger.debug("No speculative decoding heads detected in model config")
            return

        if not self.config.enable_spec_decode:
            logger.info(
                f"Spec heads detected ({head_info.head_type}) but spec decode "
                f"disabled in config. Set enable_spec_decode=True to use."
            )
            return

        logger.info(
            f"Speculative decoding heads detected: type={head_info.head_type}, "
            f"num_heads={head_info.num_heads}, draft_length={head_info.draft_length}"
        )

        # SpeculativeDecoder requires a draft model. For self-speculative (MTP/EAGLE-3
        # with built-in heads), the draft is the same model. For external draft models,
        # load from draft_model config path.
        # Full integration with BatchGenerator's continuous batching is future work.
        # For Phase 4, we store the head info for use by the single-request path
        # in BatchedEngine.
        self._spec_decoder = (
            head_info  # Store head info; actual decoder created on demand
        )

    def get_spec_head_info(self) -> Any | None:
        """Return detected speculative decoding head info."""
        return self._spec_head_info

    def set_spec_decoder(self, decoder: Any) -> None:
        """Set the speculative decoder for the batch path.

        Called by BatchedEngine/EngineCore after loading a draft model
        and creating a SpeculativeDecoder. Once set, the scheduler's
        step loop will generate and verify draft tokens each step.

        Args:
            decoder: A SpeculativeDecoder instance (cross-model draft),
                     or an MTPStrategy/SpecStrategy instance.
        """
        self._spec_decoder = decoder
        self.config.enable_spec_decode = True
        logger.info(f"Scheduler spec decoder set: type={type(decoder).__name__}")

    def set_mtp_decoder(self, mtp_decoder: Any, _config: Any = None) -> None:
        """Set an MTP decoder for batch-path speculative decoding."""
        self._mtp_decoder = mtp_decoder
        self.config.enable_spec_decode = True
        logger.info(f"Scheduler MTP decoder set: type={type(mtp_decoder).__name__}")

    def enable_ngram_spec(
        self,
        min_n: int = 1,
        max_n: int = 5,
        k: int = 5,
        mode: str = "lps",
    ) -> None:
        """Enable or reconfigure N-gram speculative decoding at runtime."""
        config = NgramConfig(
            min_n=min_n,
            max_n=max_n,
            k=k,
            mode=mode,
            max_model_len=self.config.max_kv_size or 32768,
        )
        self._ngram_proposer = NgramProposer(config)
        self.config.ngram_spec_enabled = True
        self.config.ngram_spec_min_n = min_n
        self.config.ngram_spec_max_n = max_n
        self.config.ngram_spec_k = k
        self.config.ngram_spec_mode = mode
        logger.info(
            f"N-gram spec decode enabled: mode={mode}, min_n={min_n}, max_n={max_n}, k={k}"
        )

    # ── Speculative decoding batch-path methods ──

    def _try_spec_decode_draft(self, req: Request) -> None:
        """Generate K draft tokens for a request using the spec decoder.

        Supports three spec decode backends:
        1. SpeculativeDecoder (cross-model draft model)
        2. MTPDecoder (self-speculative via built-in prediction heads)
        3. NgramProposer (model-free N-gram pattern matching)

        All produce draft token IDs stored in ``self._spec_drafts[req.request_id]``
        and verified on the next scheduler step against the target model output.

        Args:
            req: The running request to generate draft tokens for.
        """
        rid = req.request_id
        if rid in self._pending_abort_ids:
            return
        if not req.output_token_ids:
            return

        # Path 1: Cross-model speculative decoder
        if self.config.enable_spec_decode and isinstance(
            self._spec_decoder, SpeculativeDecoder
        ):
            self._try_cross_model_draft(req)
            return

        # Path 2: MTP decoder (self-speculative, built-in prediction heads)
        if self.config.enable_spec_decode and self._mtp_decoder is not None:
            self._try_mtp_draft(req)
            return

        # Path 3: N-gram proposer (model-free)
        if self._ngram_proposer is not None:
            self._try_ngram_draft(req)

    def _try_cross_model_draft(self, req: Request) -> None:
        """Generate draft tokens using the cross-model spec decoder."""
        try:
            decoder = self._spec_decoder
            K = decoder.config.draft_length
            if K <= 0:
                return

            draft_result = self._generate_draft_tokens(req, decoder, K)

            if draft_result is not None and draft_result.token_ids:
                self._spec_drafts[req.request_id] = draft_result.token_ids
                # Bug 2 fix: record position where drafts start in output_token_ids
                self._spec_draft_start_pos[req.request_id] = len(
                    req.output_token_ids or []
                )
                rid = req.request_id
                if rid not in self._spec_stats:
                    self._spec_stats[rid] = {
                        "proposals": 0,
                        "accepted": 0,
                        "rejected": 0,
                        "mode": "cross_model",
                    }
                self._spec_stats[rid]["proposals"] += len(draft_result.token_ids)
                self._spec_total_proposals += len(draft_result.token_ids)
                logger.debug(
                    f"Spec decode draft for {rid}: "
                    f"{len(draft_result.token_ids)} tokens (cross-model)"
                )

        except Exception as e:
            logger.debug(f"Spec decode draft failed for {req.request_id}: {e}")

    def _try_mtp_draft(self, req: Request) -> None:
        """Generate draft token using the MTP decoder (self-speculative).

        MTP uses the model's own multi-token prediction heads to propose one
        draft token per step. The draft is verified on the next scheduler step
        when the target model produces the actual next token.

        Unlike cross-model spec decode, MTP requires no external draft model
        and no separate KV cache. The MTP head shares the backbone's hidden
        state. Draft generation runs on the MLX executor thread (GPU).

        The actual MTP forward pass (model.mtp_forward) is only used in the
        single-request path. For batch mode, we use a simpler heuristic:
        propose the most likely next token from the last hidden state. If the
        model supports n_confirmed, the verify step on the next batch step
        handles acceptance/rejection with zero-cost rollback.
        """
        try:
            mtp_decoder = self._mtp_decoder
            if mtp_decoder is None:
                return

            # MTP drafting in batch mode requires the model's KV cache state
            # from the last decode step. Without it, the forward pass has zero
            # prior context and produces meaningless drafts that waste spec
            # budget. Disable batch MTP until per-request KV cache passthrough
            # is implemented.
            return
            # The code below is unreachable but preserved for future reference
            # when per-request draft model KV cache is available.
            import mlx.core as mx

            last_tok = req.output_token_ids[-1]
            input_ids = mx.array([[last_tok]])

            # Use the MTP decoder's model for the draft proposal
            model = mtp_decoder.model
            inner = getattr(model, "language_model", model)

            # Check if model has MTP forward capability
            if hasattr(inner, "mtp_forward"):
                # Run the backbone forward to get hidden state, then MTP head
                out, hidden = model(
                    input_ids,
                    cache=None,  # No cache — just a forward for draft
                    return_hidden=True,
                )
                mx.synchronize()
                # MTP draft from hidden state
                primary = int(mx.argmax(out[0, -1, :]).item())
                draft = mtp_decoder._mtp_draft(hidden[:, -1:, :], primary)

                draft_ids = [draft]
            else:
                # Fallback: greedy argmax from a single forward pass
                out = model(input_ids)
                if hasattr(out, "logits"):
                    out = out.logits
                draft = int(mx.argmax(out[0, -1, :]).item())
                draft_ids = [draft]

            if draft_ids:
                rid = req.request_id
                self._spec_drafts[rid] = draft_ids
                # Bug 2 fix: record position where drafts start in output_token_ids
                self._spec_draft_start_pos[rid] = len(req.output_token_ids or [])
                if rid not in self._spec_stats:
                    self._spec_stats[rid] = {
                        "proposals": 0,
                        "accepted": 0,
                        "rejected": 0,
                        "mode": "mtp",
                    }
                self._spec_stats[rid]["proposals"] += len(draft_ids)
                self._spec_total_proposals += len(draft_ids)
                logger.debug(f"MTP spec draft for {rid}: {len(draft_ids)} tokens")
        except Exception as e:
            logger.debug(f"MTP spec draft failed for {req.request_id}: {e}")

    def _try_ngram_draft(self, req: Request) -> None:
        """Generate draft tokens using the N-gram proposer (model-free).

        The N-gram proposer requires no GPU work — it does O(1) or O(n) pattern
        matching on the token ID sequence. This is the primary spec decode path
        for the batch scheduler since it has zero GPU overhead and benefits all
        requests with repeated patterns (code, reasoning, etc.).
        """
        proposer = self._ngram_proposer
        if proposer is None:
            return
        try:
            # Build full token context: prompt tokens + generated tokens
            prompt_ids = req.prompt_token_ids or []
            all_ids = prompt_ids + list(req.output_token_ids)
            if len(all_ids) < proposer.config.min_n:
                return

            draft_ids = proposer.propose(all_ids)
            if draft_ids:
                rid = req.request_id
                self._spec_drafts[rid] = draft_ids
                # Bug 2 fix: record position where drafts start in output_token_ids
                self._spec_draft_start_pos[rid] = len(req.output_token_ids or [])
                if rid not in self._spec_stats:
                    self._spec_stats[rid] = {
                        "proposals": 0,
                        "accepted": 0,
                        "rejected": 0,
                        "mode": "ngram",
                    }
                self._spec_stats[rid]["proposals"] += len(draft_ids)
                self._spec_total_proposals += len(draft_ids)
                logger.debug(f"N-gram spec draft for {rid}: {len(draft_ids)} tokens")
        except Exception as e:
            logger.debug(f"N-gram spec draft failed for {req.request_id}: {e}")

    def _generate_draft_tokens(
        self, req: Request, decoder: SpeculativeDecoder, _K: int = 0
    ) -> DraftResult | None:
        """Generate draft tokens using the draft model.

        Args:
            req: The running request.
            decoder: The speculative decoder with draft model.
            _K: Unused (decoder knows its own draft_length).

        Returns:
            DraftResult with proposed tokens, or None on failure.
        """
        import mlx.core as mx
        from mlx_lm.models.cache import make_prompt_cache

        # Build input from last output token
        last_tok = req.output_token_ids[-1]
        input_ids = mx.array([[last_tok]])

        # Use the draft model's own cache (maintained across calls)
        # If no cache exists yet, create one
        if not hasattr(req, "_spec_draft_cache") or req._spec_draft_cache is None:
            req._spec_draft_cache = make_prompt_cache(decoder.draft)

        cache = req._spec_draft_cache
        # Snapshot cache BEFORE drafting so _verify_spec_drafts can rollback
        # to the correct pre-draft state on rejection. Without this, the
        # verify path snapshots the already-advanced cache and the restore
        # is a no-op, causing the cache to double-advance by (draft+accepted)
        # tokens instead of just (accepted) tokens.
        self._spec_draft_cache_snapshots[req.request_id] = (
            SpeculativeDecoder._snapshot_cache(cache)
        )
        return decoder.generate_draft(input_ids, cache)

    def _verify_spec_drafts(self, outputs: list) -> None:
        """Verify pending draft tokens against actual model output.

        Works with both cross-model and N-gram spec decode backends.
        For cross-model: also rolls back the draft model cache on rejection.
        For N-gram: only statistical tracking (no cache to manage).

        When YUNSHU_GPU_REJECTION=1, uses MLX batched comparison to find
        the first mismatch across all draft tokens for each request.
        """
        if not self._spec_drafts:
            return

        has_cross_model = isinstance(self._spec_decoder, SpeculativeDecoder)

        # Check if GPU rejection is enabled for batch comparison
        from .gpu_rejection import should_enable_gpu_rejection

        use_gpu = should_enable_gpu_rejection()

        verified_ids = []
        for output in outputs:
            rid = output.request_id
            draft_ids = self._spec_drafts.get(rid)
            if draft_ids is None:
                continue

            req = self.running.get(rid)
            if req is None:
                verified_ids.append(rid)
                continue

            actual_tokens = req.output_token_ids
            # Bug 2 fix: Use _spec_draft_start_pos to align the comparison
            # window precisely. Drafts were proposed starting at a specific
            # position in output_token_ids; compare from that position.
            # Fallback to [-n_draft:] if start_pos is not tracked.
            n_draft = len(draft_ids)
            start_pos = self._spec_draft_start_pos.get(rid)
            if start_pos is not None and start_pos < len(actual_tokens):
                n_available = min(n_draft, len(actual_tokens) - start_pos)
                recent_actual = actual_tokens[start_pos : start_pos + n_available]
            elif start_pos is not None:
                # No tokens generated at start_pos yet — nothing to compare
                recent_actual = []
            else:
                # Fallback: take last n_draft tokens. This is imprecise when
                # stream_interval > 1 (more tokens generated per step), but
                # still better than skipping verification entirely.
                recent_actual = actual_tokens[-n_draft:] if n_draft > 0 else []
            n_compare = min(len(draft_ids), len(recent_actual))

            # When no actual tokens are available at the draft start position,
            # skip verification entirely — counting all drafts as "rejected"
            # inflates rejection metrics and makes spec decode look worse than
            # it is.  The drafts will be re-verified on a subsequent step when
            # the target model has caught up to start_pos.
            if n_compare == 0:
                verified_ids.append(rid)
                continue

            # GPU-accelerated batch comparison: find first mismatch via MLX
            if use_gpu and n_compare > 0:
                import mlx.core as mx

                draft_arr = mx.array(draft_ids[:n_compare])
                actual_arr = mx.array(recent_actual[:n_compare])
                match_mask = draft_arr == actual_arr
                # cumsum trick: count consecutive matches from the start
                cum_mismatch = mx.cumsum((~match_mask).astype(mx.int32))
                accepted = int(mx.sum(cum_mismatch == 0).item())
            else:
                accepted = 0
                for i, draft_tok in enumerate(draft_ids):
                    if i < len(recent_actual) and recent_actual[i] == draft_tok:
                        accepted += 1
                    else:
                        break

            rejected = len(draft_ids) - accepted

            if rid not in self._spec_stats:
                self._spec_stats[rid] = {"proposals": 0, "accepted": 0, "rejected": 0}
            self._spec_stats[rid]["accepted"] += accepted
            self._spec_stats[rid]["rejected"] += rejected
            self._spec_total_accepted += accepted
            self._spec_total_rejected += rejected

            # Rollback draft model cache only for cross-model backend
            if (
                has_cross_model
                and accepted < len(draft_ids)
                and hasattr(req, "_spec_draft_cache")
                and req._spec_draft_cache is not None
            ):
                try:
                    # Use the pre-draft snapshot saved in _generate_draft_tokens.
                    # The previous code took a snapshot here of the already-advanced
                    # cache, making restore a no-op — the cache would then be
                    # advanced by (total_draft + accepted) tokens instead of just
                    # (accepted) tokens.
                    pre_draft_snapshot = self._spec_draft_cache_snapshots.pop(rid, None)
                    if pre_draft_snapshot is not None:
                        SpeculativeDecoder._restore_cache(
                            req._spec_draft_cache, pre_draft_snapshot
                        )
                    else:
                        # No pre-draft snapshot — cannot safely rollback.
                        # The cache is already at post-draft position. Replaying
                        # accepted tokens on top would double-advance. Skip the
                        # replay entirely to minimize corruption.
                        logger.warning(
                            "No pre-draft snapshot for %s — skipping cache replay "
                            "to avoid double-advance",
                            rid,
                        )
                        verified_ids.append(rid)
                        continue
                    import mlx.core as mx

                    for tok in draft_ids[:accepted]:
                        self._spec_decoder.draft(
                            mx.array([[tok]]),
                            cache=req._spec_draft_cache,
                        )
                except Exception as e:
                    logger.debug(f"Draft cache rollback failed for {rid}: {e}")

            logger.debug(
                f"Spec verify for {rid}: "
                f"{accepted}/{len(draft_ids)} accepted, "
                f"{rejected} rejected"
            )

            verified_ids.append(rid)

        for rid in verified_ids:
            self._spec_drafts.pop(rid, None)
            self._spec_draft_start_pos.pop(rid, None)
            self._spec_draft_cache_snapshots.pop(rid, None)

    def _cleanup_spec_state(self, req_id: str) -> None:
        """Clean up spec decode state for a finished/aborted request.

        Called from _process_responses() when a request finishes, and from
        _process_aborts() when a request is aborted.
        """
        # Count any pending drafts as rejected before clearing,
        # so _spec_total_proposals matches accepted + rejected.
        pending = self._spec_drafts.pop(req_id, None)
        if pending:
            n = len(pending)
            self._spec_total_rejected += n
            stats = self._spec_stats.get(req_id)
            if stats:
                stats["rejected"] = stats.get("rejected", 0) + n
        self._spec_draft_start_pos.pop(req_id, None)
        self._spec_stats.pop(req_id, None)
        self._spec_draft_cache_snapshots.pop(req_id, None)
        # Invalidate the per-request draft cache to prevent stale KV state
        # from corrupting predictions if the request is re-scheduled after
        # preemption.
        _req = self.running.get(req_id) or self._uid_to_req.get(req_id)
        if _req is not None and hasattr(_req, "_spec_draft_cache"):
            _req._spec_draft_cache = None

    # ── End speculative decoding batch-path methods ──

    # ── Batch-path SpecPrefill, Spec-Aware Scheduling, Batched Draft Collection ──

    def _apply_batch_spec_prefill(self, to_insert: list[Request]) -> list[Request]:
        """Apply SpecPrefill to waiting requests before insertion.

        For each request with a prompt exceeding the threshold, computes
        which tokens are skippable and attaches the selected indices to
        the request metadata. The actual sparse prefill happens during
        the BatchGenerator insert phase.

        Args:
            to_insert: List of requests to potentially apply SpecPrefill to.

        Returns:
            The same list with spec_prefill metadata attached where applicable.
        """
        if self._batch_spec_prefill is None:
            return to_insert

        for req in to_insert:
            if (
                req.prompt_token_ids
                and len(req.prompt_token_ids)
                >= self._batch_spec_prefill.config.threshold
            ):
                selected = self._batch_spec_prefill.compute_skippable_tokens(
                    req.prompt_token_ids,
                )
                if selected is not None:
                    req._spec_prefill_selected = selected
                    logger.debug(
                        f"Batch SpecPrefill: {len(selected)}/{len(req.prompt_token_ids)} "
                        f"tokens selected for {req.request_id}"
                    )

        return to_insert

    def spec_prefill_step(self, tokens: list[int]) -> list[int] | None:
        """Calculate attention-based token importance scores for a prompt.

        Public API for batch-path SpecPrefill. When enabled, uses the draft
        model's attention scores to identify which prompt tokens to skip,
        reducing prefill time for long prompts.

        Args:
            tokens: Prompt token IDs to score.

        Returns:
            List of selected (important) token indices, or None if
            SpecPrefill is disabled or scoring fails.
        """
        if self._batch_spec_prefill is None:
            return None
        return self._batch_spec_prefill.compute_skippable_tokens(tokens)

    def collect_batch_drafts(self) -> DraftCollection:
        """Collect draft tokens from all spec strategies for all running requests.

        This is the key integration point for batched draft verification.
        Instead of verifying drafts per-request, this collects drafts from
        all strategies (N-gram, cross-model, MTP, Medusa) for ALL running
        requests and returns them as a structured DraftCollection.

        The scheduler can then batch-verify all drafts in a single forward
        pass on the next step, significantly improving verification throughput.

        Returns:
            DraftCollection with all collected drafts keyed by request_id.
        """
        return self._draft_collector.collect_all_drafts(
            running=self.running,
            spec_decoder=self._spec_decoder
            if isinstance(self._spec_decoder, SpeculativeDecoder)
            else None,
            mtp_decoder=self._mtp_decoder,
            ngram_proposer=self._ngram_proposer,
            pending_abort_ids=self._pending_abort_ids,
            spec_drafts=self._spec_drafts,
        )

    def set_batch_spec_prefill_draft_model(self, draft_model: Any) -> None:
        """Set the draft model for batch-path SpecPrefill.

        Called by EngineCore/BatchedEngine after loading a draft model.
        Once set, long prompts in the batch path will use attention-based
        sparse prefill to reduce TTFT.
        """
        if self._batch_spec_prefill is not None:
            self._batch_spec_prefill.config.draft_model = draft_model
            logger.info("Batch SpecPrefill draft model set")

    def set_spec_aware_tbo(self, tbo_enabled: bool) -> None:
        """Update TBO status for spec-aware slot allocation.

        Called by EngineCore when TBO is enabled/disabled at runtime.
        """
        if self._spec_aware_scheduler is not None:
            self._spec_aware_scheduler.tbo_enabled = tbo_enabled

    # ── End Batch-path SpecPrefill, Spec-Aware Scheduling, Batched Draft Collection ──

    def get_batch_rope_deltas(self, uids: list[int]) -> list[float]:
        """Get per-request mRoPE deltas for batch decode.

        Returns deltas aligned to the UID order, defaulting to 0.0
        for text-only requests.
        """
        return self._rope_delta_mgr.get_batch_deltas(uids)

    def deep_reset(self) -> None:
        if self._batch_gen is not None:
            try:
                self._batch_gen.close()
            except Exception:
                logger.debug("failed", exc_info=True)
            self._batch_gen = None
        self.waiting.clear()
        self.running.clear()
        self.requests.clear()
        self.finished_ids.clear()
        self._uid_to_req.clear()
        self._detokenizers.clear()
        self._thinking_processors.clear()
        self._thinking_state.clear()
        self._pending_abort_ids.clear()
        self._uids_to_remove.clear()
        self._failed_insert_ids.clear()
        self._pending_prefill.clear()
        # Reset cumulative stats counters so get_stats() reflects fresh state
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._num_requests = 0
        self._step_counter = 0
        self._deferred_clear_at = None
        # Reset attention score tracker
        if self._attention_score_tracker is not None:
            self._attention_score_tracker = AttentionScoreTracker(
                max_blocks_per_request=self.config.attention_eviction_max_blocks,
            )
        # Reset chunked prefill production counters
        self._chunked_prefill_chunks_processed = 0
        self._chunked_prefill_fairness.clear()
        self._chunked_prefill_enqueued_at.clear()
        self._chunked_prefill_budget_used = 0
        self._chunked_prefill_failed_ids.clear()
        self._active_partial_prefills = 0
        self._spec_decoder = None
        self._spec_head_info = None
        self._mtp_decoder = None
        if self._ngram_proposer is not None:
            # Recreate ngram proposer (clears all learned patterns)
            self._ngram_proposer = NgramProposer(
                NgramConfig(
                    min_n=self.config.ngram_spec_min_n,
                    max_n=self.config.ngram_spec_max_n,
                    k=self.config.ngram_spec_k,
                    mode=self.config.ngram_spec_mode,
                    max_model_len=self.config.max_kv_size or 32768,
                )
            )
        self._spec_drafts.clear()
        self._spec_draft_start_pos.clear()
        self._spec_stats.clear()
        self._spec_draft_cache_snapshots.clear()
        self._spec_total_proposals = 0
        self._spec_total_accepted = 0
        self._spec_total_rejected = 0
        self._kv_prefix_hashes.clear()
        # Clear saved-prefix UID tracking: deep_reset drops the BatchGenerator, and the
        # fresh one restarts uid_count at 0, so stale uids here would collide with new
        # requests' uids → _save_one_prefix's `if uid in _saved_prefix_uids: return` would
        # silently skip prefix-caching for the first post-reset batch.
        if hasattr(self, "_saved_prefix_uids"):
            self._saved_prefix_uids.clear()
        self._last_token_time.clear()
        self._itl_samples.clear()
        self._rope_delta_mgr.clear()
        # clear encoder-decoder cache
        self._encoder_cache.clear()
        # Reset spec-aware scheduler and draft collector stats
        if self._spec_aware_scheduler is not None:
            self._spec_aware_scheduler = SpecAwareBatchScheduler(
                # re-create with the EFFECTIVE decode cap (min of
                # max_num_seqs and completion_batch_size), matching the init above.
                # deep_reset used the bare max_num_seqs, so after any reset (fail-recovery
                # / model reload) compute_spec_budget over-admitted up to max_num_seqs
                # while the BatchGenerator only decodes completion_batch_size — re-opening
                # the OOM/policy-defeat surface the effective cap closed.
                max_num_seqs=min(
                    self.config.max_num_seqs, self.config.completion_batch_size
                ),
                spec_overhead_per_request=self.config.spec_overhead_per_request,
            )
        if self._draft_collector is not None:
            self._draft_collector = BatchedDraftCollection()

    def get_thinking_store(self) -> ThinkingSegmentSubstore:
        """Return the ThinkingSegmentSubstore for external access.

        Used by EngineCore / gateway to query stats, clear conversations,
        or perform cross-request thinking KV lookups.
        """
        return self._thinking_store

    def get_stats(self) -> dict:
        stats = {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "total_requests": len(self.requests),
            "finished": len(self.finished_ids),
            "step_counter": self._step_counter,
            "total_prompt_tokens": self._total_prompt_tokens,
            "total_completion_tokens": self._total_completion_tokens,
            "num_requests_processed": self._num_requests,
            "total_preemptions": sum(r.num_preemptions for r in self.requests.values()),
            # Yunshu's own chunked/hybrid prefill subsystem is
            # currently INACTIVE — `should_chunk` is hardwired False (the old
            # manual chunking corrupted KV; now disabled), so
            # `_pending_prefill` is always empty and none of the chunk counters
            # ever advance. mlx-lm's native `prefill_step_size` does the real
            # prefill chunking. These fields are retained for API stability but
            # are flagged so the metrics surface no longer implies a live feature.
            "chunked_prefill_active": False,  # see note above — Yunshu chunking is off
            "hybrid_prefill_enabled": self.config.enable_hybrid_prefill,
            "hybrid_prefill_pending": len(self._pending_prefill),  # always 0 (inactive)
            "hybrid_chunk_size": self.config.hybrid_chunk_size,
            "chunked_prefill_chunks_processed": self._chunked_prefill_chunks_processed,  # always 0
            "chunked_prefill_budget": self.config.chunked_prefill_budget,
            "chunked_prefill_budget_used": self._chunked_prefill_budget_used,  # always 0
            "chunked_prefill_timeout_seconds": self.config.chunked_prefill_timeout_seconds,
            "chunked_prefill_abort_on_timeout": self.config.chunked_prefill_abort_on_timeout,
            "active_partial_prefills": self._active_partial_prefills,
            "max_num_partial_prefills": self.config.max_num_partial_prefills,
            "max_long_partial_prefills": self.config.max_long_partial_prefills,
            "long_prefill_token_threshold": self.config.long_prefill_token_threshold,
        }
        # Attention-score-based eviction (H2O) stats
        if self._attention_score_tracker is not None:
            stats["attention_eviction"] = self._attention_score_tracker.get_stats()
        stats["attention_eviction_enabled"] = self._attention_score_tracker is not None
        # Append thinking-segment substore stats
        try:
            stats["thinking_segment_store"] = self._thinking_store.get_stats()
        except Exception:
            logger.debug("thinking segment store stats unavailable", exc_info=True)
        # Append speculative decoding stats
        stats["spec_enabled"] = self.config.enable_spec_decode and isinstance(
            self._spec_decoder, SpeculativeDecoder
        )
        stats["mtp_spec_enabled"] = self._mtp_decoder is not None
        stats["ngram_spec_enabled"] = self._ngram_proposer is not None
        stats["spec_proposals"] = self._spec_total_proposals
        stats["spec_accepted"] = self._spec_total_accepted
        stats["spec_rejected"] = self._spec_total_rejected
        if self._spec_total_proposals > 0:
            stats["spec_acceptance_rate"] = round(
                self._spec_total_accepted / self._spec_total_proposals, 3
            )
        else:
            stats["spec_acceptance_rate"] = 0.0
        stats["spec_pending_drafts"] = len(self._spec_drafts)
        if self._ngram_proposer is not None:
            stats["ngram_spec"] = self._ngram_proposer.get_stats()
        if self._mtp_decoder is not None:
            try:
                s = self._mtp_decoder.stats
                stats["mtp_spec"] = {
                    "accepts": s.accepts,
                    "rejects": s.rejects,
                    "cooldowns": s.cooldowns,
                    "tokens_generated": s.tokens_generated,
                    "total_cycles": s.total_cycles,
                }
            except Exception:
                logger.debug("MTP stats unavailable", exc_info=True)
        # encoder-decoder cache stats
        stats["encoder_cache"] = self._encoder_cache.get_stats()
        # Batch RoPE deltas (mRoPE multimodal decode support)
        stats["batch_rope_deltas"] = len(self._last_batch_rope_deltas)
        # Metal kernel stats removed (kernels deleted — slower than mx.fast).
        # Batch-path SpecPrefill stats
        stats["batch_spec_prefill"] = {
            "enabled": self._batch_spec_prefill is not None,
        }
        if self._batch_spec_prefill is not None:
            stats["batch_spec_prefill"].update(self._batch_spec_prefill.get_stats())
        # Spec-aware batch scheduling stats
        if self._spec_aware_scheduler is not None:
            stats["spec_aware_scheduler"] = self._spec_aware_scheduler.get_stats()
        # Batched draft collection stats
        if self._draft_collector is not None:
            stats["draft_collector"] = self._draft_collector.get_stats()
        # Cache-locality reordering stats
        stats["cache_locality"] = {
            "tracked_prefixes": len(self._kv_prefix_hashes),
            "unique_groups": len(set(self._kv_prefix_hashes.values()))
            if self._kv_prefix_hashes
            else 0,
        }
        return stats


@dataclass
class SchedulerOutput:
    """Output from one scheduler step."""

    outputs: list[RequestOutput] = field(default_factory=list)
