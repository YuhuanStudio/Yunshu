from __future__ import annotations

"""Yunshu Scheduler — continuous batching via mlx-lm BatchGenerator.

Studied from oMLX's scheduler.py and engine_core.py, written from scratch:
- mlx-lm BatchGenerator as backend (same as oMLX)
- Request lifecycle: waiting → running → finished
- Per-request sampler + SequenceStateMachine (mlx-lm pattern)
- Per-request detokenizer (never pool — reset() leaks byte buffers)
- Deferred cache clearing (oMLX #435: prevent IOKit kernel panics)
- Thread-safe abort via pending set (oMLX pattern)
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

from .priority_queue import RequestPriorityQueue, make_waiting_queue
from .request import Request, RequestOutput, RequestStatus, SamplingParams
from .speculative_decoder import SpeculativeDecoder, DraftResult
from .ngram_proposer import NgramProposer, NgramConfig
from yunshu_kv.thinking_segment import ThinkingSegmentSubstore, ThinkingSegmentConfig

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
    threshold: int = 8192          # Minimum prompt length to trigger
    keep_rate: float = 0.20        # Fraction of tokens to keep
    chunk_size: int = 32           # Chunk size for token selection
    draft_model: Any = None        # Draft model for attention scoring


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

    def add(self, request_id: str, tokens: list[int], strategy: str = "unknown") -> None:
        """Add draft tokens for a request."""
        if tokens:
            self.drafts[request_id] = tokens
            self.strategy_counts[strategy] = self.strategy_counts.get(strategy, 0) + len(tokens)
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
            self.strategy_counts[strategy] = self.strategy_counts.get(strategy, 0) + count
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
        decode_slots = num_running

        # Reserve slots proportional to spec verification overhead.
        # Each running request with spec decode active consumes extra slot
        # budget for draft verification. The overhead is the fraction of a
        # full slot each verification costs.
        spec_slots = 0
        if spec_overhead > 0 and num_running > 0:
            # TBO overlap: draft generation overlaps with verification,
            # reducing effective overhead by ~50%.
            effective_overhead = spec_overhead
            if self.tbo_enabled:
                effective_overhead *= 0.5
                self._stats["tbo_overlap_steps"] += 1
            spec_slots = max(1, round(num_running * effective_overhead))

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
            if spec_decoder is not None and isinstance(spec_decoder, SpeculativeDecoder):
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
        self, req: Request, proposer: NgramProposer,
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
        self, req: Request, mtp_decoder: Any,
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
        self, req: Request, decoder: SpeculativeDecoder,
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

    Inspired by vLLM's H2O (Heavy-Hitter Oracle) eviction: instead of evicting
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
            scores[i] = scores.get(i, 0.0) + score
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
    """Request scheduling policy (oMLX pattern)."""
    FCFS = auto()       # First-Come-First-Served
    PRIORITY = auto()   # Priority-based (higher priority = scheduled first)


@dataclass
class SchedulerConfig:
    """Scheduler tuning parameters (maps to oMLX's SchedulerConfig)."""
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
    # External prefill (memory preflight, chunked progress, mid-prefill abort)
    use_external_prefill: bool = False
    prefill_chunk_size: int = 2048
    request_timeout_seconds: float = 300  # 5 min timeout for waiting requests
    max_waiting_requests: int = 1024     # Backpressure: reject new requests when queue is full
    memory_guard_enabled: bool = True    # Preflight memory check before scheduling
    memory_guard_soft_limit: float = 0.85  # Warn when active memory exceeds this fraction of total
    # Sarathi-style hybrid chunked prefill (interleave prefill chunks with decode)
    hybrid_chunk_size: int = 512        # Tokens per prefill chunk when interleaving
    enable_hybrid_prefill: bool = False  # Enable chunked prefill+decode interleaving
    # Request retraction (C14: SGLang pattern)
    enable_retraction: bool = True     # Temporarily evict decode for prefill under pressure
    retraction_memory_threshold: float = 0.90  # Retract when memory utilization exceeds this
    retraction_max_count: int = 4       # Max decode requests to retract per step
    # Speculative decoding (Phase 4)
    enable_spec_decode: bool = False     # Enable speculative decoding
    draft_model: str = ""                # Draft model name or path (empty = auto-detect from target)
    spec_draft_length: int = 5           # Number of draft tokens per step (K)
    # N-gram speculative decoding (model-free, always available)
    ngram_spec_enabled: bool = False     # Enable N-gram speculative decoding in batch path
    ngram_spec_min_n: int = 1            # Min ngram length
    ngram_spec_max_n: int = 5            # Max ngram length
    ngram_spec_k: int = 5                # Draft tokens per step
    ngram_spec_mode: str = "lps"         # Proposer mode: lps, hashpool, lcg
    # Batch-path SpecPrefill (sparse prefill for long prompts)
    batch_spec_prefill_enabled: bool = False  # Enable via YUNSHU_BATCH_SPEC_PREFILL=1
    batch_spec_prefill_threshold: int = 8192  # Min prompt length to trigger
    batch_spec_prefill_keep_rate: float = 0.20  # Fraction of tokens to keep
    # Spec-aware batch scheduling
    spec_overhead_per_request: float = 0.1  # Slot overhead per spec-active request
    # SCHED-3: Starvation prevention aging
    aging_weight: float = 0.1  # Age bonus per second in waiting queue (higher = less starvation)
    aging_enabled: bool = True  # Enable/disable aging in scheduling
    # H2O attention-score-based eviction (vLLM pattern)
    enable_attention_eviction: bool = False  # Enable attention score tracking for smarter KV eviction
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

    def __init__(self, base_sampler, logits_processors):
        self._base_sampler = base_sampler
        self._logits_processors = logits_processors
        self._tokens: list[int] = []

    def __call__(self, logits):
        # Apply logits processors: each takes (tokens, logits) -> logits
        for proc in self._logits_processors:
            logits = proc(self._tokens, logits)
        token = self._base_sampler(logits)
        # Track token for subsequent calls
        try:
            tid = token.item() if hasattr(token, 'item') else int(token)
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

    oMLX's key patterns:
    - Deep-copy tokenizer to avoid Rust RefCell races (#537)
    - Per-request detokenizer (never pool)
    - Deferred cache clearing with 8-step delay (#435)
    - Thread-safe abort via pending set
    - ServerMetrics integration for dashboard
    - PrefillProgressTracker for live prefill progress
    """

    _DEFERRED_CLEAR_DELAY = 8

    def __init__(self, model: Any, tokenizer: Any, config: SchedulerConfig | None = None):
        self.model = model
        self.tokenizer = copy.deepcopy(tokenizer)
        self.config = config or SchedulerConfig()
        self.model_id: str = config.model_name if config else ""

        # Request queues (vLLM pattern)
        self.waiting: RequestPriorityQueue[Request] = make_waiting_queue(self.config.policy)
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

        # Per-request detokenizers (oMLX: never pool)
        self._detokenizers: dict[str, Any] = {}

        # Sarathi-style chunked prefill: tracks partially-prefilled requests
        # Maps request_id → {'remaining_tokens': list[int], 'batch_uid': int | None}
        self._pending_prefill: dict[str, dict] = {}

        # Per-request thinking budget processors
        self._thinking_processors: dict[str, Any] = {}

        # KV prefix cache for batch-path insert_segments (C16)
        self._prefix_cache: Any = None

        # Cache-locality request reordering (SGLang/vLLM pattern)
        # Maps request_id → first KV block hash. Requests with the same hash
        # share a prefix and are scheduled consecutively for better cache locality.
        self._kv_prefix_hashes: dict[str, int] = {}

        # Deferred cache clearing (oMLX #435)
        self._step_counter: int = 0
        self._deferred_clear_at: int | None = None

        # Stats
        self._total_prompt_tokens: int = 0
        self._total_completion_tokens: int = 0
        self._num_requests: int = 0

        # External integrations (set by EngineCore)
        self._server_metrics: Any | None = None
        self._prefill_tracker: Any | None = None

        # External prefill (lazy init)
        self._external_prefiller: Any | None = None
        self._memory_monitor: Any | None = None

        # Thinking-segment KV substore for reasoning cache reuse (§3.6 / Δ-6)
        self._thinking_store = ThinkingSegmentSubstore(ThinkingSegmentConfig())

        # KV offload manager (async tier-to-tier block migration, §12.3)
        # Set by EngineCore after initialization.
        self._kv_offload_manager: Any | None = None

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
                    from .gpu_ngram import GPUNgramProposer, GPUNgramConfig
                    gpu_config = GPUNgramConfig(
                        min_n=self.config.ngram_spec_min_n,
                        max_n=self.config.ngram_spec_max_n,
                        k=self.config.ngram_spec_k,
                        max_model_len=self.config.max_kv_size or 32768,
                        gpu_fallback=True,
                    )
                    self._ngram_proposer = GPUNgramProposer(gpu_config)
                    logger.info("GPU-accelerated N-gram proposer enabled (YUNSHU_GPU_NGRAM=1)")
                except Exception:
                    logger.debug("GPU N-gram init failed, falling back to CPU", exc_info=True)
                    self._ngram_proposer = NgramProposer(NgramConfig(
                        min_n=self.config.ngram_spec_min_n,
                        max_n=self.config.ngram_spec_max_n,
                        k=self.config.ngram_spec_k,
                        mode=self.config.ngram_spec_mode,
                        max_model_len=self.config.max_kv_size or 32768,
                    ))
            else:
                self._ngram_proposer = NgramProposer(NgramConfig(
                    min_n=self.config.ngram_spec_min_n,
                    max_n=self.config.ngram_spec_max_n,
                    k=self.config.ngram_spec_k,
                    mode=self.config.ngram_spec_mode,
                    max_model_len=self.config.max_kv_size or 32768,
                ))

        # Speculative decoding — batch-path draft/verify state
        # Maps request_id → list[int] of draft token IDs from the spec decoder.
        # Drafts are generated after a decode step and verified against the
        # target model's output on the next step (vLLM "verify after" pattern).
        self._spec_drafts: dict[str, list[int]] = {}

        # Per-request spec decode statistics
        self._spec_stats: dict[str, dict[str, int]] = {}  # req_id → {proposals, accepted, rejected}

        # Aggregate spec decode counters for get_stats()
        self._spec_total_proposals: int = 0
        self._spec_total_accepted: int = 0
        self._spec_total_rejected: int = 0

        # Per-request thinking state tracking for segment store
        # Maps request_id → dict with:
        #   'in_thinking': bool — currently in reasoning state
        #   'thinking_start_idx': int | None — output_token_ids index where thinking began
        #   'was_in_thinking': bool — previous step's thinking state (for transition detection)
        self._thinking_state: dict[str, dict] = {}

        # mRoPE batch delta manager (oMLX pattern)
        from .mrope import BatchRopeDeltaManager
        self._rope_delta_mgr = BatchRopeDeltaManager()
        self._last_batch_rope_deltas: list[tuple[int, float]] = []

        # Encoder-decoder cache (§12.2: vLLM EncoderCacheManager pattern)
        from .encoder_cache import EncoderCacheManager
        self._encoder_cache = EncoderCacheManager()

        # Metal kernel manager for custom GPU kernels (paged attention, GEMV, KIVI)
        # Set by EngineCore when YUNSHU_METAL_KERNELS=1 is enabled.
        self._metal_kernel_manager: Any | None = None

        # Hybrid KV cache for Mamba/hybrid models (layer-type-aware routing)
        # Set by EngineCore when model has mixed attention + SSM layers.
        self._hybrid_kv: Any | None = None

        # ITL tracking (C2/ITL-1: inter-token latency per request)
        self._last_token_time: dict[str, float] = {}
        self._itl_samples: dict[str, list[float]] = {}

        # H2O attention-score-based eviction (vLLM pattern)
        self._attention_score_tracker: AttentionScoreTracker | None = None
        if self.config.enable_attention_eviction:
            self._attention_score_tracker = AttentionScoreTracker(
                max_blocks_per_request=self.config.attention_eviction_max_blocks,
            )
            logger.info("Attention-score-based eviction (H2O) enabled")

        # Chunked prefill production counters
        self._chunked_prefill_chunks_processed: int = 0
        self._chunked_prefill_fairness: dict[str, int] = {}  # req_id -> chunks served
        self._chunked_prefill_enqueued_at: dict[str, float] = {}  # req_id -> time.monotonic()
        self._chunked_prefill_timeout_seconds: float = 30.0  # Max time before forced completion

        # Batch-path SpecPrefill (attention-based sparse prefill for long prompts)
        import os as _os
        self._batch_spec_prefill: BatchPathSpecPrefill | None = None
        if self.config.batch_spec_prefill_enabled or _os.environ.get("YUNSHU_BATCH_SPEC_PREFILL", "").strip() in ("1", "true", "yes"):
            self._batch_spec_prefill = BatchPathSpecPrefill(BatchSpecPrefillConfig(
                enabled=True,
                threshold=self.config.batch_spec_prefill_threshold or int(
                    _os.environ.get("YUNSHU_BATCH_SPEC_PREFILL_THRESHOLD", "8192")
                ),
                keep_rate=self.config.batch_spec_prefill_keep_rate or float(
                    _os.environ.get("YUNSHU_BATCH_SPEC_PREFILL_KEEP_RATE", "0.20")
                ),
            ))
            logger.info(
                f"Batch SpecPrefill enabled: threshold={self.config.batch_spec_prefill_threshold}, "
                f"keep_rate={self.config.batch_spec_prefill_keep_rate}"
            )

        # Spec-aware batch scheduler (slot allocation with spec overhead)
        self._spec_aware_scheduler = SpecAwareBatchScheduler(
            max_num_seqs=self.config.max_num_seqs,
            spec_overhead_per_request=self.config.spec_overhead_per_request,
        )

        # Batched draft collection (collect drafts from all strategies for all running)
        self._draft_collector = BatchedDraftCollection()

        # Batch composer (vLLM/SGLang pattern: ScheduleBatch → ForwardBatch)
        from .forward_batch import BatchComposer
        self._batch_composer = BatchComposer(
            max_batch_size=self.config.max_num_seqs,
            max_prefill_slots=self.config.prefill_batch_size if hasattr(self.config, 'prefill_batch_size') else 8,
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

    def _init_batch_generator(self) -> None:
        """Create BatchGenerator on first use (lazy init)."""
        if self._batch_gen is not None:
            return

        from mlx_lm.generate import BatchGenerator, generation_stream
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=1.0)
        self._batch_gen = BatchGenerator(
            self.model,
            max_tokens=self.config.completion_batch_size,
            sampler=sampler,
            prefill_batch_size=self.config.prefill_batch_size,
            completion_batch_size=self.config.completion_batch_size,
            prefill_step_size=self.config.prefill_step_size,
            max_kv_size=self.config.max_kv_size,
            stream=generation_stream,
        )
        logger.info("BatchGenerator initialized")

    def shutdown(self) -> None:
        """Shutdown scheduler and release BatchGenerator resources."""
        self.deep_reset()
        if self._batch_gen is not None:
            if hasattr(self._batch_gen, 'close'):
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

    def set_kv_offload_manager(self, manager: Any) -> None:
        """Set KV offload manager for periodic tier migration (§12.3).

        Called by EngineCore after creating the KVOffloadManager.
        The scheduler calls manager.maybe_offload() periodically from step().
        """
        self._kv_offload_manager = manager

    def set_metal_kernel_manager(self, manager: Any) -> None:
        """Set Metal kernel manager for custom GPU kernel operations.

        Called by EngineCore when YUNSHU_METAL_KERNELS=1 is enabled.
        Provides Metal-accelerated paged attention decode, GEMV, and
        KIVI 2-bit KV cache compression to the batch path.
        """
        self._metal_kernel_manager = manager

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

        SGLang and vLLM sort running requests by shared KV block prefixes
        before each forward pass for the same reason.

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

        # Emit grouped first (shared prefix), then ungrouped
        result: list = []
        for group in prefix_groups.values():
            result.extend(group)
        result.extend(no_prefix)
        return result

    def _get_external_prefiller(self) -> Any:
        """Lazy-initialize the ExternalPrefiller."""
        if self._external_prefiller is None:
            from .external_prefill import ExternalPrefiller
            self._external_prefiller = ExternalPrefiller(
                model=self.model,
                tokenizer=self.tokenizer,
            )
        return self._external_prefiller

    def add_request(self, request: Request) -> None:
        """Add request to waiting queue (called from event loop thread)."""
        if len(self.waiting) >= self.config.max_waiting_requests:
            request.status = RequestStatus.FINISHED_ERROR
            request.finish_reason = "queue_full"
            logger.warning(
                f"Rejecting request {request.request_id}: waiting queue full "
                f"({len(self.waiting)}/{self.config.max_waiting_requests})"
            )
            return
        request.status = RequestStatus.WAITING
        self.requests[request.request_id] = request
        import time as _time
        request._submit_time = _time.monotonic()
        self.waiting.push(request, priority=request.sampling_params.priority)

    def abort_request(self, request_id: str) -> bool:
        """Thread-safe abort (deferred to next step, oMLX pattern)."""
        self._pending_abort_ids.add(request_id)
        return request_id in self.requests

    def has_requests(self) -> bool:
        return bool(self.waiting) or bool(self.running) or bool(self._pending_abort_ids)

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
                outputs.append(RequestOutput(
                    request_id=fail_id,
                    finished=True,
                    finish_reason="error",
                    error=f"Request {fail_id} failed to insert into batch generator",
                    prompt_tokens=getattr(fail_req, 'num_prompt_tokens', 0) if fail_req else 0,
                    completion_tokens=0,
                ))
            self._failed_insert_ids.clear()

        if self._batch_gen is None:
            return SchedulerOutput(outputs=outputs)

        # 3. Run one BatchGenerator step (prefill + first decode)
        # Note: outputs may already contain error outputs from failed inserts (step 2b).
        try:
            prompt_responses, gen_responses = self._batch_gen.next()

            # 3a. Retrieve batch RoPE deltas for decode (mRoPE multimodal support)
            # Provides per-request mRoPE deltas aligned to UID order. Currently
            # stored for future use in multimodal batch decode; text-only requests
            # return 0.0 deltas.
            if self.running:
                try:
                    _uids = [uid for uid, rid in self._uid_to_req.items() if rid in self.running]
                    if _uids:
                        _rope_deltas = self.get_batch_rope_deltas(_uids)
                        self._last_batch_rope_deltas = list(zip(_uids, _rope_deltas))
                except Exception:
                    logger.debug("batch rope deltas collection failed", exc_info=True)

            # 4. Process prompt responses (prefill completion)
            if prompt_responses:
                self._process_prefill_responses(prompt_responses)

            # 5. Process initial generation responses
            if gen_responses:
                outputs = self._process_responses(gen_responses)

            # 6. Run additional decode steps to get more tokens per step
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

            # 6a. Sarathi-style hybrid prefill: interleave remaining prefill
            # chunks with decode steps for better tail latency.
            # When enable_hybrid_prefill=True and there are pending partial
            # prefills, feed one chunk per iteration then decode, repeating
            # until all chunks are consumed or a limit is reached.
            if self.config.enable_hybrid_prefill and self._pending_prefill:
                outputs = self._hybrid_prefill_step(outputs)

            # 6b. Speculative decoding: verify drafts, then generate new ones
            # (vLLM "verify after" pattern — verify pending drafts against
            # target model output, then draft K tokens for next step)
            spec_decoder_active = (
                self.config.enable_spec_decode
                and isinstance(self._spec_decoder, SpeculativeDecoder)
            )
            mtp_active = (
                self.config.enable_spec_decode
                and self._mtp_decoder is not None
            )
            ngram_active = self._ngram_proposer is not None
            if spec_decoder_active or mtp_active or ngram_active:
                self._verify_spec_drafts(outputs)
                # Batched draft collection: collect drafts from all strategies
                # for all running requests in one pass, then merge into _spec_drafts.
                batch_drafts = self.collect_batch_drafts()
                for rid, tokens in batch_drafts.drafts.items():
                    if rid not in self._spec_drafts:
                        self._spec_drafts[rid] = tokens
                # Generate drafts for still-active requests (per-request fallback
                # for strategies not covered by batch collection, e.g. MTP/cross-model)
                for req_id in list(self.running.keys()):
                    req = self.running.get(req_id)
                    if req is not None and req.output_token_ids:
                        if req_id not in self._spec_drafts:
                            self._try_spec_decode_draft(req)
        except Exception as e:
            logger.error(f"BatchGenerator step error: {e}", exc_info=True)
            from .exceptions import is_cache_corruption_error
            if is_cache_corruption_error(e):
                logger.warning("Cache corruption detected — resetting BatchGenerator")
                self.deep_reset()
            return SchedulerOutput(outputs=[])

        # 7. Deferred cache clearing
        self._step_counter += 1
        self._maybe_clear_cache()

        # 7b. Periodic memory pressure eviction (C12)
        if self._step_counter % 64 == 0 and self._memory_monitor is not None:
            self._maybe_evict_kv_cache()

        # 7c. Periodic KV offload check (§12.3)
        # When KVOffloadManager is configured, periodically check if blocks
        # should be migrated from hot → warm → SSD. Uses sync mode since
        # step() runs on the MLX executor thread.
        if self._kv_offload_manager is not None and self._step_counter % 128 == 0:
            self._maybe_kv_offload()

        # 7d. Periodic encoder-decoder cache eviction (§12.2)
        # Evict expired encoder hidden-state entries to reclaim memory.
        if self._step_counter % 64 == 0:
            self._encoder_cache.evict_all_expired()

        # 8. Cleanup finished
        self._cleanup_finished()

        return SchedulerOutput(outputs=outputs)

    def _schedule_waiting(self) -> None:
        """Move waiting requests into BatchGenerator.

        Supports FCFS (default) and PRIORITY scheduling policies.
        PRIORITY scheduling sorts by request priority (higher = first).

        Request preemption (vLLM pattern): when max_num_seqs is reached
        and policy is PRIORITY, preempts the lowest-priority running
        request to make room for a higher-priority waiting request.
        Under FCFS, no preemption occurs (new requests wait).

        When enable_hybrid_prefill is True and there are active decode
        requests, inserts only hybrid_chunk_size tokens per step
        (Sarathi-style chunked prefill), interleaving prefill chunks
        with decode steps for better tail latency.

        When use_external_prefill is True, runs external prefill before
        BatchGenerator.insert() for memory preflight, progress tracking,
        and mid-prefill abort support.
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
                continue
            submit = getattr(req, '_submit_time', now)
            if timeout > 0 and (now - submit) > timeout:
                req.status = RequestStatus.FINISHED_TIMEOUT
                req.finish_reason = "timeout"
                self.finished_ids.add(req.request_id)
                # Track timed-out request so step() generates an error output
                # for EngineCore to finalize (otherwise resources leak).
                self._failed_insert_ids.append(req.request_id)
                logger.warning(f"Request {req.request_id} timed out after {now - submit:.0f}s in waiting queue")
                continue
            to_insert.append(req)

        # No need to sort — the heap maintains order (FCFS or PRIORITY)

        # SCHED-3: Apply aging to prevent starvation of low-priority requests.
        # Requests that have waited a long time get an age bonus that boosts
        # their effective priority, eventually overtaking newer high-priority requests.
        # The age bonus is: age_seconds * aging_weight, added to the request's
        # original priority for scheduling purposes.
        if self.config.aging_enabled and len(to_insert) > 1:
            aging_weight = self.config.aging_weight
            _aged_insert = []
            for _req in to_insert:
                _submit = getattr(_req, '_submit_time', now)
                _age = max(0.0, now - _submit)
                _effective_priority = _req.sampling_params.priority + _age * aging_weight
                _aged_insert.append((_effective_priority, _req))
            _aged_insert.sort(key=lambda x: -x[0])  # Higher effective priority first
            to_insert = [_req for _, _req in _aged_insert]

        # Cache-locality reordering: sort to_insert by KV prefix hash so
        # requests sharing the same system prompt / conversation prefix are
        # inserted into the BatchGenerator consecutively. This improves KV
        # cache block locality during prefill and decode (SGLang/vLLM pattern).
        to_insert = self._reorder_by_cache_locality(to_insert)

        # Respect max_num_seqs limit — with preemption under PRIORITY policy
        active_count = len(self.running)

        # Spec-aware slot allocation: when speculative decoding is active,
        # reserve slots for draft verification overhead.
        has_spec = (
            (self.config.enable_spec_decode and isinstance(self._spec_decoder, SpeculativeDecoder))
            or self._mtp_decoder is not None
            or self._ngram_proposer is not None
        )
        if has_spec and self._spec_aware_scheduler is not None:
            budget = self._spec_aware_scheduler.compute_spec_budget(active_count)
            available_slots = budget.available_for_new
        else:
            available_slots = max(0, self.config.max_num_seqs - active_count)

        # Batch-path SpecPrefill: compute skippable tokens for long prompts
        # before insertion, reducing prefill time.
        if self._batch_spec_prefill is not None and to_insert:
            to_insert = self._apply_batch_spec_prefill(to_insert)

        if len(to_insert) > available_slots and self.config.policy == SchedulingPolicy.PRIORITY:
            # vLLM preemption pattern: evict lowest-priority running requests
            # to make room for higher-priority waiting requests
            to_preempt = len(to_insert) - available_slots
            preempted = self._preempt_lowest_priority(to_preempt)
            if preempted > 0:
                available_slots += preempted
                logger.info(
                    f"Preempted {preempted} running requests for {len(to_insert)} waiting "
                    f"(priority policy)"
                )

        if len(to_insert) > available_slots:
            # C14: Memory-pressure retraction (SGLang pattern)
            # Under pressure, temporarily retract decode requests to make room for prefill
            if self.config.enable_retraction and self._memory_monitor is not None:
                try:
                    info = self._memory_monitor.get_memory_info()
                    if info.utilization_pct >= self.config.retraction_memory_threshold * 100:
                        retracted = self._retract_decode_requests(
                            min(self.config.retraction_max_count, len(to_insert) - available_slots)
                        )
                        if retracted > 0:
                            available_slots += retracted
                            logger.info(
                                f"Retracted {retracted} decode requests under memory pressure "
                                f"(util={info.utilization_pct:.1f}%)"
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
        if self.config.memory_guard_enabled and active_count > 0 and to_insert:
            try:
                import mlx.core as mx
                active_mem = mx.get_active_memory()
                from .utils.hardware import get_hardware_info
                hw = get_hardware_info()
                total_mem = hw.total_memory_bytes
                soft_limit = int(total_mem * self.config.memory_guard_soft_limit)
                if active_mem > soft_limit:
                    logger.debug(
                        f"Memory guard: deferring {len(to_insert)} requests "
                        f"({active_mem / 1024**3:.1f}GB active > {soft_limit / 1024**3:.1f}GB soft limit)"
                    )
                    for req in reversed(to_insert):
                        self.waiting.push_front(req, priority=req.sampling_params.priority)
                    to_insert = []
            except Exception:
                logger.debug("memory guard check failed in scheduling", exc_info=True)
                pass  # Memory guard is best-effort

        # Track batch composition via BatchComposer (vLLM/SGLang pattern)
        if to_insert:
            from .forward_batch import RequestSlot
            pending_slots = [
                RequestSlot(
                    request_id=req.request_id,
                    prompt_tokens=req.prompt_token_ids or [],
                    max_tokens=req.sampling_params.max_tokens if req.sampling_params else 512,
                    priority=req.sampling_params.priority if req.sampling_params else 0,
                    is_prefill=True,
                    num_prompt_tokens=len(req.prompt_token_ids or []),
                    arrival_time=getattr(req, '_submit_time', now),
                )
                for req in to_insert
            ]
            active_slots = [
                RequestSlot(
                    request_id=rid,
                    prompt_tokens=[],
                    is_prefill=False,
                    priority=r.sampling_params.priority if r.sampling_params else 0,
                )
                for rid, r in self.running.items()
            ]
            self._batch_composer.compose(pending_slots, active_slots)

        for req in to_insert:
            try:
                # ── External prefill path ──
                if self.config.use_external_prefill:
                    prefill_ok = self._run_external_prefill(req)
                    if not prefill_ok:
                        continue  # request was aborted or errored

                sp = req.sampling_params
                sampler = self._make_sampler(sp)
                sm = self._make_state_machine(sp.stop, sp.stop_token_ids)

                # ── Thinking-segment KV lookup before prefill (§3.6 / Δ-6) ──
                # Check for reusable thinking KV segments from prior turns in
                # the same conversation. If found, attach cached KV data so the
                # engine can potentially skip re-thinking identical steps.
                if req.prompt_token_ids:
                    try:
                        # Use request_id as conversation identifier; callers may
                        # set a stable conversation_id via request metadata.
                        conv_id = getattr(req, 'conversation_id', None) or req.request_id
                        conv_segments = self._thinking_store.get_conversation_segments(conv_id)
                        if conv_segments:
                            # Attach the most recently accessed segment for
                            # potential KV reuse (prefill optimisation).
                            best = max(conv_segments, key=lambda s: s.last_accessed)
                            req.prompt_cache = best.kv_data
                            req.cached_tokens = getattr(req, 'cached_tokens', 0) + best.num_tokens
                            logger.debug(
                                f"Thinking KV reuse: {conv_id} → segment "
                                f"{best.step_hash} ({best.num_tokens} tokens)"
                            )
                    except Exception as e:
                        logger.debug(f"Thinking KV lookup failed for {req.request_id}: {e}")

                # ── Chunked prefill: split long prompts across steps ──
                # Two modes:
                # 1. Sarathi-style (enable_hybrid_prefill=True): Always chunk
                #    when decode requests are running, use hybrid_chunk_size.
                # 2. Standard chunked (SCHED-2): When prompt exceeds
                #    prefill_chunk_size, chunk to avoid monopolising the batch.
                #    This interleaves prefill chunks with decode even without
                #    the full Sarathi mode, reducing head-of-line blocking.
                tokens_to_insert = req.prompt_token_ids
                effective_chunk_size = 0
                should_chunk = False

                if (
                    self.config.enable_hybrid_prefill
                    and self._has_active_requests()
                    and len(tokens_to_insert) > self.config.hybrid_chunk_size
                ):
                    # Sarathi-style: aggressive chunking for latency
                    effective_chunk_size = self.config.hybrid_chunk_size
                    should_chunk = True
                elif (
                    not self.config.enable_hybrid_prefill
                    and self.config.prefill_chunk_size > 0
                    and len(tokens_to_insert) > self.config.prefill_chunk_size
                ):
                    # SCHED-2: Standard chunked prefill — split long prompts
                    # that exceed prefill_chunk_size even without hybrid mode.
                    # This prevents a single long prefill from starving
                    # running decode requests for multiple consecutive steps.
                    effective_chunk_size = self.config.prefill_chunk_size
                    should_chunk = True

                if should_chunk and effective_chunk_size > 0:
                    chunk = tokens_to_insert[:effective_chunk_size]
                    remaining = tokens_to_insert[effective_chunk_size:]
                    # Store remaining tokens for subsequent steps
                    self._pending_prefill[req.request_id] = {
                        'remaining_tokens': remaining,
                        'batch_uid': None,
                        # Track chunking mode for _process_pending_prefill
                        'chunk_size': effective_chunk_size,
                        'total_prompt_len': len(tokens_to_insert),
                        'offset': effective_chunk_size,
                    }
                    # Chunked prefill production tracking: fairness + timeout
                    self._chunked_prefill_fairness[req.request_id] = 0
                    self._chunked_prefill_enqueued_at[req.request_id] = time.monotonic()
                    tokens_to_insert = chunk

                # C16: Try KV prefix cache hit for batch-path acceleration
                cached_kv = None
                remaining_tokens = tokens_to_insert
                if self._prefix_cache is not None and not self._pending_prefill.get(req.request_id):
                    try:
                        import mlx.core as mx
                        ids_arr = mx.array(tokens_to_insert)
                        cached_kv, _, matched = self._prefix_cache.get(ids_arr)
                        if cached_kv is not None and matched > 0:
                            remaining_tokens = tokens_to_insert[matched:]
                            if matched > 32:
                                logger.info(
                                    f"Batch prefix cache hit: {matched}/{len(tokens_to_insert)} tokens "
                                    f"for {req.request_id}"
                                )
                    except Exception:
                        logger.debug("prefix cache lookup failed in batch path", exc_info=True)

                # §12.2: Check encoder cache for encoder-decoder models.
                # If the request has a cached encoder hidden state (from a prior
                # request with the same encoder input), attach it so the decoder
                # can skip re-encoding.
                if hasattr(req, 'encoder_request_id'):
                    cached_encoder = self._encoder_cache.get(req.encoder_request_id)
                    if cached_encoder is not None:
                        req.cached_encoder_output = cached_encoder
                        logger.debug(
                            "Encoder cache hit for %s — reusing encoder output",
                            req.request_id,
                        )

                if cached_kv is not None and len(remaining_tokens) > 0:
                    # Use insert_segments with cached KV state
                    uids = self._batch_gen.insert_segments(
                        segments=[[remaining_tokens]],
                        max_tokens=[sp.max_tokens],
                        caches=[cached_kv],
                        all_tokens=[tokens_to_insert],
                        samplers=[sampler],
                        state_machines=[sm],
                    )
                else:
                    uids = self._batch_gen.insert(
                        prompts=[tokens_to_insert],
                        max_tokens=[sp.max_tokens],
                        samplers=[sampler],
                        state_machines=[sm],
                    )

                req.batch_uid = uids[0]
                req.status = RequestStatus.RUNNING
                req.prefill_start = time.monotonic()
                self.running[req.request_id] = req
                self._uid_to_req[uids[0]] = req.request_id

                # H2O: Register request for attention score tracking
                if self._attention_score_tracker is not None:
                    self._attention_score_tracker.register_request(req.request_id)

                # Register mRoPE delta for batch decode (oMLX pattern)
                if getattr(req, 'rope_deltas', 0.0) != 0.0:
                    self._rope_delta_mgr.register(uids[0], req.rope_deltas)

                # Create fresh detokenizer (oMLX: never pool)
                req.detokenizer = self._create_detokenizer()
                self._detokenizers[req.request_id] = req.detokenizer

                # Create thinking budget processor if configured
                sp = req.sampling_params
                if sp.thinking_budget is not None or sp.reasoning_effort is not None:
                    from .thinking_budget import ThinkingBudgetProcessor, ThinkingBudgetConfig, parse_thinking_budget
                    config = parse_thinking_budget({
                        'thinking_budget': sp.thinking_budget,
                        'reasoning_effort': sp.reasoning_effort,
                    })
                    if config is not None:
                        self._thinking_processors[req.request_id] = ThinkingBudgetProcessor(config)
                elif getattr(sp, 'enable_thinking', False) or getattr(req, 'enable_thinking', False):
                    # Auto-detected thinking mode with default budget
                    from .thinking_budget import (
                        ThinkingBudgetProcessor, ThinkingBudgetConfig,
                        detect_needs_think_prefix,
                    )
                    if detect_needs_think_prefix(req.prompt_token_ids or [], self.tokenizer):
                        self._thinking_processors[req.request_id] = ThinkingBudgetProcessor(
                            ThinkingBudgetConfig(max_thinking_tokens=8192)
                        )

                self._total_prompt_tokens += req.num_prompt_tokens
                self._num_requests += 1

                # Track prefill progress
                if self._prefill_tracker is not None:
                    self._prefill_tracker.update(
                        req.request_id, 0, req.num_prompt_tokens, self.model_id,
                    )

                # ── Speculative decoding head detection (Phase 4) ──
                # On first request, check if model has spec heads (EAGLE/MTP/Medusa)
                # and create a SpeculativeDecoder if detected.
                if self._spec_decoder is None and self._spec_head_info is None:
                    self._try_init_spec_decoder()

            except Exception as e:
                logger.error(f"Failed to insert request {req.request_id}: {e}", exc_info=True)
                req.status = RequestStatus.FINISHED_ERROR
                req.finish_reason = "error"
                # Signal completion so callers don't hang
                self._uid_to_req.pop(getattr(req, 'batch_uid', None), None)
                self.finished_ids.add(req.request_id)
                # Track failed insert so step() generates an error output for
                # EngineCore to finalize (otherwise resources leak).
                self._failed_insert_ids.append(req.request_id)

    def _run_external_prefill(self, req: Request) -> bool:
        """Run external prefill for a request.

        Returns True if prefill completed successfully, False if aborted/errored.
        Sets request status to PREFILLING during prefill, then transitions to
        RUNNING or FINISHED_ABORTED/FINISHED_ERROR on completion/failure.

        The external prefill provides:
        1. Memory preflight check (raises PrefillMemoryExceededError if OOM)
        2. Chunked progress tracking via PrefillProgressTracker
        3. Mid-prefill abort via pending_abort_ids
        """
        from .external_prefill import PrefillAbortedError

        prefiller = self._get_external_prefiller()
        req.status = RequestStatus.PREFILLING
        req.prefill_start = time.monotonic()

        # Progress callback for PrefillProgressTracker
        def _on_progress(completed: int, total: int) -> None:
            if self._prefill_tracker is not None:
                self._prefill_tracker.update(
                    req.request_id, completed, total, self.model_id,
                )

        try:
            result = prefiller.prefill_chunked(
                token_ids=req.prompt_token_ids,
                chunk_size=self.config.prefill_chunk_size,
                on_progress=_on_progress,
                request_id=req.request_id,
                pending_aborts=self._pending_abort_ids,
                memory_monitor=self._memory_monitor,
            )

            # Record prefill metrics
            req.cached_tokens = result.cached_tokens
            req.prefill_end = time.monotonic()
            logger.debug(
                f"External prefill completed for {req.request_id}: "
                f"{result.num_tokens} tokens in {result.duration_s:.3f}s "
                f"({result.cached_tokens} cached)"
            )
            return True

        except PrefillAbortedError:
            logger.info(f"External prefill aborted for {req.request_id}")
            req.status = RequestStatus.FINISHED_ABORTED
            req.finish_reason = "abort"
            self.finished_ids.add(req.request_id)
            return False

        except Exception as e:
            from .exceptions import PrefillMemoryExceededError
            if isinstance(e, PrefillMemoryExceededError):
                logger.warning(
                    f"Prefill memory exceeded for {req.request_id}: {e}"
                )
            else:
                logger.error(
                    f"External prefill failed for {req.request_id}: {e}",
                    exc_info=True,
                )
            req.status = RequestStatus.FINISHED_ERROR
            req.finish_reason = "error"
            self.finished_ids.add(req.request_id)
            return False

    def _preempt_lowest_priority(self, count: int) -> int:
        """Preempt the lowest-priority running requests (vLLM pattern).

        Under PRIORITY policy, finds the running requests with the lowest
        priority (ties broken by arrival_time, newest first) and preempts
        them. Preempted requests are placed back at the front of the
        waiting queue with their KV state freed.

        Args:
            count: Number of requests to preempt.

        Returns:
            Number of requests actually preempted.
        """
        preempted = 0
        for _ in range(count):
            if not self.running:
                break

            # Find lowest-priority running request (vLLM: max() with
            # (priority, arrival_time) key — lowest priority = highest
            # value when we sort descending by priority for scheduling)
            # In Yunshu: higher priority number = higher priority, so
            # we evict the minimum priority (least important).
            # Ties broken by latest arrival_time (newest first).
            victim_id = min(
                self.running.keys(),
                key=lambda rid: (
                    self.running[rid].priority,
                    -self.running[rid].arrival_time,
                ),
            )
            victim = self.running.pop(victim_id)
            self._preempt_request(victim)
            preempted += 1

        return preempted

    def _preempt_request(self, request: Request) -> None:
        """Preempt a running request and return it to the waiting queue.

        vLLM block-level preemption with partial recomputation (SCHED-1):
        1. Extract KV cache from BatchGenerator before removal
        2. Save the prompt prefix portion to the KV prefix cache
        3. Preserve cached prefix tokens (RadixTree maintains these)
        4. Only reset computed tokens beyond the cached prefix
        5. Set status to PREEMPTED
        6. Put back at front of waiting queue for re-scheduling

        When the request is re-scheduled, the prefix cache will be checked
        and only the uncached tail needs re-prefilling, significantly
        reducing re-prefill overhead compared to whole-request preemption.
        """
        uid = request.batch_uid

        # SCHED-1: Extract KV cache before removal so we can save the prefix.
        # BatchGenerator.remove(uids, return_prompt_caches=True) returns
        # {uid: (cache_list, tokens_list)} for generation-stage requests,
        # or {uid: (cache_list, tokens)} for prompt-stage requests.
        extracted_caches = {}
        if uid is not None and self._batch_gen is not None:
            try:
                extracted_caches = self._batch_gen.remove([uid], return_prompt_caches=True)
            except Exception as e:
                # Fallback: remove without cache extraction if API differs
                logger.debug(f"Failed to extract cache for preempted UID {uid}: {e}")
                try:
                    self._batch_gen.remove([uid])
                except Exception as e2:
                    logger.debug(f"Failed to remove preempted UID {uid}: {e2}")

        # SCHED-1: Save the prompt prefix portion of the KV cache to the
        # prefix cache.  Only the prompt tokens (not generated tokens) are
        # saved because the prefix cache keys off prompt token sequences.
        saved_prefix = 0
        if (
            self._prefix_cache is not None
            and uid in extracted_caches
            and request.prompt_token_ids
        ):
            try:
                import mlx.core as mx
                cache_and_tokens = extracted_caches[uid]
                if cache_and_tokens is not None:
                    cache_data = cache_and_tokens[0]
                    tokens_data = cache_and_tokens[1]
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
                logger.debug("Failed to save KV prefix during preemption", exc_info=True)

        self._uid_to_req.pop(uid, None)
        self._detokenizers.pop(request.request_id, None)
        self._thinking_processors.pop(request.request_id, None)
        self._thinking_state.pop(request.request_id, None)
        self._pending_prefill.pop(request.request_id, None)
        self._cleanup_spec_state(request.request_id)

        # H2O: Log attention-based eviction order for debugging.
        # When the tracker is enabled, the scheduler can use
        # get_eviction_order() to decide which blocks to evict first
        # instead of the default tail-eviction. This logging helps
        # operators verify that attention-aware eviction is working.
        if self._attention_score_tracker is not None:
            eviction_order = self._attention_score_tracker.get_eviction_order(request.request_id)
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
        if cached_prefix == 0 and self._prefix_cache is not None and request.prompt_token_ids:
            try:
                import mlx.core as mx
                ids_arr = mx.array(request.prompt_token_ids)
                _, _, matched = self._prefix_cache.get(ids_arr)
                if matched > 0:
                    cached_prefix = matched
            except Exception:
                logger.debug("failed", exc_info=True)

        request.status = RequestStatus.PREEMPTED
        # Preserve cached prefix tokens — only reset beyond cache boundary
        request.num_computed_tokens = min(cached_prefix, request.num_computed_tokens)
        request.batch_uid = None
        request.num_preemptions += 1

        self.waiting.push_front(request, priority=request.sampling_params.priority)

        prefix_info = f", cached_prefix={cached_prefix}" if cached_prefix > 0 else ""
        logger.info(
            f"Preempted request {request.request_id} "
            f"(preemptions={request.num_preemptions}, "
            f"output_tokens={request.num_output_tokens}{prefix_info})"
        )

    def _retract_decode_requests(self, count: int) -> int:
        """Temporarily retract decode requests under memory pressure (C14).

        SGLang pattern: swap out decode requests (which have lower per-token
        memory cost than prefill) to make room for new prefill requests.
        Retracted requests are placed at the front of the waiting queue and
        will be re-inserted with their existing KV state (via prefix cache).

        Args:
            count: Maximum number of requests to retract.

        Returns:
            Number of requests retracted.
        """
        retracted = 0
        # Sort running requests by output tokens (longest = most memory, evict first)
        candidates = sorted(
            [r for r in self.running.values() if r.batch_uid is not None],
            key=lambda r: r.num_output_tokens,
            reverse=True,
        )

        for victim in candidates:
            if retracted >= count:
                break
            self.running.pop(victim.request_id, None)
            self._preempt_request(victim)
            retracted += 1

        return retracted

    def _process_pending_prefill(self) -> None:
        """Process pending partial prefill chunks with production hardening.

        Handles two chunked-prefill modes:

        1. Sarathi-style hybrid (enable_hybrid_prefill=True): Process
           exactly ONE pending prefill chunk per step to interleave with
           decode steps. Prevents long prefills from starving generation.

        2. Standard chunked prefill (SCHED-2): When a prompt exceeded
           prefill_chunk_size, continue feeding chunks on each step.
           Also interleaves with decode (one chunk per step) to avoid
           head-of-line blocking.  Chunk size is tracked per-request
           in the pending_prefill state dict.

        Production hardening (Wave 108):
        - Fairness: tracks chunks served per request; when multiple
          pending prefills compete, the one with fewer chunks served
          is processed first.
        - Timeout: if a chunked prefill has been pending for more than
          _chunked_prefill_timeout_seconds (default 30s), all remaining
          tokens are fed in one shot to prevent indefinite starvation.
        - Cleanup: ensures _pending_prefill, _chunked_prefill_fairness,
          and _chunked_prefill_enqueued_at are cleaned for aborted or
          missing requests.

        When hybrid prefill is off and no standard chunking is active,
        all pending chunks are fed at once for maximum throughput.
        """
        if not self._pending_prefill:
            return

        _now = time.monotonic()
        completed_ids: list[str] = []
        chunks_fed = 0

        # ── Timeout handling ──
        # Check for chunked prefills that have been pending too long.
        # These get all their remaining tokens fed in one shot.
        timed_out_ids: list[str] = []
        for req_id, enqueued_at in list(self._chunked_prefill_enqueued_at.items()):
            if req_id not in self._pending_prefill:
                # Already completed or cleaned up elsewhere
                self._chunked_prefill_enqueued_at.pop(req_id, None)
                self._chunked_prefill_fairness.pop(req_id, None)
                continue
            if _now - enqueued_at > self._chunked_prefill_timeout_seconds:
                timed_out_ids.append(req_id)

        for req_id in timed_out_ids:
            state = self._pending_prefill.get(req_id)
            if state is None:
                continue
            remaining = state.get('remaining_tokens', [])
            if remaining and self._batch_gen is not None:
                req = self.running.get(req_id)
                if req is not None and req_id not in self._pending_abort_ids:
                    logger.warning(
                        f"Chunked prefill timeout for {req_id}: "
                        f"feeding {len(remaining)} remaining tokens in one shot "
                        f"(pending for {_now - self._chunked_prefill_enqueued_at.get(req_id, 0):.1f}s)"
                    )
                    try:
                        sp = req.sampling_params
                        sampler = self._make_sampler(sp)
                        sm = self._make_state_machine(sp.stop, sp.stop_token_ids)
                        uids = self._batch_gen.insert(
                            prompts=[remaining],
                            max_tokens=[sp.max_tokens],
                            samplers=[sampler],
                            state_machines=[sm],
                        )
                        self._uid_to_req[uids[0]] = req_id
                        self._chunked_prefill_chunks_processed += 1
                    except Exception as e:
                        logger.error(
                            f"Failed to force-feed timed-out chunked prefill for {req_id}: {e}",
                            exc_info=True,
                        )
            completed_ids.append(req_id)

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
            if req_id in completed_ids:
                continue

            remaining = state['remaining_tokens']
            if not remaining:
                completed_ids.append(req_id)
                continue

            # Check if the request was aborted
            if req_id in self._pending_abort_ids:
                completed_ids.append(req_id)
                continue

            req = self.running.get(req_id)
            if req is None:
                completed_ids.append(req_id)
                continue

            # Skip actual insertion if BatchGenerator is not ready
            if self._batch_gen is None:
                continue

            # Determine chunking mode and interleave limit.
            # SCHED-2: standard chunked prefill also limits to 1 chunk per
            # step when decode requests are running, matching Sarathi-style
            # interleaving semantics for fairness.
            is_sarathi = self.config.enable_hybrid_prefill
            is_standard_chunked = state.get('chunk_size') is not None

            if (
                (is_sarathi or is_standard_chunked)
                and self._has_active_requests()
                and chunks_fed >= 1
            ):
                # One chunk per step when decode is active → interleave
                # Remaining chunks will be fed on subsequent steps.
                break

            # Determine chunk size:
            # - SCHED-2 standard: use the per-request chunk_size from state
            # - Sarathi hybrid: use hybrid_chunk_size from config
            # - Legacy fallback: hybrid_chunk_size
            chunk_size = state.get('chunk_size') or self.config.hybrid_chunk_size

            # Use semantic chunk boundaries when optimizer is available
            if self._chunked_prefill_optimizer is not None and len(remaining) > chunk_size:
                try:
                    chunks = self._chunked_prefill_optimizer.compute_optimal_chunks(
                        remaining, chunk_size, max_chunks=1,
                    )
                    if chunks:
                        semantic_end = chunks[0].end_token
                        if semantic_end > 0 and semantic_end < len(remaining):
                            chunk_size = max(semantic_end, chunk_size // 2)
                except Exception:
                    logger.debug("semantic chunking fallback", exc_info=True)
            chunk = remaining[:chunk_size]
            state['remaining_tokens'] = remaining[chunk_size:]

            # SCHED-2: update offset tracker for progress reporting
            if 'offset' in state:
                state['offset'] += len(chunk)

            try:
                sp = req.sampling_params
                sampler = self._make_sampler(sp)
                sm = self._make_state_machine(sp.stop, sp.stop_token_ids)

                uids = self._batch_gen.insert(
                    prompts=[chunk],
                    max_tokens=[sp.max_tokens],
                    samplers=[sampler],
                    state_machines=[sm],
                )

                # Update tracking
                self._uid_to_req[uids[0]] = req_id
                chunks_fed += 1
                self._chunked_prefill_chunks_processed += 1
                self._chunked_prefill_fairness[req_id] = (
                    self._chunked_prefill_fairness.get(req_id, 0) + 1
                )

                total_prompt = state.get('total_prompt_len', 0)
                offset = state.get('offset', len(chunk))

                if not state['remaining_tokens']:
                    completed_ids.append(req_id)
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
                completed_ids.append(req_id)

        for rid in completed_ids:
            self._pending_prefill.pop(rid, None)
            self._chunked_prefill_fairness.pop(rid, None)
            self._chunked_prefill_enqueued_at.pop(rid, None)

        # ── Prometheus observation ──
        try:
            from yunshu_gateway.middleware.prometheus_exporter import get_prometheus_metrics
            pm = get_prometheus_metrics()
            pm.set_gauge("chunked_prefill_active_chunks", float(len(self._pending_prefill)))
            pm.set_gauge("chunked_prefill_total_chunks_processed", float(self._chunked_prefill_chunks_processed))
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
                    logger.warning("Cache corruption in hybrid prefill — resetting BatchGenerator")
                    self.deep_reset()
                break

        return outputs

    def _process_responses(self, responses: list) -> list[RequestOutput]:
        """Distribute GenerationBatch.Response to per-request outputs (oMLX pattern).

        Includes ITL tracking and progressive KV quantization.
        """
        _now = time.perf_counter()
        outputs = []
        for resp in responses:
            uid = resp.uid
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
                    total_tokens = len(req.prompt_token_ids or []) + len(req.output_token_ids)
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
            if hasattr(resp, 'logprobs') and resp.logprobs is not None:
                try:
                    logprobs = resp.logprobs
                except Exception:
                    logger.debug("logprobs extraction failed", exc_info=True)

            current_state = getattr(resp, 'current_state', 'normal') or 'normal'
            finish_reason = resp.finish_reason

            # ── Thinking-segment KV tracking (§3.6 / Δ-6) ──
            is_thinking = current_state == 'reasoning'
            ts = self._thinking_state.get(req_id)
            if ts is None:
                ts = {'in_thinking': False, 'thinking_start_idx': None, 'was_in_thinking': False, 'total_reasoning_tokens': 0}
                self._thinking_state[req_id] = ts

            # Detect thinking-start transition (normal → reasoning)
            if is_thinking and not ts['in_thinking']:
                ts['thinking_start_idx'] = len(req.output_token_ids)
                ts['in_thinking'] = True

            # Detect thinking-end transition (reasoning → normal):
            # store the completed thinking segment in the KV substore.
            if ts['in_thinking'] and not is_thinking and ts['thinking_start_idx'] is not None:
                thinking_end_idx = len(req.output_token_ids)
                thinking_tokens = list(req.output_token_ids[ts['thinking_start_idx']:thinking_end_idx])
                # Accumulate reasoning tokens from this completed segment
                ts['total_reasoning_tokens'] += len(thinking_tokens)
                context_tokens = list(req.prompt_token_ids) if req.prompt_token_ids else []
                # Extract KV data from response if available, else None
                kv_data = getattr(resp, 'prompt_cache', None)
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
                ts['in_thinking'] = False
                ts['thinking_start_idx'] = None

            ts['was_in_thinking'] = is_thinking

            # Thinking budget enforcement
            thinking_proc = self._thinking_processors.get(req_id)
            if thinking_proc is not None and not is_finished:
                budget_result = thinking_proc.process_token(current_state)
                if budget_result['force_stop']:
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
                    (ts.get('total_reasoning_tokens', 0) or 0)
                    + (len(req.output_token_ids) - ts['thinking_start_idx']
                       if ts and ts.get('thinking_start_idx') is not None and ts.get('in_thinking')
                       else 0)
                ),
                cached_tokens=getattr(req, 'cached_tokens', 0),
            )
            outputs.append(output)

            if finish_reason:
                # Finalize detokenizer (oMLX pattern)
                detok = self._detokenizers.pop(req_id, None)
                self._thinking_processors.pop(req_id, None)
                self._thinking_state.pop(req_id, None)
                # Cleanup speculative decoding state
                self._cleanup_spec_state(req_id)
                # Clean up chunked prefill state (request may finish while
                # still in the middle of chunked prefill, e.g. thinking budget overflow)
                self._pending_prefill.pop(req_id, None)
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

                # Update request state
                status_map = {
                    "stop": RequestStatus.FINISHED_STOPPED,
                    "length": RequestStatus.FINISHED_LENGTH,
                }
                req.status = status_map.get(finish_reason, RequestStatus.FINISHED_STOPPED)
                req.finish_reason = finish_reason
                self._uid_to_req.pop(uid, None)

                # Deferred cache clearing (oMLX #435)
                self._deferred_clear_at = (
                    self._step_counter + self._DEFERRED_CLEAR_DELAY
                )

                self._total_completion_tokens += req.num_output_tokens

                # Record in ServerMetrics (oMLX pattern)
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
                    if itl_list and hasattr(self._server_metrics, 'record_itl'):
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
                if hasattr(resp, 'end_of_prompt') and resp.end_of_prompt:
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

                    # Cache encoder outputs for encoder-decoder models (§12.2).
                    # If the response carries encoder hidden states (e.g. from a
                    # Whisper/T5-style encoder-decoder model), store them in the
                    # encoder cache for potential reuse in subsequent requests.
                    encoder_output = getattr(resp, 'encoder_outputs', None)
                    if encoder_output is not None:
                        self._encoder_cache.put(req_id, encoder_output)

    def _process_aborts(self) -> None:
        """Process deferred abort requests (oMLX pattern)."""
        if not self._pending_abort_ids:
            return

        abort_uids = []
        for req_id in list(self._pending_abort_ids):
            req = self.running.get(req_id)
            if req and req.batch_uid is not None:
                abort_uids.append(req.batch_uid)

        if abort_uids and self._batch_gen:
            self._batch_gen.remove(abort_uids)

        for req_id in list(self._pending_abort_ids):
            req = self.requests.get(req_id)
            if req:
                req.status = RequestStatus.FINISHED_ABORTED
                req.finish_reason = "abort"
                uid = getattr(req, 'batch_uid', None)
                self._uid_to_req.pop(uid, None)
                if uid is not None:
                    self._rope_delta_mgr.unregister(uid)
            self.running.pop(req_id, None)
            self._detokenizers.pop(req_id, None)
            self._thinking_processors.pop(req_id, None)
            self._thinking_state.pop(req_id, None)
            self._pending_prefill.pop(req_id, None)
            self._cleanup_spec_state(req_id)
            # H2O: cleanup attention score tracking
            if self._attention_score_tracker is not None:
                self._attention_score_tracker.remove_request(req_id)
            # Chunked prefill production tracking cleanup
            self._chunked_prefill_fairness.pop(req_id, None)
            self._chunked_prefill_enqueued_at.pop(req_id, None)

        self._pending_abort_ids.clear()

    def _maybe_clear_cache(self) -> None:
        """Deferred Metal cache cleanup (oMLX #435)."""
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
            if usage >= threshold and hasattr(self, '_prefix_cache') and self._prefix_cache is not None:
                mgr = getattr(self._prefix_cache, '_manager', None)
                if mgr is not None and hasattr(mgr, 'memory_pressure_evict'):
                    evicted = mgr.memory_pressure_evict(pressure_threshold=threshold)
                    if evicted > 0:
                        logger.info(
                            f"Memory pressure eviction: {evicted} KV blocks freed "
                            f"(usage {usage:.1%})"
                        )
        except Exception:
            logger.debug("failed", exc_info=True)

    def _maybe_kv_offload(self) -> None:
        """Periodic KV offload check via KVOffloadManager (§12.3).

        Called from step() every 128 steps. Uses the manager's sync
        offload path since step() runs on the MLX executor thread.
        The manager's policy decides whether offloading is needed and
        which blocks to migrate (hot → warm → SSD).
        """
        if self._kv_offload_manager is None:
            return
        try:
            # Use sync offload since we're on the executor thread
            mgr = self._kv_offload_manager
            if not mgr.config.enabled:
                return

            # Gather context for the policy
            context: dict[str, Any] = {
                "step_counter": self._step_counter,
                "memory_usage": 0.0,
                "free_blocks": 0,
            }
            try:
                import mlx.core as mx
                active_mem = mx.get_active_memory()
                from .utils.hardware import get_hardware_info
                hw = get_hardware_info()
                total_mem = hw.total_memory_bytes
                if total_mem > 0:
                    context["memory_usage"] = active_mem / total_mem
            except Exception:
                logger.debug("memory context gather in offload failed", exc_info=True)

            # Let the policy decide
            if not mgr._policy.should_offload(context):
                return

            hot_mgr = mgr._get_hot_manager()
            if hot_mgr is None:
                return

            max_blocks = context.get("max_blocks", mgr.config.lru_max_blocks_per_cycle)
            block_hashes = mgr._policy.select_blocks(hot_mgr, max_blocks, context)
            if not block_hashes:
                return

            # Execute sync offload
            result = mgr.offload_blocks_sync(block_hashes)
            if result.blocks_offloaded > 0:
                logger.info(
                    "KV offload: %d blocks migrated (%s → %s)",
                    result.blocks_offloaded,
                    result.source_tier.value,
                    result.dest_tier.value,
                )
        except Exception:
            logger.debug("KV offload check failed", exc_info=True)

    def _cleanup_finished(self) -> None:
        """Remove finished requests from running dict."""
        for req_id in list(self.running.keys()):
            req = self.running[req_id]
            if RequestStatus.is_finished(req.status):
                self.running.pop(req_id, None)
                self.finished_ids.add(req_id)
                self._kv_prefix_hashes.pop(req_id, None)
                # Clean up chunked prefill state for finished/aborted requests.
                # Without this, _pending_prefill leaks when a request finishes
                # while still in the middle of chunked prefill.
                self._pending_prefill.pop(req_id, None)
                # §12.2: Evict encoder cache entry for finished request.
                # The encoder output is no longer needed once the decoder
                # has completed generation.
                self._encoder_cache.evict(req_id)
                # H2O: cleanup attention score tracking for finished request
                if self._attention_score_tracker is not None:
                    self._attention_score_tracker.remove_request(req_id)
                # Chunked prefill production tracking cleanup
                self._chunked_prefill_fairness.pop(req_id, None)
                self._chunked_prefill_enqueued_at.pop(req_id, None)

    def _create_detokenizer(self):
        if self.tokenizer is None:
            return None
        detok = self.tokenizer.detokenizer
        detok.reset()
        return detok

    def _make_sampler(self, sp: SamplingParams):
        from mlx_lm.sample_utils import make_sampler, make_logits_processors
        # Seed handling: mlx-lm's make_sampler() does NOT accept a seed parameter.
        # Instead, we set the MLX global RNG seed before sampler creation so that
        # the categorical_sampling call inside the sampler closure uses the
        # specified seed. This ensures reproducibility per-request.
        if sp.seed is not None:
            import mlx.core as mx
            mx.random.seed(sp.seed)

        base_sampler = make_sampler(
            temp=sp.temperature,
            top_p=sp.top_p,
            top_k=sp.top_k,
            min_p=sp.min_p,
            xtc_probability=getattr(sp, 'xtc_probability', 0.0),
            xtc_threshold=getattr(sp, 'xtc_threshold', 0.0),
        )

        # Build logits processors for repetition/presence/frequency penalties + logit_bias
        logits_processors = make_logits_processors(
            repetition_penalty=sp.repetition_penalty if sp.repetition_penalty != 1.0 else None,
            presence_penalty=sp.presence_penalty if sp.presence_penalty != 0.0 else None,
            frequency_penalty=sp.frequency_penalty if sp.frequency_penalty != 0.0 else None,
            logit_bias=getattr(sp, 'logit_bias', None),
        )

        # SAMP-2: Append user-provided custom logits processors
        custom_procs = getattr(sp, 'logits_processors', None)
        if custom_procs:
            if logits_processors is None:
                logits_processors = []
            logits_processors.extend(custom_procs)

        if logits_processors:
            # Wrap sampler to apply logits processors before sampling.
            # Logits processors take (tokens, logits) and return modified logits.
            # We store generated tokens per-request via _generation_tokens.
            sampler = _LogitsProcessorSampler(base_sampler, logits_processors)
        else:
            sampler = base_sampler

        # JSON schema constrained generation
        json_schema = getattr(sp, 'json_schema', None)
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

    def _make_state_machine(self, stop: list[str] | None = None, stop_token_ids: list[int] | None = None):
        from mlx_lm.generate import SequenceStateMachine

        eos_ids = list(self.tokenizer.eos_token_ids) if hasattr(self.tokenizer, 'eos_token_ids') else []
        common_stops = [((t,), None) for t in eos_ids]
        for w in (stop or []):
            t = tuple(self.tokenizer.encode(w, add_special_tokens=False))
            common_stops.append((t, None))
        # Add raw stop token IDs (e.g., from stop_token_ids parameter)
        for tid in (stop_token_ids or []):
            if ((tid,), None) not in common_stops:
                common_stops.append(((tid,), None))

        transitions = {"normal": list(common_stops)}

        if getattr(self.tokenizer, 'has_thinking', False):
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
        """Fail all active requests (error recovery)."""
        failed = list(self.running.keys())
        for req_id in failed:
            req = self.running.get(req_id)
            if req:
                req.status = RequestStatus.FINISHED_ERROR
                req.finish_reason = "error"
        self.running.clear()
        self._uid_to_req.clear()
        return failed

    def remove_finished_request(self, request_id: str) -> None:
        self.requests.pop(request_id, None)
        self.finished_ids.discard(request_id)
        self._kv_prefix_hashes.pop(request_id, None)

    def _try_init_spec_decoder(self) -> None:
        """Try to initialize speculative decoding by detecting spec heads in the model.

        Scans the model config for EAGLE-3 / MTP / Medusa / MLPSpeculator patterns.
        If detected and config.enable_spec_decode is True, creates a SpeculativeDecoder.
        The decoder is used for single-request speculative decoding in the serving path.
        """
        from .speculative_decoder import detect_spec_heads

        # Get model config
        model_config = {}
        config_obj = getattr(self.model, 'config', None) or getattr(self.model, 'args', None)
        if config_obj is not None:
            if hasattr(config_obj, 'to_dict'):
                model_config = config_obj.to_dict()
            elif hasattr(config_obj, '__dict__'):
                model_config = {k: v for k, v in config_obj.__dict__.items()
                                if not k.startswith('_')}

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
        self._spec_decoder = head_info  # Store head info; actual decoder created on demand

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
        logger.info(
            f"Scheduler spec decoder set: type={type(decoder).__name__}"
        )

    def set_mtp_decoder(self, mtp_decoder: Any, _config: Any = None) -> None:
        """Set an MTP decoder for batch-path speculative decoding."""
        self._mtp_decoder = mtp_decoder
        self.config.enable_spec_decode = True
        logger.info(
            f"Scheduler MTP decoder set: type={type(mtp_decoder).__name__}"
        )

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
        logger.info(f"N-gram spec decode enabled: mode={mode}, min_n={min_n}, max_n={max_n}, k={k}")

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
        if (
            self.config.enable_spec_decode
            and isinstance(self._spec_decoder, SpeculativeDecoder)
        ):
            self._try_cross_model_draft(req)
            return

        # Path 2: MTP decoder (self-speculative, built-in prediction heads)
        if (
            self.config.enable_spec_decode
            and self._mtp_decoder is not None
        ):
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
                rid = req.request_id
                if rid not in self._spec_stats:
                    self._spec_stats[rid] = {"proposals": 0, "accepted": 0, "rejected": 0, "mode": "cross_model"}
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

            # MTP proposes a single draft token per step from the last
            # output token's hidden state. We need the model to run
            # mtp_forward(last_hidden, last_token). However, in batch
            # mode we don't have the hidden state readily available.
            # Instead, we run a lightweight forward to get logits and
            # take the greedy prediction as our draft.
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
                if hasattr(out, 'logits'):
                    out = out.logits
                draft = int(mx.argmax(out[0, -1, :]).item())
                draft_ids = [draft]

            if draft_ids:
                rid = req.request_id
                self._spec_drafts[rid] = draft_ids
                if rid not in self._spec_stats:
                    self._spec_stats[rid] = {"proposals": 0, "accepted": 0, "rejected": 0, "mode": "mtp"}
                self._spec_stats[rid]["proposals"] += len(draft_ids)
                self._spec_total_proposals += len(draft_ids)
                logger.debug(
                    f"MTP spec draft for {rid}: "
                    f"{len(draft_ids)} tokens"
                )
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
                if rid not in self._spec_stats:
                    self._spec_stats[rid] = {"proposals": 0, "accepted": 0, "rejected": 0, "mode": "ngram"}
                self._spec_stats[rid]["proposals"] += len(draft_ids)
                self._spec_total_proposals += len(draft_ids)
                logger.debug(
                    f"N-gram spec draft for {rid}: "
                    f"{len(draft_ids)} tokens"
                )
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
        if not hasattr(req, '_spec_draft_cache') or req._spec_draft_cache is None:
            req._spec_draft_cache = make_prompt_cache(decoder.draft)

        cache = req._spec_draft_cache
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
            n_compare = min(len(draft_ids), len(actual_tokens))
            recent_actual = actual_tokens[-n_compare:]

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
                and hasattr(req, '_spec_draft_cache')
                and req._spec_draft_cache is not None
            ):
                try:
                    snapshot = SpeculativeDecoder._snapshot_cache(req._spec_draft_cache)
                    SpeculativeDecoder._restore_cache(req._spec_draft_cache, snapshot)
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

    def _cleanup_spec_state(self, req_id: str) -> None:
        """Clean up spec decode state for a finished/aborted request.

        Called from _process_responses() when a request finishes, and from
        _process_aborts() when a request is aborted.
        """
        self._spec_drafts.pop(req_id, None)
        self._spec_stats.pop(req_id, None)

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
            if req.prompt_token_ids and len(req.prompt_token_ids) >= self._batch_spec_prefill.config.threshold:
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
            spec_decoder=self._spec_decoder if isinstance(self._spec_decoder, SpeculativeDecoder) else None,
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
        # Reset attention score tracker
        if self._attention_score_tracker is not None:
            self._attention_score_tracker = AttentionScoreTracker(
                max_blocks_per_request=self.config.attention_eviction_max_blocks,
            )
        # Reset chunked prefill production counters
        self._chunked_prefill_chunks_processed = 0
        self._chunked_prefill_fairness.clear()
        self._chunked_prefill_enqueued_at.clear()
        self._spec_decoder = None
        self._spec_head_info = None
        self._mtp_decoder = None
        if self._ngram_proposer is not None:
            # Recreate ngram proposer (clears all learned patterns)
            self._ngram_proposer = NgramProposer(NgramConfig(
                min_n=self.config.ngram_spec_min_n,
                max_n=self.config.ngram_spec_max_n,
                k=self.config.ngram_spec_k,
                mode=self.config.ngram_spec_mode,
                max_model_len=self.config.max_kv_size or 32768,
            ))
        self._spec_drafts.clear()
        self._spec_stats.clear()
        self._spec_total_proposals = 0
        self._spec_total_accepted = 0
        self._spec_total_rejected = 0
        self._rope_delta_mgr.clear()
        # §12.2: clear encoder-decoder cache
        self._encoder_cache.clear()
        # Reset spec-aware scheduler and draft collector stats
        if self._spec_aware_scheduler is not None:
            self._spec_aware_scheduler = SpecAwareBatchScheduler(
                max_num_seqs=self.config.max_num_seqs,
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
            "total_preemptions": sum(
                r.num_preemptions for r in self.requests.values()
            ),
            "hybrid_prefill_enabled": self.config.enable_hybrid_prefill,
            "hybrid_prefill_pending": len(self._pending_prefill),
            "hybrid_chunk_size": self.config.hybrid_chunk_size,
            "chunked_prefill_chunks_processed": self._chunked_prefill_chunks_processed,
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
        stats["spec_enabled"] = (
            self.config.enable_spec_decode
            and isinstance(self._spec_decoder, SpeculativeDecoder)
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
        # §12.2: encoder-decoder cache stats
        stats["encoder_cache"] = self._encoder_cache.get_stats()
        # Batch RoPE deltas (mRoPE multimodal decode support)
        stats["batch_rope_deltas"] = len(self._last_batch_rope_deltas)
        # Metal kernel manager stats (when enabled via YUNSHU_METAL_KERNELS=1)
        stats["metal_kernels"] = {
            "available": self._metal_kernel_manager is not None,
        }
        if self._metal_kernel_manager is not None:
            try:
                from .metal_kernels import get_compilation_status
                stats["metal_kernels"].update(get_compilation_status())
            except Exception:
                logger.debug("Metal kernel stats unavailable", exc_info=True)
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
            "unique_groups": len(set(self._kv_prefix_hashes.values())) if self._kv_prefix_hashes else 0,
        }
        return stats


@dataclass
class SchedulerOutput:
    """Output from one scheduler step (oMLX pattern)."""
    outputs: list[RequestOutput] = field(default_factory=list)
