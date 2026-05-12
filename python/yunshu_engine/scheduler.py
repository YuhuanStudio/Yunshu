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
from __future__ import annotations

import copy
import gc
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from .request import Request, RequestOutput, RequestStatus, SamplingParams
from yunshu_kv.thinking_segment import ThinkingSegmentSubstore, ThinkingSegmentConfig

logger = logging.getLogger(__name__)


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
    # Speculative decoding (Phase 4)
    enable_spec_decode: bool = False     # Enable speculative decoding
    draft_model: str = ""                # Draft model name or path (empty = auto-detect from target)
    spec_draft_length: int = 5           # Number of draft tokens per step (K)


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
            pass
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
        self.waiting: deque[Request] = deque()
        self.running: dict[str, Request] = {}
        self.requests: dict[str, Request] = {}
        self.finished_ids: set[str] = set()

        # Thread-safe abort (CPython GIL guarantees set.add atomicity)
        self._pending_abort_ids: set[str] = set()

        # UIDs to remove from BatchGenerator (thinking budget overflow, etc.)
        self._uids_to_remove: list[int] = []

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

        # Speculative decoding (Phase 4: EAGLE-3 single-request path)
        self._spec_decoder: Any | None = None
        self._spec_head_info: Any | None = None  # SpecHeadInfo from detect_spec_heads()

        # Per-request thinking state tracking for segment store
        # Maps request_id → dict with:
        #   'in_thinking': bool — currently in reasoning state
        #   'thinking_start_idx': int | None — output_token_ids index where thinking began
        #   'was_in_thinking': bool — previous step's thinking state (for transition detection)
        self._thinking_state: dict[str, dict] = {}

        # mRoPE batch delta manager (oMLX pattern)
        from .mrope import BatchRopeDeltaManager
        self._rope_delta_mgr = BatchRopeDeltaManager()

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
        self.waiting.append(request)

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

        if self._batch_gen is None:
            return SchedulerOutput(outputs=[])

        # 3. Run one BatchGenerator step (prefill + first decode)
        outputs = []
        try:
            prompt_responses, gen_responses = self._batch_gen.next()

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
        except Exception as e:
            logger.error(f"BatchGenerator step error: {e}")
            from .exceptions import is_cache_corruption_error
            if is_cache_corruption_error(e):
                logger.warning("Cache corruption detected — resetting BatchGenerator")
                self.deep_reset()
            return SchedulerOutput(outputs=[])

        # 7. Deferred cache clearing
        self._step_counter += 1
        self._maybe_clear_cache()

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
            req = self.waiting.popleft()
            if req.request_id in self._pending_abort_ids:
                self._pending_abort_ids.discard(req.request_id)
                continue
            submit = getattr(req, '_submit_time', now)
            if timeout > 0 and (now - submit) > timeout:
                req.status = RequestStatus.FINISHED_TIMEOUT
                req.finish_reason = "timeout"
                logger.warning(f"Request {req.request_id} timed out after {now - submit:.0f}s in waiting queue")
                continue
            to_insert.append(req)

        # Sort by priority if using PRIORITY policy (oMLX pattern)
        if self.config.policy == SchedulingPolicy.PRIORITY:
            to_insert.sort(
                key=lambda r: getattr(r.sampling_params, 'priority', 0),
                reverse=True,
            )

        # Respect max_num_seqs limit — with preemption under PRIORITY policy
        active_count = len(self.running)
        available_slots = max(0, self.config.max_num_seqs - active_count)

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
            overflow = to_insert[available_slots:]
            to_insert = to_insert[:available_slots]
            # Put overflow back at front of waiting queue
            for req in reversed(overflow):
                self.waiting.appendleft(req)

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
                        self.waiting.appendleft(req)
                    to_insert = []
            except Exception:
                pass  # Memory guard is best-effort

        for req in to_insert:
            try:
                # ── External prefill path ──
                if self.config.use_external_prefill:
                    prefill_ok = self._run_external_prefill(req)
                    if not prefill_ok:
                        continue  # request was aborted or errored

                sp = req.sampling_params
                sampler = self._make_sampler(sp)
                sm = self._make_state_machine(sp.stop)

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

                # ── Sarathi-style hybrid chunked prefill ──
                # When active generation is running, insert only a chunk of
                # tokens to interleave prefill with decode for better latency.
                tokens_to_insert = req.prompt_token_ids
                if (
                    self.config.enable_hybrid_prefill
                    and self._has_active_requests()
                    and len(tokens_to_insert) > self.config.hybrid_chunk_size
                ):
                    chunk = tokens_to_insert[:self.config.hybrid_chunk_size]
                    remaining = tokens_to_insert[self.config.hybrid_chunk_size:]
                    # Store remaining tokens for subsequent steps
                    self._pending_prefill[req.request_id] = {
                        'remaining_tokens': remaining,
                        'batch_uid': None,
                    }
                    tokens_to_insert = chunk

                # C16: Try KV prefix cache hit for batch-path acceleration
                cached_kv = None
                remaining_tokens = tokens_to_insert
                if self._prefix_cache is not None and not self._pending_prefill.get(req.request_id):
                    try:
                        import mlx.core as mx
                        ids_arr = mx.array(tokens_to_insert)
                        cached_kv, _rem, matched = self._prefix_cache.get(ids_arr)
                        if cached_kv is not None and matched > 0:
                            remaining_tokens = tokens_to_insert[matched:]
                            if matched > 32:
                                logger.info(
                                    f"Batch prefix cache hit: {matched}/{len(tokens_to_insert)} tokens "
                                    f"for {req.request_id}"
                                )
                    except Exception:
                        logger.debug("prefix cache lookup failed in batch path", exc_info=True)

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
                elif getattr(sp, 'enable_thinking', False):
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
                logger.error(f"Failed to insert request {req.request_id}: {e}")
                req.status = RequestStatus.FINISHED_ERROR
                req.finish_reason = "error"
                # Signal completion so callers don't hang
                self._uid_to_req.pop(getattr(req, 'batch_uid', None), None)
                self.finished_ids.add(req.request_id)

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
                    f"External prefill failed for {req.request_id}: {e}"
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

        Following vLLM's _preempt_request pattern:
        1. Free KV resources (remove from BatchGenerator)
        2. Reset computed tokens
        3. Set status to PREEMPTED
        4. Increment preemption counter
        5. Put back at front of waiting queue for re-scheduling

        The request will be re-inserted into BatchGenerator on the next
        scheduler step, effectively re-prefilling from scratch.
        """
        uid = request.batch_uid
        if uid is not None and self._batch_gen is not None:
            try:
                self._batch_gen.remove([uid])
            except Exception as e:
                logger.debug(f"Failed to remove preempted UID {uid}: {e}")

        self._uid_to_req.pop(uid, None)
        self._detokenizers.pop(request.request_id, None)
        self._thinking_processors.pop(request.request_id, None)
        self._thinking_state.pop(request.request_id, None)
        self._pending_prefill.pop(request.request_id, None)

        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        request.batch_uid = None
        request.num_preemptions += 1

        self.waiting.appendleft(request)

        logger.info(
            f"Preempted request {request.request_id} "
            f"(preemptions={request.num_preemptions}, "
            f"output_tokens={request.num_output_tokens})"
        )

    def _process_pending_prefill(self) -> None:
        """Process one chunk from each pending partial prefill (Sarathi pattern).

        On each scheduler step, when there are pending prefill requests
        (requests whose prompt was too long and got chunked), we feed
        one hybrid_chunk_size chunk into the BatchGenerator per step.

        This interleaves prefill chunks with decode steps so that
        existing generation requests don't stall while a long prompt
        is being processed.
        """
        if not self._pending_prefill:
            return

        completed_ids = []
        for req_id, state in list(self._pending_prefill.items()):
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

            # Feed one chunk
            chunk_size = self.config.hybrid_chunk_size
            chunk = remaining[:chunk_size]
            state['remaining_tokens'] = remaining[chunk_size:]

            try:
                sp = req.sampling_params
                sampler = self._make_sampler(sp)
                sm = self._make_state_machine(sp.stop)

                uids = self._batch_gen.insert(
                    prompts=[chunk],
                    max_tokens=[sp.max_tokens],
                    samplers=[sampler],
                    state_machines=[sm],
                )

                # Update tracking
                self._uid_to_req[uids[0]] = req_id

                if not state['remaining_tokens']:
                    completed_ids.append(req_id)
                    logger.debug(
                        f"Chunked prefill complete for {req_id}: "
                        f"final chunk {len(chunk)} tokens"
                    )
                else:
                    logger.debug(
                        f"Chunked prefill step for {req_id}: "
                        f"{len(chunk)} tokens, "
                        f"{len(state['remaining_tokens'])} remaining"
                    )
            except Exception as e:
                logger.error(
                    f"Failed to process chunked prefill for {req_id}: {e}"
                )
                completed_ids.append(req_id)

        for rid in completed_ids:
            self._pending_prefill.pop(rid, None)

    def _process_responses(self, responses: list) -> list[RequestOutput]:
        """Distribute GenerationBatch.Response to per-request outputs (oMLX pattern)."""
        outputs = []
        for resp in responses:
            uid = resp.uid
            req_id = self._uid_to_req.get(uid)
            if req_id is None:
                continue

            req = self.running.get(req_id)
            if req is None:
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
                ts = {'in_thinking': False, 'thinking_start_idx': None, 'was_in_thinking': False}
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
            )
            outputs.append(output)

            if finish_reason:
                # Finalize detokenizer (oMLX pattern)
                detok = self._detokenizers.pop(req_id, None)
                self._thinking_processors.pop(req_id, None)
                self._thinking_state.pop(req_id, None)
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
                pass

    def _cleanup_finished(self) -> None:
        """Remove finished requests from running dict."""
        for req_id in list(self.running.keys()):
            req = self.running[req_id]
            if RequestStatus.is_finished(req.status):
                self.running.pop(req_id, None)
                self.finished_ids.add(req_id)

    def _create_detokenizer(self):
        if self.tokenizer is None:
            return None
        detok = self.tokenizer.detokenizer
        detok.reset()
        return detok

    def _make_sampler(self, sp: SamplingParams):
        from mlx_lm.sample_utils import make_sampler, make_logits_processors
        # Seed: if specified, set MLX global RNG seed for reproducibility
        if sp.seed is not None:
            import mlx.core as mx
            mx.random.seed(sp.seed)

        base_sampler = make_sampler(
            temp=sp.temperature,
            top_p=sp.top_p,
            top_k=sp.top_k,
            min_p=sp.min_p,
        )

        # Build logits processors for repetition/presence/frequency penalties + logit_bias
        logits_processors = make_logits_processors(
            repetition_penalty=sp.repetition_penalty if sp.repetition_penalty != 1.0 else None,
            presence_penalty=sp.presence_penalty if sp.presence_penalty != 0.0 else None,
            frequency_penalty=sp.frequency_penalty if sp.frequency_penalty != 0.0 else None,
            logit_bias=getattr(sp, 'logit_bias', None),
        )

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

    def _make_state_machine(self, stop: list[str] | None = None):
        from mlx_lm.generate import SequenceStateMachine

        eos_ids = list(self.tokenizer.eos_token_ids) if hasattr(self.tokenizer, 'eos_token_ids') else []
        common_stops = [((t,), None) for t in eos_ids]
        for w in (stop or []):
            t = tuple(self.tokenizer.encode(w, add_special_tokens=False))
            common_stops.append((t, None))

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

    def _try_init_spec_decoder(self) -> None:
        """Try to initialize speculative decoding by detecting spec heads in the model.

        Scans the model config for EAGLE-3 / MTP / Medusa / MLPSpeculator patterns.
        If detected and config.enable_spec_decode is True, creates a SpeculativeDecoder.
        The decoder is used for single-request speculative decoding in the serving path.
        """
        from .speculative_decoder import detect_spec_heads, SpecHeadInfo

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
                pass
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
        self._pending_prefill.clear()
        self._spec_decoder = None
        self._spec_head_info = None
        self._rope_delta_mgr.clear()

    def shutdown(self) -> None:
        self.deep_reset()

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
        }
        # Append thinking-segment substore stats
        try:
            stats["thinking_segment_store"] = self._thinking_store.get_stats()
        except Exception:
            logger.debug("thinking segment store stats unavailable", exc_info=True)
        return stats


@dataclass
class SchedulerOutput:
    """Output from one scheduler step (oMLX pattern)."""
    outputs: list[RequestOutput] = field(default_factory=list)
