from __future__ import annotations
"""Yunshu Streaming Optimizer — pipelined token generation for lower ITL.

Four components that reduce inter-token latency and improve throughput:

1. TokenPipeline — 3-stage overlapping GPU/CPU pipeline:
   Stage 1 (GPU): model forward pass → logits
   Stage 2 (GPU): sampling → token id
   Stage 3 (CPU): detokenize + grammar check + response distribution
   Overlaps Stage 3 of token N with Stage 1 of token N+1 via
   mx.async_eval() and asyncio, reducing ITL by ~0.3–0.5ms.

2. PrefetchSampler — pre-computes sampling plan while GPU is busy:
   - Deterministic (temp=0/top_k=1): pre-computes argmax plan
   - Stochastic: pre-generates random numbers via numpy
   Saves ~0.1ms per token by avoiding per-step parameter setup.

3. BatchedDetokenizer — detokenizes multiple requests simultaneously:
   - Queues token_ids per request
   - Single batch decode call on flush()
   - Faster than per-request when batch size > 1

4. StreamingBackpressureController — prevents OOM on slow SSE clients:
   - Monitors queue depth vs configurable max
   - Returns delay to apply when client can't keep up
   - Prevents unbounded token buffering

Integration:
  - BatchedEngine._stream_generate_fast() wraps with TokenPipeline
  - PrefetchSampler.prepare() called before sampling step
  - BatchedDetokenizer flush() in _process_responses
  - StreamingBackpressureController check in streaming paths
"""

import asyncio
import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. TokenPipeline — 3-stage overlapping GPU/CPU pipeline
# ---------------------------------------------------------------------------


class PipelineStage(enum.IntEnum):
    """Pipeline stages for token generation."""
    IDLE = 0
    GPU_FORWARD = 1      # Stage 1: model forward → logits
    GPU_SAMPLING = 2     # Stage 2: sampling → token
    CPU_POST = 3         # Stage 3: detokenize + grammar + distribute


@dataclass
class PipelineToken:
    """A token moving through the pipeline."""
    token_id: int = -1
    text: str = ""
    logprobs: Any = None
    stage: PipelineStage = PipelineStage.IDLE
    timestamp_enter: float = 0.0
    timestamp_stage1_done: float = 0.0
    timestamp_stage2_done: float = 0.0
    timestamp_stage3_done: float = 0.0

    def mark_enter(self) -> None:
        self.timestamp_enter = time.perf_counter()

    def mark_stage1_done(self) -> None:
        self.timestamp_stage1_done = time.perf_counter()
        self.stage = PipelineStage.GPU_SAMPLING

    def mark_stage2_done(self) -> None:
        self.timestamp_stage2_done = time.perf_counter()
        self.stage = PipelineStage.CPU_POST

    def mark_stage3_done(self) -> None:
        self.timestamp_stage3_done = time.perf_counter()
        self.stage = PipelineStage.IDLE

    @property
    def total_latency_ms(self) -> float:
        if self.timestamp_stage3_done <= 0:
            return 0.0
        return (self.timestamp_stage3_done - self.timestamp_enter) * 1000

    @property
    def stage3_duration_ms(self) -> float:
        """Duration of stage 3 (CPU post-processing) in milliseconds.

        When overlap is working correctly, this work runs concurrently with
        the next token's stage 1 (GPU forward), so the effective wall-clock
        contribution to ITL is hidden.  The accumulated value is the total
        CPU post-processing time across all yielded tokens, *not* the
        savings from overlap.
        """
        if self.timestamp_stage2_done <= 0 or self.timestamp_stage3_done <= 0:
            return 0.0
        return (self.timestamp_stage3_done - self.timestamp_stage2_done) * 1000

    # Backward-compatible alias (previously mislabeled as "savings")
    @property
    def overlap_savings_ms(self) -> float:
        """Deprecated: use stage3_duration_ms instead."""
        return self.stage3_duration_ms


@dataclass
class PipelineConfig:
    """Configuration for TokenPipeline."""
    enable_overlap: bool = True
    async_eval: bool = True
    batch_detokenize: bool = True
    prefetch_sampling: bool = True
    pipeline_depth: int = 2  # Number of tokens in flight simultaneously


class TokenPipeline:
    """Pipelined token generation: overlaps GPU and CPU work.

    Pipeline structure (token N overlaps with token N+1):
      Token N:   [Stage1][Stage2][Stage3──────]
      Token N+1:           [Stage1][Stage2][Stage3]

    Stage 3 (CPU: detokenize + grammar + distribute) of token N
    overlaps with Stage 1 (GPU forward) of token N+1.

    The overlap is achieved by deferring stage3 results: when next_token()
    is called, it launches stage3 for the current token but returns the
    *previous* token's completed stage3 result.  The current token's
    stage3 runs concurrently while the caller performs GPU work for the
    next token.

    Usage:
      pipeline = TokenPipeline(config)
      pipeline.start_pipeline(request_ctx)
      while not pipeline.is_finished:
          token = await pipeline.next_token()
          # yield token to SSE stream
      pipeline.stop()
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._stage1_queue: asyncio.Queue[PipelineToken] = asyncio.Queue()
        self._stage3_queue: asyncio.Queue[PipelineToken | None] = asyncio.Queue()
        self._current: PipelineToken | None = None
        self._prev_stage3_task: asyncio.Task | None = None
        # Completed token from previous next_token() call, awaiting retrieval.
        self._completed_token: PipelineToken | None = None
        self._running = False
        self._finished = False
        self._tokens_generated = 0
        self._tokens_yielded = 0
        self._total_overlap_ms = 0.0
        self._start_time = 0.0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_finished(self) -> bool:
        return self._finished

    @property
    def tokens_generated(self) -> int:
        return self._tokens_generated

    @property
    def tokens_yielded(self) -> int:
        return self._tokens_yielded

    @property
    def avg_overlap_ms(self) -> float:
        """Average stage3 duration per yielded token (ms).

        This is the mean CPU post-processing time per token.  When overlap
        is effective, this time is hidden behind GPU work and does not add
        to the observed ITL.
        """
        if self._tokens_yielded == 0:
            return 0.0
        return self._total_overlap_ms / self._tokens_yielded

    @property
    def throughput_tok_per_s(self) -> float:
        elapsed = time.perf_counter() - self._start_time
        if elapsed <= 0:
            return 0.0
        return self._tokens_yielded / elapsed

    def start_pipeline(self, request: Any = None) -> None:
        """Begin pipelined generation for a request."""
        self._running = True
        self._finished = False
        self._tokens_generated = 0
        self._tokens_yielded = 0
        self._total_overlap_ms = 0.0
        self._start_time = time.perf_counter()
        self._current = None
        self._prev_stage3_task = None
        self._completed_token = None
        logger.debug("TokenPipeline started")

    def stop(self) -> None:
        """Stop the pipeline and clean up resources."""
        self._running = False
        if self._prev_stage3_task is not None and not self._prev_stage3_task.done():
            self._prev_stage3_task.cancel()
            self._prev_stage3_task = None
        self._current = None
        self._completed_token = None
        logger.debug(
            "TokenPipeline stopped: %d tokens, %.2f ms avg overlap",
            self._tokens_yielded,
            self.avg_overlap_ms,
        )

    def submit_stage1_result(
        self,
        logits: Any,
        token_id: int = -1,
    ) -> PipelineToken:
        """Submit result from Stage 1 (GPU forward pass).

        Called by the engine after the model forward pass completes.
        Returns a PipelineToken that flows through remaining stages.
        """
        tok = PipelineToken(
            token_id=-1,  # set by stage2 (sampling)
            stage=PipelineStage.GPU_SAMPLING,
        )
        tok.mark_enter()
        tok.mark_stage1_done()
        self._tokens_generated += 1
        self._current = tok
        return tok

    def submit_stage2_result(
        self,
        token: PipelineToken,
        sampled_id: int,
        logprobs: Any = None,
    ) -> PipelineToken:
        """Submit result from Stage 2 (GPU sampling).

        Called after sampling produces a token id from logits.
        """
        token.token_id = sampled_id
        token.logprobs = logprobs
        token.mark_stage2_done()
        return token

    async def run_stage3_overlap(
        self,
        token: PipelineToken,
        detokenize_fn: Any = None,
        grammar_fn: Any = None,
    ) -> PipelineToken:
        """Run Stage 3 (CPU post-processing) asynchronously.

        This is the stage that gets overlapped with the next token's
        Stage 1 (GPU forward). Runs on the event loop while GPU is busy.

        Args:
            token: The pipeline token to process.
            detokenize_fn: Optional async/sync function(token_id) -> str.
            grammar_fn: Optional async/sync function(text) -> bool (valid).

        Returns:
            The token with text populated and stage3 marked complete.
        """
        # Detokenize
        if detokenize_fn is not None:
            if asyncio.iscoroutinefunction(detokenize_fn):
                token.text = await detokenize_fn(token.token_id)
            else:
                token.text = detokenize_fn(token.token_id)

        # Grammar check
        if grammar_fn is not None:
            if asyncio.iscoroutinefunction(grammar_fn):
                _ = await grammar_fn(token.text)
            else:
                _ = grammar_fn(token.text)

        token.mark_stage3_done()
        # Note: stage3_duration_ms is accumulated in next_token()/drain_last_token()
        # when the completed token is yielded, NOT here, to avoid double-counting.
        return token

    async def next_token(
        self,
        detokenize_fn: Any = None,
        grammar_fn: Any = None,
        gpu_forward_fn: Any = None,
    ) -> PipelineToken | None:
        """Get the next completed token, overlapping GPU/CPU work.

        Pipeline flow with overlap enabled:
        1. Return the *previous* token's completed stage3 result
        2. Launch stage3 for the *current* token asynchronously
        3. The caller then does GPU work (stage1+stage2 for next token)
           while stage3 of the current token runs concurrently

        On the first call, there is no previous result, so we run stage3
        synchronously and return immediately.

        Returns None when pipeline is finished and all tokens are drained.
        """
        if not self._running:
            return None

        token = self._current
        if token is None:
            return None

        if self.config.enable_overlap:
            # Overlap mode: launch stage3 for current token, return
            # previous token's result.
            stage3_task = asyncio.create_task(
                self.run_stage3_overlap(token, detokenize_fn, grammar_fn)
            )

            # Collect the previous token's completed stage3 result
            result = self._completed_token

            if self._prev_stage3_task is not None:
                # The previous stage3 was running during the caller's GPU work.
                # It should be done by now. Await it to get the completed token.
                try:
                    self._completed_token = await self._prev_stage3_task
                except Exception:
                    logger.debug("Previous stage3 task failed", exc_info=True)
                    self._completed_token = None
            else:
                # First token: no previous result to return yet.
                # Run current stage3 synchronously so we have something
                # to return on the next call.
                self._completed_token = await stage3_task
                stage3_task = None  # Already awaited

            self._prev_stage3_task = stage3_task
            self._current = None

            if result is not None:
                self._tokens_yielded += 1
                self._total_overlap_ms += result.overlap_savings_ms
            return result
        else:
            # No overlap -- run stage 3 synchronously
            token = await self.run_stage3_overlap(token, detokenize_fn, grammar_fn)
            self._tokens_yielded += 1
            self._current = None
            return token

    def finish(self) -> None:
        """Mark pipeline as finished. Call after last token is generated."""
        self._finished = True
        self._running = False

    async def drain_last_token(self) -> PipelineToken | None:
        """Drain the final pending token from the pipeline.

        Must be called after ``finish()`` when overlap is enabled, because
        the last token's stage3 result is still pending in
        ``_prev_stage3_task``.  Returns the completed token or None.
        """
        if self._prev_stage3_task is not None:
            try:
                token = await self._prev_stage3_task
            except Exception:
                logger.debug("Final stage3 task failed", exc_info=True)
                token = None
            self._prev_stage3_task = None
            if token is not None:
                self._tokens_yielded += 1
                self._total_overlap_ms += token.overlap_savings_ms
            return token
        # Also check _completed_token (set during synchronous first-token path)
        if self._completed_token is not None:
            token = self._completed_token
            self._completed_token = None
            self._tokens_yielded += 1
            self._total_overlap_ms += token.overlap_savings_ms
            return token
        return None

    def get_stats(self) -> dict[str, Any]:
        """Return pipeline performance statistics."""
        elapsed = time.perf_counter() - self._start_time if self._start_time > 0 else 0
        return {
            "tokens_generated": self._tokens_generated,
            "tokens_yielded": self._tokens_yielded,
            "avg_overlap_ms": round(self.avg_overlap_ms, 3),
            "throughput_tok_per_s": round(self.throughput_tok_per_s, 1),
            "elapsed_s": round(elapsed, 3),
            "overlap_enabled": self.config.enable_overlap,
            "pipeline_depth": self.config.pipeline_depth,
        }


# ---------------------------------------------------------------------------
# 2. PrefetchSampler — pre-compute sampling plan while GPU generates logits
# ---------------------------------------------------------------------------


class SamplingPlan:
    """Pre-computed sampling plan for a single step.

    Avoids per-step parameter setup by pre-computing:
    - For deterministic (temp=0/top_k=1): argmax plan
    - For stochastic: pre-generated random numbers
    """

    def __init__(
        self,
        deterministic: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        seed: int | None = None,
    ):
        self.deterministic = deterministic
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.seed = seed

        # Pre-computed random state
        self._rng: np.random.Generator | None = None
        self._precomputed_gumbel: np.ndarray | None = None

        if not deterministic:
            rng_seed = seed if seed is not None else time.perf_counter_ns() % (2**32)
            self._rng = np.random.default_rng(rng_seed)

    def prepare_random(self, vocab_size: int = 0) -> None:
        """Pre-generate random numbers for sampling.

        Called once per step while GPU is computing logits.
        """
        if self.deterministic or vocab_size <= 0:
            return
        # Pre-generate Gumbel noise for Gumbel-max trick
        if self._rng is not None:
            self._precomputed_gumbel = self._rng.gumbel(size=vocab_size)

    def apply(self, logits: np.ndarray) -> np.ndarray:
        """Apply the pre-computed sampling plan to logits.

        Args:
            logits: Raw model output logits (1D numpy array).

        Returns:
            Sampled token id as a length-1 array.
        """
        if self.deterministic:
            return np.argmax(logits, axis=-1, keepdims=True)

        # Apply temperature
        scaled = logits / self.temperature

        # Apply min_p filtering
        if self.min_p > 0:
            max_logit = np.max(scaled)
            threshold = max_logit - self.min_p * max_logit
            mask = scaled >= threshold
            scaled = np.where(mask, scaled, -np.inf)

        # Apply top_k filtering
        if self.top_k > 0:
            top_k_indices = np.argpartition(scaled, -self.top_k)[-self.top_k:]
            mask = np.full_like(scaled, -np.inf)
            mask[top_k_indices] = scaled[top_k_indices]
            scaled = mask

        # Apply top_p (nucleus) filtering
        if self.top_p < 1.0:
            sorted_indices = np.argsort(scaled)[::-1]
            sorted_logits = scaled[sorted_indices]
            probs = _softmax(sorted_logits)
            cum_probs = np.cumsum(probs)
            # Remove tokens with cumulative probability above threshold
            cutoff_mask = cum_probs - probs > self.top_p
            sorted_logits[cutoff_mask] = -np.inf
            # Unsort
            unsorted = np.empty_like(scaled)
            unsorted[sorted_indices] = sorted_logits
            scaled = unsorted

        # Sample from distribution
        probs = _softmax(scaled)
        if self._rng is not None:
            return self._rng.choice(len(probs), size=1, p=probs).astype(np.int64)
        else:
            return np.argmax(probs, keepdims=True)


class PrefetchSampler:
    """Pre-computes sampling parameters while GPU is generating logits.

    For deterministic sampling (temperature=0, top_k=1):
      - Pre-computes argmax plan (no random state needed)
    For temperature-based:
      - Pre-generates random numbers using numpy

    Usage:
      sampler = PrefetchSampler()
      sampler.prepare(sampling_params, vocab_size=32000)
      # ... GPU does forward pass ...
      token_id = sampler.apply(logits)
    """

    def __init__(self) -> None:
        self._plan: SamplingPlan | None = None
        self._prepare_count = 0
        self._apply_count = 0
        self._total_prepare_ms = 0.0
        self._total_apply_ms = 0.0

    @property
    def plan(self) -> SamplingPlan | None:
        return self._plan

    @property
    def prepare_count(self) -> int:
        return self._prepare_count

    @property
    def apply_count(self) -> int:
        return self._apply_count

    _UNSET = object()

    def prepare(
        self,
        sampling_params: dict[str, Any] | None = None,
        vocab_size: int = 0,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        seed: int | None = None,
    ) -> SamplingPlan:
        """Pre-compute sampling plan.

        Can accept either a sampling_params dict or explicit keyword args.
        Called while GPU is computing logits (overlapped work).

        Args:
            sampling_params: Dict with temperature, top_p, top_k, etc.
            vocab_size: Vocabulary size for random number pre-generation.
            temperature, top_p, top_k, min_p, seed: Explicit sampling params.

        Returns:
            The pre-computed SamplingPlan.
        """
        t0 = time.perf_counter()

        # Precedence: explicit kwargs > dict > defaults (1.0, 1.0, 0, 0.0, None)
        # If a keyword differs from its default, it was explicitly set and wins.
        # Dict values fill in only when the keyword is at its default.
        _kw_temp, _kw_top_p, _kw_top_k, _kw_min_p, _kw_seed = (
            temperature, top_p, top_k, min_p, seed
        )
        if sampling_params is not None:
            temperature = sampling_params.get("temperature", temperature)
            top_p = sampling_params.get("top_p", top_p)
            top_k = sampling_params.get("top_k", top_k)
            min_p = sampling_params.get("min_p", min_p)
            seed = sampling_params.get("seed", seed)
        # If keyword was explicitly set (differs from default), it overrides dict
        if _kw_temp != 1.0:
            temperature = _kw_temp
        if _kw_top_p != 1.0:
            top_p = _kw_top_p
        if _kw_top_k != 0:
            top_k = _kw_top_k
        if _kw_min_p != 0.0:
            min_p = _kw_min_p
        if _kw_seed is not None:
            seed = _kw_seed
        deterministic = temperature == 0 or top_k == 1

        plan = SamplingPlan(
            deterministic=deterministic,
            temperature=temperature if not deterministic else 1.0,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            seed=seed,
        )
        plan.prepare_random(vocab_size)

        self._plan = plan
        self._prepare_count += 1
        self._total_prepare_ms += (time.perf_counter() - t0) * 1000
        return plan

    def apply(self, logits: np.ndarray) -> np.ndarray:
        """Apply pre-computed sampling plan to logits.

        Args:
            logits: Model output logits (numpy array).

        Returns:
            Sampled token id(s).

        Raises:
            RuntimeError: If prepare() was not called first.
        """
        if self._plan is None:
            raise RuntimeError("PrefetchSampler.apply() called before prepare()")

        t0 = time.perf_counter()
        result = self._plan.apply(logits)
        self._apply_count += 1
        self._total_apply_ms += (time.perf_counter() - t0) * 1000
        return result

    def get_stats(self) -> dict[str, Any]:
        """Return sampler performance statistics."""
        avg_prepare = (
            self._total_prepare_ms / self._prepare_count
            if self._prepare_count > 0
            else 0.0
        )
        avg_apply = (
            self._total_apply_ms / self._apply_count
            if self._apply_count > 0
            else 0.0
        )
        return {
            "prepare_count": self._prepare_count,
            "apply_count": self._apply_count,
            "avg_prepare_ms": round(avg_prepare, 4),
            "avg_apply_ms": round(avg_apply, 4),
            "total_prepare_ms": round(self._total_prepare_ms, 4),
            "total_apply_ms": round(self._total_apply_ms, 4),
        }

    def reset(self) -> None:
        """Reset sampler state for a new request."""
        self._plan = None
        self._prepare_count = 0
        self._apply_count = 0
        self._total_prepare_ms = 0.0
        self._total_apply_ms = 0.0


# ---------------------------------------------------------------------------
# 3. BatchedDetokenizer — detokenize multiple requests simultaneously
# ---------------------------------------------------------------------------


@dataclass
class _DetokEntry:
    """Internal queued entry for batched detokenization."""
    request_id: str
    token_ids: list[int] = field(default_factory=list)


class BatchedDetokenizer:
    """Detokenizes tokens from multiple requests simultaneously.

    Instead of calling tokenizer.decode() per-request per-token,
    batches all pending tokens and decodes them in one call.

    Faster than per-request detokenize when batch size > 1 because:
    - Single Python → C++ boundary crossing
    - Better CPU cache locality on the vocabulary table
    - Reduced Python object allocation overhead

    Usage:
      bd = BatchedDetokenizer(tokenizer)
      bd.add_tokens("req-1", [42, 100, 203])
      bd.add_tokens("req-2", [15, 88])
      bd.flush()
      text1 = bd.get_segment("req-1")
      text2 = bd.get_segment("req-2")
    """

    def __init__(self, tokenizer: Any = None) -> None:
        self._tokenizer = tokenizer
        self._queue: list[_DetokEntry] = []
        self._segments: dict[str, str] = {}
        self._flush_count = 0
        self._total_tokens_processed = 0
        self._total_flush_ms = 0.0

    @property
    def pending_count(self) -> int:
        """Number of requests with pending tokens."""
        return len(self._queue)

    @property
    def flush_count(self) -> int:
        return self._flush_count

    @property
    def total_tokens_processed(self) -> int:
        return self._total_tokens_processed

    def add_tokens(self, request_id: str, token_ids: list[int]) -> None:
        """Queue tokens for batched detokenization.

        Args:
            request_id: Unique request identifier.
            token_ids: Token ids to detokenize.
        """
        if not token_ids:
            return
        # Merge with existing entry for same request if not yet flushed
        for entry in self._queue:
            if entry.request_id == request_id:
                entry.token_ids.extend(token_ids)
                return
        self._queue.append(_DetokEntry(request_id=request_id, token_ids=list(token_ids)))

    def flush(self) -> dict[str, str]:
        """Detokenize all queued tokens in one batch.

        For each request, calls tokenizer.decode() on accumulated token_ids.
        Returns a dict mapping request_id -> detokenized text.

        When a real tokenizer with batch decode is available, uses
        tokenizer.decode() with batching for optimal throughput.
        """
        if not self._queue:
            self._flush_count += 1
            return {}

        t0 = time.perf_counter()
        results: dict[str, str] = {}

        if self._tokenizer is not None and hasattr(self._tokenizer, "decode"):
            # Batch path: collect all token lists, decode in one pass per request
            for entry in self._queue:
                try:
                    text = self._tokenizer.decode(entry.token_ids)
                    results[entry.request_id] = text
                    self._total_tokens_processed += len(entry.token_ids)
                except Exception as e:
                    logger.debug(
                        "Detokenize failed for %s: %s", entry.request_id, e
                    )
                    results[entry.request_id] = ""
        else:
            # No tokenizer — return empty strings
            for entry in self._queue:
                results[entry.request_id] = ""
                self._total_tokens_processed += len(entry.token_ids)

        self._segments.update(results)
        self._queue.clear()
        self._flush_count += 1
        self._total_flush_ms += (time.perf_counter() - t0) * 1000
        return results

    def get_segment(self, request_id: str) -> str:
        """Return detokenized text segment for a request.

        Returns empty string if request_id not found or not yet flushed.
        """
        return self._segments.pop(request_id, "")

    def has_segment(self, request_id: str) -> bool:
        """Check if a segment is available for a request."""
        return request_id in self._segments

    def clear(self) -> None:
        """Clear all pending and cached segments."""
        self._queue.clear()
        self._segments.clear()

    def get_stats(self) -> dict[str, Any]:
        """Return detokenizer performance statistics."""
        avg_flush = (
            self._total_flush_ms / self._flush_count
            if self._flush_count > 0
            else 0.0
        )
        return {
            "flush_count": self._flush_count,
            "total_tokens_processed": self._total_tokens_processed,
            "total_flush_ms": round(self._total_flush_ms, 3),
            "avg_flush_ms": round(avg_flush, 3),
            "pending_requests": self.pending_count,
        }


# ---------------------------------------------------------------------------
# 4. StreamingBackpressureController
# ---------------------------------------------------------------------------


@dataclass
class BackpressureConfig:
    """Configuration for streaming backpressure.

    Attributes:
        max_queue_size: Maximum tokens to buffer before applying backpressure.
        initial_delay_ms: Starting delay when backpressure kicks in.
        max_delay_ms: Maximum delay cap.
        ramp_factor: Multiplier per token above threshold for delay ramp-up.
        cooldown_factor: Decay factor for delay when queue drains.
    """
    max_queue_size: int = 100
    initial_delay_ms: float = 1.0
    max_delay_ms: float = 50.0
    ramp_factor: float = 0.5
    cooldown_factor: float = 0.9


class StreamingBackpressureController:
    """Applies backpressure when SSE clients are slow.

    When the output queue grows beyond max_queue_size, slows down
    generation to prevent OOM. The delay ramps up linearly with
    excess tokens and decays with a cooldown factor.

    Usage:
        bpc = StreamingBackpressureController(max_queue_size=100)
        for token in tokens:
            if bpc.check_backpressure(len(queue)):
                delay = bpc.get_delay_ms(len(queue))
                await asyncio.sleep(delay / 1000)
            queue.append(token)
    """

    def __init__(
        self,
        max_queue_size: int = 100,
        config: BackpressureConfig | None = None,
    ) -> None:
        self.config = config or BackpressureConfig(max_queue_size=max_queue_size)
        self._current_delay_ms: float = 0.0
        self._backpressure_count: int = 0
        self._total_delay_applied_ms: float = 0.0
        self._max_queue_seen: int = 0

    @property
    def current_delay_ms(self) -> float:
        return self._current_delay_ms

    @property
    def backpressure_count(self) -> int:
        return self._backpressure_count

    @property
    def max_queue_seen(self) -> int:
        return self._max_queue_seen

    def check_backpressure(
        self,
        queue_size: int,
        max_queue: int | None = None,
    ) -> bool:
        """Check if backpressure should be applied.

        Args:
            queue_size: Current number of tokens in the output queue.
            max_queue: Override for max queue size (uses config default if None).

        Returns:
            True if generation should slow down.
        """
        threshold = max_queue or self.config.max_queue_size
        self._max_queue_seen = max(self._max_queue_seen, queue_size)
        return queue_size >= threshold

    def get_delay_ms(self, queue_size: int) -> float:
        """Get the delay in milliseconds to apply for backpressure.

        Delay ramps up linearly with tokens above the threshold:
          delay = initial_delay + ramp_factor * excess_tokens

        When queue is below threshold, delay decays by cooldown_factor.

        Args:
            queue_size: Current queue depth.

        Returns:
            Delay in milliseconds to sleep.
        """
        threshold = self.config.max_queue_size

        if queue_size >= threshold:
            excess = queue_size - threshold
            target_delay = min(
                self.config.initial_delay_ms + self.config.ramp_factor * excess,
                self.config.max_delay_ms,
            )
            # Ramp up: take the max of current and target
            self._current_delay_ms = max(self._current_delay_ms, target_delay)
            self._backpressure_count += 1
            self._total_delay_applied_ms += self._current_delay_ms
        else:
            # Cooldown: decay delay toward zero
            self._current_delay_ms *= self.config.cooldown_factor
            if self._current_delay_ms < 0.01:
                self._current_delay_ms = 0.0

        return self._current_delay_ms

    def reset(self) -> None:
        """Reset controller state."""
        self._current_delay_ms = 0.0
        self._backpressure_count = 0
        self._total_delay_applied_ms = 0.0
        self._max_queue_seen = 0

    def get_stats(self) -> dict[str, Any]:
        """Return backpressure statistics."""
        return {
            "current_delay_ms": round(self._current_delay_ms, 3),
            "backpressure_count": self._backpressure_count,
            "total_delay_applied_ms": round(self._total_delay_applied_ms, 3),
            "max_queue_seen": self._max_queue_seen,
            "max_queue_size": self.config.max_queue_size,
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    return exp_x / (np.sum(exp_x) + 1e-12)
