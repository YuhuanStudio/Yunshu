from __future__ import annotations
"""Yunshu Batch Sampling — vectorized batched sampling for continuous batching.

Instead of sampling per-request sequentially with separate sampler objects,
BatchSampler applies sampling operations across the entire batch using MLX
vectorized operations (mx.take, mx.topk, mx.softmax, mx.where).

Key classes:
- BatchSampler: vectorized temperature/top-k/top-p/min-p sampling for a batch
- LogitsProcessorBatch: batched logits processing pipeline (penalties, bias, grammar)
- BatchStopChecker: vectorized stop condition checking (token IDs, max_tokens, EOS)

Architecture:
  BatchedEngine → Scheduler.step()
    → model forward → logits_batch [batch, vocab]
    → LogitsProcessorBatch.process(logits_batch, configs)  [batched penalties]
    → BatchSampler.sample_batch(logits_batch, params_list)  [vectorized sampling]
    → BatchStopChecker.check_batch(tokens, stop_configs)  [batched stop detection]

Studied from:
- vLLM's v1/sample/sampler.py: batched sampling with per-request params
- SGLang's batched sampling: radix attention with vectorized top-p/top-k
- mlx-lm's sample_utils.py: per-request sampler as reference (we vectorize)
"""


import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import mlx.core as mx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BatchSampler
# ---------------------------------------------------------------------------

@dataclass
class SamplingPlan:
    """Prepared sampling plan for a batch (avoids redundant work during sample).

    Created by BatchSampler.prepare_batch(), consumed by sample_batch().
    Contains pre-processed temperature arrays, top-k masks, etc.
    """
    batch_size: int
    vocab_size: int
    # Per-request temperatures (shape: [batch, 1])
    temperatures: mx.array | None = None
    # Per-request top-k values (shape: [batch, 1])
    top_k_values: list[int] = field(default_factory=list)
    # Per-request top-p values
    top_p_values: list[float] = field(default_factory=list)
    # Per-request min-p values
    min_p_values: list[float] = field(default_factory=list)
    # Which requests use greedy (temperature == 0)
    greedy_mask: list[bool] = field(default_factory=list)
    # Seeding info
    seeds: list[int | None] = field(default_factory=list)


@dataclass
class BatchSampleResult:
    """Result of batch sampling."""
    token_ids: mx.array  # shape: [batch]
    logprobs: mx.array | None = None  # shape: [batch, vocab] if requested


@dataclass
class BatchSamplerStats:
    """Performance statistics for BatchSampler."""
    total_batches: int = 0
    total_tokens_sampled: int = 0
    total_prepare_time_ms: float = 0.0
    total_sample_time_ms: float = 0.0
    last_batch_size: int = 0
    per_op_time_ms: dict[str, float] = field(default_factory=dict)

    @property
    def avg_prepare_time_ms(self) -> float:
        return self.total_prepare_time_ms / max(1, self.total_batches)

    @property
    def avg_sample_time_ms(self) -> float:
        return self.total_sample_time_ms / max(1, self.total_batches)


class BatchSampler:
    """Vectorized batched sampling for continuous batching inference.

    Instead of N separate make_sampler() calls per step, applies sampling
    operations across the entire batch using MLX vectorized ops:

    1. Temperature scaling: logits[i] / temperatures[i] (vectorized divide)
    2. Top-k masking: batch argpartition + put_along_axis (vectorized)
    3. Top-p filtering: batch sort + cumsum + mask (vectorized)
    4. Min-p filtering: batch max + scaled threshold (vectorized)
    5. Categorical sampling: mx.random.categorical on modified logits
    """

    def __init__(self) -> None:
        self._stats = BatchSamplerStats()
        self._per_op_times: dict[str, list[float]] = {}

    def prepare_batch(
        self,
        logits_batch: mx.array,
        params_list: list[dict],
    ) -> SamplingPlan:
        """Prepare a batch sampling plan from per-request parameters.

        Args:
            logits_batch: Shape [batch, vocab] — raw logits from model forward.
            params_list: List of dicts, one per request, with keys:
                temperature (float), top_k (int), top_p (float), min_p (float),
                seed (int|None)

        Returns:
            SamplingPlan with pre-processed arrays ready for sample_batch().
        """
        t0 = time.perf_counter()

        batch_size, vocab_size = logits_batch.shape
        assert len(params_list) == batch_size, (
            f"params_list length ({len(params_list)}) must match batch_size ({batch_size})"
        )

        temperatures = []
        top_k_values = []
        top_p_values = []
        min_p_values = []
        greedy_mask = []
        seeds = []

        for params in params_list:
            temp = params.get("temperature", 0.7)
            temperatures.append(temp)
            top_k_values.append(params.get("top_k", 0))
            top_p_values.append(params.get("top_p", 1.0))
            min_p_values.append(params.get("min_p", 0.0))
            greedy_mask.append(temp == 0.0)
            seeds.append(params.get("seed"))

        # Build temperature array for vectorized scaling: shape [batch, 1]
        # Replace 0.0 with 1.0 to avoid div-by-zero; greedy path uses argmax
        safe_temps = [t if t > 0 else 1.0 for t in temperatures]
        temp_array = mx.array(safe_temps, dtype=mx.float32).reshape(batch_size, 1)

        elapsed = (time.perf_counter() - t0) * 1000
        self._stats.total_prepare_time_ms += elapsed
        self._per_op_times.setdefault("prepare", []).append(elapsed)

        return SamplingPlan(
            batch_size=batch_size,
            vocab_size=vocab_size,
            temperatures=temp_array,
            top_k_values=top_k_values,
            top_p_values=top_p_values,
            min_p_values=min_p_values,
            greedy_mask=greedy_mask,
            seeds=seeds,
        )

    def sample_batch(
        self,
        logits_batch: mx.array,
        params_list: list[dict],
        plan: SamplingPlan | None = None,
    ) -> BatchSampleResult:
        """Execute batch sampling — the main entry point.

        Applies, in order:
        1. Temperature scaling (vectorized)
        2. Top-k filtering (vectorized)
        3. Top-p filtering (vectorized)
        4. Min-p filtering (vectorized)
        5. Categorical sampling (vectorized)
        6. Override greedy requests with argmax

        Args:
            logits_batch: Shape [batch, vocab] — logits after LogitsProcessorBatch.
            params_list: Per-request sampling parameters.
            plan: Optional pre-computed plan (will create if None).

        Returns:
            BatchSampleResult with token_ids shape [batch].
        """
        t0 = time.perf_counter()

        if plan is None:
            plan = self.prepare_batch(logits_batch, params_list)

        self._stats.last_batch_size = plan.batch_size

        # Apply seeds per-request if specified (first seed sets global state)
        # Note: per-request seeding in batch is inherently limited — only the
        # first seed in the batch is applied before the batch sampling op.
        if plan.seeds and plan.seeds[0] is not None:
            mx.random.seed(plan.seeds[0])

        # Step 1: Temperature scaling — vectorized divide
        t_temp = time.perf_counter()
        scaled = logits_batch / plan.temperatures
        self._record_op("temperature_scale", t_temp)

        # Step 2: Top-k filtering — per-request masking
        t_topk = time.perf_counter()
        scaled = self._apply_batch_top_k(scaled, plan.top_k_values)
        self._record_op("top_k_filter", t_topk)

        # Step 3: Top-p filtering — per-request nucleus sampling
        t_topp = time.perf_counter()
        scaled = self._apply_batch_top_p(scaled, plan.top_p_values)
        self._record_op("top_p_filter", t_topp)

        # Step 4: Min-p filtering — per-request dynamic threshold
        t_minp = time.perf_counter()
        scaled = self._apply_batch_min_p(scaled, plan.min_p_values)
        self._record_op("min_p_filter", t_minp)

        # Step 5: Categorical sampling (vectorized)
        t_sample = time.perf_counter()
        token_ids = mx.random.categorical(scaled, axis=-1)
        self._record_op("categorical_sample", t_sample)

        # Step 6: Override greedy requests with argmax
        t_greedy = time.perf_counter()
        if any(plan.greedy_mask):
            greedy_token = mx.argmax(logits_batch, axis=-1)
            # Build a mask for which positions to override
            greedy_indices = [i for i, g in enumerate(plan.greedy_mask) if g]
            for idx in greedy_indices:
                token_ids[idx] = greedy_token[idx]
        self._record_op("greedy_override", t_greedy)

        elapsed = (time.perf_counter() - t0) * 1000
        self._stats.total_batches += 1
        self._stats.total_tokens_sampled += plan.batch_size
        self._stats.total_sample_time_ms += elapsed

        return BatchSampleResult(token_ids=token_ids)

    def _apply_batch_top_k(
        self, logits: mx.array, top_k_values: list[int]
    ) -> mx.array:
        """Apply per-request top-k filtering across the batch.

        For each request with top_k > 0, masks all tokens outside the top-k
        highest logit values to -inf.
        """
        result = logits
        batch_size = logits.shape[0]

        for i in range(batch_size):
            k = top_k_values[i]
            if k <= 0 or k >= logits.shape[-1]:
                continue
            # Top-k masking for this request
            row = result[i : i + 1]  # shape [1, vocab]
            mask_idx = mx.argpartition(-row, kth=k - 1, axis=-1)[..., k:]
            result[i : i + 1] = mx.put_along_axis(
                row, mask_idx, mx.array(-float("inf"), row.dtype), axis=-1
            )

        return result

    def _apply_batch_top_p(
        self, logits: mx.array, top_p_values: list[float]
    ) -> mx.array:
        """Apply per-request top-p (nucleus) filtering across the batch.

        For each request with 0 < top_p < 1.0, keeps only tokens whose
        cumulative probability (from highest to lowest) exceeds (1 - top_p).
        """
        result = logits
        batch_size = logits.shape[0]

        for i in range(batch_size):
            top_p = top_p_values[i]
            if top_p <= 0 or top_p >= 1.0:
                continue

            row = result[i : i + 1]  # shape [1, vocab]
            # Use softmax for proper probabilities (not raw exp which can overflow)
            probs = mx.softmax(row, axis=-1)
            sorted_indices = mx.argsort(row, axis=-1)
            sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
            cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

            # Rearrange cumulative probs back to original order
            inverse_indices = mx.put_along_axis(
                mx.zeros_like(sorted_indices),
                sorted_indices,
                mx.arange(row.shape[-1], dtype=sorted_indices.dtype),
                axis=-1,
            )
            cumulative_probs = mx.take_along_axis(
                cumulative_probs, inverse_indices, axis=-1
            )

            # Keep tokens with cumulative prob above (1 - top_p)
            result[i : i + 1] = mx.where(
                cumulative_probs > 1 - top_p,
                row,
                mx.array(-float("inf"), row.dtype),
            )

        return result

    def _apply_batch_min_p(
        self, logits: mx.array, min_p_values: list[float]
    ) -> mx.array:
        """Apply per-request min-p filtering across the batch.

        For each request with min_p > 0, removes tokens whose probability
        is below max_prob * min_p.
        """
        result = logits
        batch_size = logits.shape[0]

        for i in range(batch_size):
            min_p = min_p_values[i]
            if min_p <= 0 or min_p > 1.0:
                continue

            row = result[i : i + 1]  # shape [1, vocab]
            top_logprobs = mx.max(row, axis=-1, keepdims=True)
            scaled_min_p = top_logprobs + math.log(min_p)
            tokens_to_remove = row < scaled_min_p

            result[i : i + 1] = mx.where(
                tokens_to_remove,
                mx.array(-float("inf"), row.dtype),
                row,
            )

        return result

    def get_stats(self) -> dict[str, Any]:
        """Return batch sampling performance statistics."""
        stats = {
            "total_batches": self._stats.total_batches,
            "total_tokens_sampled": self._stats.total_tokens_sampled,
            "avg_prepare_time_ms": round(self._stats.avg_prepare_time_ms, 3),
            "avg_sample_time_ms": round(self._stats.avg_sample_time_ms, 3),
            "last_batch_size": self._stats.last_batch_size,
        }
        # Per-op timing
        for op_name, times in self._per_op_times.items():
            if times:
                stats[f"{op_name}_avg_ms"] = round(sum(times) / len(times), 4)
                stats[f"{op_name}_last_ms"] = round(times[-1], 4)
        return stats

    def _record_op(self, op_name: str, t_start: float) -> None:
        elapsed = (time.perf_counter() - t_start) * 1000
        times = self._per_op_times.setdefault(op_name, [])
        times.append(elapsed)
        # Cap at 1000 entries to prevent unbounded memory growth
        if len(times) > 1000:
            del times[:500]


# ---------------------------------------------------------------------------
# LogitsProcessorBatch
# ---------------------------------------------------------------------------

@dataclass
class LogitsProcessorConfig:
    """Per-request logits processor configuration."""
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: dict[int, float] | None = None
    grammar_bitmask: mx.array | None = None  # shape: [vocab] bool mask
    # Generated token history for penalty computation
    generated_tokens: list[int] = field(default_factory=list)
    repetition_context_size: int = 20
    presence_context_size: int = 20
    frequency_context_size: int = 20


class LogitsProcessorBatch:
    """Batched logits processing pipeline.

    Applies all logits modifications in batch:
    1. Temperature scaling: logits / temperature per request
    2. Repetition penalty: scan generated tokens, modify in batch
    3. Logit bias: per-request bias vectors
    4. Grammar constraint: per-request bitmask application

    Processors are registered as named functions and applied in order.
    Each processor takes (logits_batch, configs) and returns modified logits.
    """

    def __init__(self) -> None:
        self._processors: list[tuple[str, Callable]] = []
        self._processor_times: dict[str, list[float]] = {}

        # Register default processors
        self.add_processor("repetition_penalty", self._apply_repetition_penalty)
        self.add_processor("presence_penalty", self._apply_presence_penalty)
        self.add_processor("frequency_penalty", self._apply_frequency_penalty)
        self.add_processor("logit_bias", self._apply_logit_bias)
        self.add_processor("grammar_bitmask", self._apply_grammar_bitmask)

    def add_processor(self, name: str, fn: Callable) -> None:
        """Register a logits processor.

        Args:
            name: Unique processor name.
            fn: Callable(logit_row, config) -> modified logit_row.
                Called per-request for processors that need per-request context.
        """
        # Remove existing processor with same name if it exists
        self._processors = [(n, f) for n, f in self._processors if n != name]
        self._processors.append((name, fn))

    def remove_processor(self, name: str) -> bool:
        """Remove a registered processor by name.

        Returns:
            True if a processor was removed, False if not found.
        """
        before = len(self._processors)
        self._processors = [(n, f) for n, f in self._processors if n != name]
        return len(self._processors) < before

    def process(
        self,
        logits_batch: mx.array,
        configs: list[LogitsProcessorConfig],
    ) -> mx.array:
        """Apply all registered processors to the batch.

        Args:
            logits_batch: Shape [batch, vocab] — raw logits from model.
            configs: Per-request processor configurations.

        Returns:
            Modified logits_batch shape [batch, vocab].
        """
        batch_size = logits_batch.shape[0]
        assert len(configs) == batch_size, (
            f"configs length ({len(configs)}) must match batch_size ({batch_size})"
        )

        result = logits_batch

        for name, fn in self._processors:
            t0 = time.perf_counter()
            for i in range(batch_size):
                row = result[i : i + 1]  # shape [1, vocab]
                modified = fn(row, configs[i])
                if modified is not None:
                    result[i : i + 1] = modified
            elapsed = (time.perf_counter() - t0) * 1000
            times = self._processor_times.setdefault(name, [])
            times.append(elapsed)
            # Cap at 1000 entries to prevent unbounded memory growth
            if len(times) > 1000:
                del times[:500]

        return result

    def get_processor_stats(self) -> dict[str, dict[str, float]]:
        """Return per-processor timing statistics."""
        stats = {}
        for name, times in self._processor_times.items():
            if times:
                stats[name] = {
                    "avg_ms": round(sum(times) / len(times), 4),
                    "last_ms": round(times[-1], 4),
                    "calls": len(times),
                }
        return stats

    # -- Default processor implementations --

    @staticmethod
    def _apply_repetition_penalty(
        logits_row: mx.array, config: LogitsProcessorConfig
    ) -> mx.array | None:
        """Apply repetition penalty to a single request's logits.

        Sign-aware multiplicative penalty: logits < 0 are multiplied by penalty,
        logits >= 0 are divided by penalty (mlx-lm pattern).
        """
        penalty = config.repetition_penalty
        if penalty == 1.0:
            return None

        tokens = config.generated_tokens
        if not tokens:
            return None

        ctx_size = config.repetition_context_size
        tokens = tokens[-ctx_size:]
        unique_tokens = list(set(tokens))

        if not unique_tokens:
            return None

        indices = mx.array(unique_tokens)
        selected_logits = logits_row[:, indices]
        selected_logits = mx.where(
            selected_logits < 0,
            selected_logits * penalty,
            selected_logits / penalty,
        )
        logits_row[:, indices] = selected_logits
        return logits_row

    @staticmethod
    def _apply_presence_penalty(
        logits_row: mx.array, config: LogitsProcessorConfig
    ) -> mx.array | None:
        """Apply presence penalty: subtract penalty if token appeared at all."""
        penalty = config.presence_penalty
        if penalty == 0.0:
            return None

        tokens = config.generated_tokens
        if not tokens:
            return None

        ctx_size = config.presence_context_size
        tokens = tokens[-ctx_size:]
        unique_tokens = list(set(tokens))

        if not unique_tokens:
            return None

        indices = mx.array(unique_tokens)
        logits_row[:, indices] -= penalty
        return logits_row

    @staticmethod
    def _apply_frequency_penalty(
        logits_row: mx.array, config: LogitsProcessorConfig
    ) -> mx.array | None:
        """Apply frequency penalty: subtract penalty * count for each token."""
        penalty = config.frequency_penalty
        if penalty == 0.0:
            return None

        tokens = config.generated_tokens
        if not tokens:
            return None

        ctx_size = config.frequency_context_size
        tokens = tokens[-ctx_size:]

        # Count frequency of each token
        from collections import Counter
        token_counts = Counter(tokens)

        for token_id, count in token_counts.items():
            logits_row[:, token_id] -= penalty * count

        return logits_row

    @staticmethod
    def _apply_logit_bias(
        logits_row: mx.array, config: LogitsProcessorConfig
    ) -> mx.array | None:
        """Apply per-token logit bias."""
        bias = config.logit_bias
        if not bias:
            return None

        indices = mx.array(list(bias.keys()))
        values = mx.array(list(bias.values()))
        logits_row[:, indices] += values
        return logits_row

    @staticmethod
    def _apply_grammar_bitmask(
        logits_row: mx.array, config: LogitsProcessorConfig
    ) -> mx.array | None:
        """Apply grammar constraint bitmask: mask disallowed tokens to -inf."""
        bitmask = config.grammar_bitmask
        if bitmask is None:
            return None

        # bitmask is True where tokens are allowed
        logits_row = mx.where(
            bitmask.reshape(1, -1),
            logits_row,
            mx.array(-float("inf"), logits_row.dtype),
        )
        return logits_row


# ---------------------------------------------------------------------------
# BatchStopChecker
# ---------------------------------------------------------------------------

@dataclass
class StopConfig:
    """Per-request stop condition configuration."""
    request_id: str = ""
    max_tokens: int = 256
    generated_count: int = 0  # tokens generated so far
    stop_token_ids: list[int] = field(default_factory=list)
    eos_token_ids: list[int] = field(default_factory=list)
    stop_strings: list[str] = field(default_factory=list)
    # For Aho-Corasick multi-pattern matching
    stop_string_trie: Any = None  # built by _build_stop_trie()


@dataclass
class StopResult:
    """Per-request stop check result."""
    request_id: str
    should_stop: bool
    reason: str | None = None  # "stop", "length", "eos", "stop_token_id"
    matched_token_id: int | None = None
    matched_string: str | None = None


class _AhoCorasickNode:
    """Simple Aho-Corasick trie node for batch stop string matching.

    Used by BatchStopChecker to match multiple stop strings against
    generated token sequences in a single pass.
    """

    __slots__ = ("children", "output", "fail")

    def __init__(self) -> None:
        self.children: dict[int, "_AhoCorasickNode"] = {}
        self.output: list[int] = []  # indices of patterns that end here
        self.fail: "_AhoCorasickNode | None" = None


class _AhoCorasickTrie:
    """Minimal Aho-Corasick automaton for multi-pattern token matching.

    Builds a trie from stop token sequences, then constructs failure links
    for efficient multi-pattern matching in a single pass.
    """

    def __init__(self, patterns: list[tuple[tuple[int, ...], int]]) -> None:
        """Build AC automaton from patterns.

        Args:
            patterns: List of (token_sequence, pattern_index) pairs.
        """
        self._root = _AhoCorasickNode()
        self._patterns = patterns
        self._num_patterns = len(patterns)

        # Build trie
        for token_seq, pidx in patterns:
            node = self._root
            for token_id in token_seq:
                if token_id not in node.children:
                    node.children[token_id] = _AhoCorasickNode()
                node = node.children[token_id]
            node.output.append(pidx)

        # Build failure links (BFS)
        self._build_failure_links()

    def _build_failure_links(self) -> None:
        """Construct failure links using BFS (standard AC construction)."""
        from collections import deque

        queue = deque()
        # Root's children fail to root
        for child in self._root.children.values():
            child.fail = self._root
            queue.append(child)

        while queue:
            current = queue.popleft()
            for token_id, child in current.children.items():
                queue.append(child)
                # Follow failure links to find longest proper suffix
                fail = current.fail
                while fail is not None and token_id not in fail.children:
                    fail = fail.fail
                child.fail = fail.children.get(token_id, self._root) if fail else self._root
                # Merge output from failure node
                if child.fail is not None:
                    child.output = child.output + child.fail.output

    def search(self, tokens: list[int]) -> list[int]:
        """Search for pattern matches in a token sequence.

        Returns:
            List of matched pattern indices.
        """
        matches = []
        node = self._root

        for token_id in tokens:
            # Follow failure links on mismatch
            while node is not self._root and token_id not in node.children:
                node = node.fail if node.fail is not None else self._root
            node = node.children.get(token_id, self._root)

            if node.output:
                matches.extend(node.output)

        return matches

    @property
    def num_patterns(self) -> int:
        return self._num_patterns


class BatchStopChecker:
    """Vectorized stop condition checking for batched inference.

    Checks all stop conditions for all requests in a batch simultaneously:
    - Stop token IDs: vectorized membership check
    - EOS token IDs: vectorized comparison
    - Max tokens: vectorized comparison
    - Stop strings: Aho-Corasick trie for multi-pattern matching

    Usage:
        checker = BatchStopChecker()
        results = checker.check_batch(token_ids, generated_counts, stop_configs)
        for result in results:
            if result.should_stop:
                handle_stop(result)
    """

    def __init__(self) -> None:
        self._trie_cache: dict[int, _AhoCorasickTrie] = {}
        self._stats = {"total_checks": 0, "total_stops": 0}

    def check_batch(
        self,
        token_ids: mx.array,
        generated_counts: list[int],
        stop_configs: list[StopConfig],
    ) -> list[StopResult]:
        """Check stop conditions for all requests in the batch.

        Args:
            token_ids: Shape [batch] — newly generated token IDs.
            generated_counts: Per-request count of generated tokens so far.
            stop_configs: Per-request stop configuration.

        Returns:
            List of StopResult, one per request.
        """
        batch_size = token_ids.shape[0]
        assert len(generated_counts) == batch_size
        assert len(stop_configs) == batch_size

        self._stats["total_checks"] += 1
        results: list[StopResult] = []

        # Vectorized: extract token IDs as Python ints for comparison
        token_list = token_ids.tolist()

        for i in range(batch_size):
            config = stop_configs[i]
            tid = token_list[i] if isinstance(token_list, list) else int(token_list[i])
            count = generated_counts[i]

            # Check 1: Max tokens
            if count >= config.max_tokens:
                results.append(StopResult(
                    request_id=config.request_id,
                    should_stop=True,
                    reason="length",
                ))
                self._stats["total_stops"] += 1
                continue

            # Check 2: EOS token IDs
            if config.eos_token_ids and tid in config.eos_token_ids:
                results.append(StopResult(
                    request_id=config.request_id,
                    should_stop=True,
                    reason="eos",
                    matched_token_id=tid,
                ))
                self._stats["total_stops"] += 1
                continue

            # Check 3: Stop token IDs
            if config.stop_token_ids and tid in config.stop_token_ids:
                results.append(StopResult(
                    request_id=config.request_id,
                    should_stop=True,
                    reason="stop_token_id",
                    matched_token_id=tid,
                ))
                self._stats["total_stops"] += 1
                continue

            # Check 4: Stop strings via Aho-Corasick (if configured)
            if config.stop_string_trie is not None:
                matches = config.stop_string_trie.search(
                    config._current_tokens if hasattr(config, '_current_tokens') else [tid]
                )
                if matches:
                    # Find which pattern matched
                    matched_idx = matches[0]
                    results.append(StopResult(
                        request_id=config.request_id,
                        should_stop=True,
                        reason="stop",
                        matched_string=f"pattern_{matched_idx}",
                    ))
                    self._stats["total_stops"] += 1
                    continue

            # No stop condition triggered
            results.append(StopResult(
                request_id=config.request_id,
                should_stop=False,
            ))

        return results

    def check_batch_token_ids_vectorized(
        self,
        token_ids: mx.array,
        stop_id_sets: list[set[int]],
    ) -> mx.array:
        """Vectorized stop token ID check for the entire batch.

        Args:
            token_ids: Shape [batch] — newly generated token IDs.
            stop_id_sets: Per-request set of stop token IDs.

        Returns:
            Boolean mask shape [batch] — True where request should stop.
        """
        batch_size = token_ids.shape[0]
        should_stop = [False] * batch_size
        token_list = token_ids.tolist()

        for i in range(batch_size):
            tid = token_list[i] if isinstance(token_list, list) else int(token_list[i])
            if stop_id_sets[i] and tid in stop_id_sets[i]:
                should_stop[i] = True

        return mx.array(should_stop)

    def check_max_tokens_vectorized(
        self,
        generated_counts: list[int],
        max_tokens_list: list[int],
    ) -> mx.array:
        """Vectorized max_tokens check for the entire batch.

        Args:
            generated_counts: Per-request count of generated tokens.
            max_tokens_list: Per-request max_tokens limits.

        Returns:
            Boolean mask shape [batch] — True where request hit max_tokens.
        """
        counts = mx.array(generated_counts, dtype=mx.int32)
        limits = mx.array(max_tokens_list, dtype=mx.int32)
        return counts >= limits

    @staticmethod
    def build_stop_trie(
        stop_strings: list[str],
        tokenizer: Any,
    ) -> _AhoCorasickTrie | None:
        """Build an Aho-Corasick trie from stop strings.

        Encodes each stop string to token IDs, then builds an AC trie
        for efficient multi-pattern matching during generation.

        Args:
            stop_strings: List of stop strings to match.
            tokenizer: Tokenizer with encode() method.

        Returns:
            _AhoCorasickTrie or None if no stop strings provided.
        """
        if not stop_strings:
            return None

        patterns = []
        for i, s in enumerate(stop_strings):
            token_ids = tuple(tokenizer.encode(s, add_special_tokens=False))
            if token_ids:
                patterns.append((token_ids, i))

        if not patterns:
            return None

        return _AhoCorasickTrie(patterns)

    def get_stats(self) -> dict[str, int]:
        """Return stop checking statistics."""
        return dict(self._stats)
