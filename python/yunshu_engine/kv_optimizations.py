"""Yunshu KV Cache Optimizations — production-grade memory and compute optimizations.

Four complementary subsystems that improve KV cache memory efficiency:

1. **AdaptiveKVQuantizer** — Per-layer adaptive KV quantization.
   Early layers (position-sensitive) stay at FP16, middle layers use INT8,
   late layers (less critical for next-token prediction) drop to INT4.
   Budget-aware: configures per-layer precision to fit within a byte budget.

2. **KVEvictionPredictor** — Predicts future block access using attention patterns.
   Uses exponential moving averages over per-block access history to predict
   which blocks will be needed soon. More intelligent than LRU/LFU because
   it can keep blocks predicted to be needed in the near future even if they
   were accessed long ago.

3. **ChunkedPrefillOptimizer** — Semantically-aware chunk boundary selection.
   Splits prompts at sentence/paragraph boundaries instead of arbitrary token
   positions. Fairly interleaves chunks from different requests in multi-request
   scenarios to prevent starvation.

4. **KVBlockCompactor** — Periodic KV cache block compaction.
   After preemption or eviction, KV cache blocks can become partially filled
   (internal fragmentation). The compactor merges adjacent partial blocks to
   free complete blocks, reducing memory waste.

References:
  - KIVI: Tuning-Free Asymmetric 2-bit Quantization (Wang et al.)
  - vLLM PagedAttention block management
  - SGLang RadixCache prefix matching
  - Sarathi chunked prefill interleaving
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Sentinel token IDs for semantic boundaries (common across tokenizers)
_SENTENCE_END_TOKENS: set[int] = set()  # Populated from tokenizer if available

# Bytes per element for different precisions
_FP16_BYTES = 2
_INT8_BYTES = 1
_INT4_BYTES = 0.5  # Packed nibbles: 2 values per byte

# EMA decay factor for eviction predictor
_EMA_DECAY = 0.9
# Minimum number of observations before predictions are reliable
_MIN_OBSERVATIONS = 3


# ═══════════════════════════════════════════════════════════════════════════════
# 1. AdaptiveKVQuantizer
# ═══════════════════════════════════════════════════════════════════════════════


class QuantTier(Enum):
    """Quantization tier for a KV cache layer."""
    FP16 = auto()  # 16-bit float — no quantization
    INT8 = auto()  # 8-bit integer
    INT4 = auto()  # 4-bit integer (packed nibbles)


@dataclass
class LayerQuantConfig:
    """Per-layer quantization configuration."""
    layer_idx: int
    tier: QuantTier
    bits: int
    bytes_per_element: float
    group_size: int = 64  # Group size for INT4/INT8 quantization
    symmetric: bool = True


@dataclass
class AdaptiveQuantStats:
    """Statistics from adaptive quantization."""
    per_layer_bits: dict[int, int] = field(default_factory=dict)
    per_layer_tier: dict[int, str] = field(default_factory=dict)
    total_bytes: int = 0
    fp16_baseline_bytes: int = 0
    memory_saved_pct: float = 0.0
    estimated_accuracy_impact: float = 0.0


class AdaptiveKVQuantizer:
    """Dynamically adjusts KV quantization bits based on layer importance.

    Research insight (KIVI, GPTQ): early layers are critical for positional
    encoding and token identity, while late layers contribute less to
    next-token prediction accuracy. By keeping early layers at FP16 and
    aggressively quantizing late layers, we save significant memory with
    minimal quality loss.

    Tier assignment strategy:
    - Layers [0, first_third): FP16 (16-bit) — position-sensitive
    - Layers [first_third, second_third): INT8 (8-bit) — balanced
    - Layers [second_third, end): INT4 (4-bit) — aggressive compression

    Budget-aware mode: when a budget_bytes limit is specified, the tier
    boundaries are adjusted to fit within the budget, potentially expanding
    INT4 coverage if needed.
    """

    def __init__(self) -> None:
        self._num_layers: int = 0
        self._num_kv_heads: int = 0
        self._head_dim: int = 0
        self._max_seq_len: int = 0
        self._layer_configs: list[LayerQuantConfig] = []
        self._configured: bool = False
        self._stats = AdaptiveQuantStats()

    def configure(
        self,
        model: Any = None,
        budget_bytes: int | None = None,
        *,
        num_layers: int = 0,
        num_kv_heads: int = 0,
        head_dim: int = 0,
        max_seq_len: int = 2048,
    ) -> list[LayerQuantConfig]:
        """Compute per-layer quantization configuration.

        Accepts either a model object (inspected for architecture info) or
        explicit architecture parameters. When budget_bytes is provided,
        adjusts tier boundaries to fit within the budget.

        Args:
            model: Model object with config/args containing architecture info.
            budget_bytes: Optional byte budget for the full KV cache.
            num_layers: Number of transformer layers (overrides model detection).
            num_kv_heads: Number of KV attention heads.
            head_dim: Dimension per attention head.
            max_seq_len: Maximum sequence length for budget estimation.

        Returns:
            List of LayerQuantConfig, one per layer.
        """
        # Extract architecture info from model if not provided directly
        if model is not None:
            config = getattr(model, 'config', None) or getattr(model, 'args', None)
            if config is not None:
                num_layers = num_layers or getattr(config, 'num_hidden_layers', 0) \
                    or getattr(config, 'n_layers', 0)
                num_kv_heads = num_kv_heads or getattr(config, 'num_key_value_heads', 0) \
                    or getattr(config, 'n_kv_heads', 0)
                head_dim = head_dim or getattr(config, 'head_dim', 0) \
                    or (getattr(config, 'hidden_size', 0) //
                        (getattr(config, 'num_attention_heads', 1) or 1))
                max_seq_len = max_seq_len or getattr(config, 'max_position_embeddings', 0) \
                    or getattr(config, 'max_sequence_length', 2048)

        if num_layers <= 0:
            raise ValueError("num_layers must be > 0 (pass model or num_layers)")
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be > 0")
        if head_dim <= 0:
            raise ValueError("head_dim must be > 0")

        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._max_seq_len = max_seq_len

        # Compute tier boundaries
        configs = self._compute_tier_config(num_layers, budget_bytes)
        self._layer_configs = configs
        self._configured = True

        # Update stats
        self._stats = self._compute_stats(configs, max_seq_len)

        logger.info(
            f"AdaptiveKVQuantizer configured: {num_layers} layers, "
            f"{num_kv_heads} KV heads, {head_dim}d, "
            f"budget={'unlimited' if budget_bytes is None else f'{budget_bytes / 1024**2:.1f}MB'}"
        )
        return configs

    def _compute_tier_config(
        self,
        num_layers: int,
        budget_bytes: int | None,
    ) -> list[LayerQuantConfig]:
        """Compute per-layer tier assignments.

        Default: 1/3 FP16, 1/3 INT8, 1/3 INT4.
        Budget-aware: adjusts boundaries to fit budget.
        """
        # Default tier boundaries
        fp16_end = max(1, num_layers // 3)
        int8_end = max(fp16_end + 1, 2 * num_layers // 3)

        # Budget-aware adjustment
        if budget_bytes is not None:
            fp16_end, int8_end = self._adjust_for_budget(
                num_layers, budget_bytes, fp16_end, int8_end,
            )

        configs = []
        for i in range(num_layers):
            if i < fp16_end:
                tier = QuantTier.FP16
                bits = 16
                bpe = _FP16_BYTES
                group_size = 0  # No group quantization for FP16
            elif i < int8_end:
                tier = QuantTier.INT8
                bits = 8
                bpe = _INT8_BYTES
                group_size = 64
            else:
                tier = QuantTier.INT4
                bits = 4
                bpe = _INT4_BYTES
                group_size = 64

            configs.append(LayerQuantConfig(
                layer_idx=i,
                tier=tier,
                bits=bits,
                bytes_per_element=bpe,
                group_size=group_size,
            ))

        return configs

    def _adjust_for_budget(
        self,
        num_layers: int,
        budget_bytes: int,
        fp16_end: int,
        int8_end: int,
    ) -> tuple[int, int]:
        """Adjust tier boundaries to fit within a byte budget.

        Strategy:
        1. Start with default boundaries
        2. Compute estimated total bytes
        3. If over budget, shrink FP16 region first, then INT8
        """
        for _ in range(num_layers + 1):
            est = self._estimate_total_bytes(num_layers, fp16_end, int8_end)
            if est <= budget_bytes:
                return fp16_end, int8_end

            # Shrink FP16 region (move boundary down)
            if fp16_end > 0:
                fp16_end -= 1
            elif int8_end > fp16_end + 1:
                int8_end -= 1
            else:
                # All INT4 — already max compression
                break

        # Final: all INT4 if still over budget
        est = self._estimate_total_bytes(num_layers, fp16_end, int8_end)
        if est > budget_bytes:
            logger.warning(
                f"Cannot fit KV cache in budget ({budget_bytes / 1024**2:.1f}MB). "
                f"Estimated minimum: {est / 1024**2:.1f}MB"
            )

        return fp16_end, int8_end

    def _estimate_total_bytes(
        self,
        num_layers: int,
        fp16_end: int,
        int8_end: int,
    ) -> int:
        """Estimate total KV cache bytes with given tier boundaries."""
        elements_per_layer = self._num_kv_heads * self._head_dim * self._max_seq_len
        # 2x for key + value
        kv_per_layer = elements_per_layer * 2

        total = 0.0
        for i in range(num_layers):
            if i < fp16_end:
                bpe = _FP16_BYTES
            elif i < int8_end:
                bpe = _INT8_BYTES
            else:
                bpe = _INT4_BYTES
            total += kv_per_layer * bpe

        return int(total)

    def _compute_stats(
        self,
        configs: list[LayerQuantConfig],
        max_seq_len: int,
    ) -> AdaptiveQuantStats:
        """Compute statistics from the current configuration."""
        stats = AdaptiveQuantStats()
        elements_per_layer = self._num_kv_heads * self._head_dim * max_seq_len * 2

        for cfg in configs:
            stats.per_layer_bits[cfg.layer_idx] = cfg.bits
            stats.per_layer_tier[cfg.layer_idx] = cfg.tier.name
            stats.total_bytes += int(elements_per_layer * cfg.bytes_per_element)

        stats.fp16_baseline_bytes = int(elements_per_layer * _FP16_BYTES * len(configs))
        if stats.fp16_baseline_bytes > 0:
            stats.memory_saved_pct = round(
                (1.0 - stats.total_bytes / stats.fp16_baseline_bytes) * 100, 1
            )

        # Accuracy impact heuristic: based on ratio of quantized layers
        quantized = sum(1 for c in configs if c.tier != QuantTier.FP16)
        stats.estimated_accuracy_impact = round(
            quantized / len(configs) * 0.5 if configs else 0.0, 3
        )

        return stats

    def quantize_layer(
        self,
        layer_idx: int,
        key: list,
        value: list,
    ) -> tuple[Any, dict[str, Any]]:
        """Apply layer-specific quantization to a KV cache layer.

        Args:
            layer_idx: Layer index (determines quantization tier).
            key: Key tensor data (nested list of floats).
            value: Value tensor data (nested list of floats).

        Returns:
            (packed_data, metadata) for the quantized layer.
            For FP16 layers, returns raw data as-is with metadata only.
        """
        if not self._configured:
            raise RuntimeError("AdaptiveKVQuantizer not configured — call configure() first")

        if layer_idx < 0 or layer_idx >= len(self._layer_configs):
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {len(self._layer_configs)})")

        cfg = self._layer_configs[layer_idx]

        if cfg.tier == QuantTier.FP16:
            # No quantization — return raw data with metadata
            return (key, value), {
                "layer_idx": layer_idx,
                "tier": "FP16",
                "bits": 16,
                "quantized": False,
            }

        # Use the existing KVQuantizer for INT4/INT8
        from .kv_quantization import KVQuantizer, KVQuantConfig

        qconfig = KVQuantConfig(
            bits=cfg.bits,
            group_size=cfg.group_size,
            symmetric=cfg.symmetric,
        )
        quantizer = KVQuantizer(qconfig)

        packed_key, key_meta = quantizer.quantize(key)
        packed_val, val_meta = quantizer.quantize(value)

        combined_meta = {
            "layer_idx": layer_idx,
            "tier": cfg.tier.name,
            "bits": cfg.bits,
            "quantized": True,
            "key_meta": key_meta,
            "val_meta": val_meta,
        }

        return (packed_key, packed_val), combined_meta

    def dequantize_layer(
        self,
        layer_idx: int,
        packed_kv: tuple[Any, Any],
    ) -> tuple[list, list]:
        """Dequantize a layer's KV data back to floats.

        Args:
            layer_idx: Layer index.
            packed_kv: Tuple of (packed_data_or_raw, metadata) from quantize_layer.

        Returns:
            Tuple of (key, value) as nested lists of floats.
        """
        if not self._configured:
            raise RuntimeError("AdaptiveKVQuantizer not configured — call configure() first")

        if layer_idx < 0 or layer_idx >= len(self._layer_configs):
            raise IndexError(f"layer_idx {layer_idx} out of range")

        cfg = self._layer_configs[layer_idx]

        data, meta = packed_kv

        if not meta.get("quantized", False):
            # FP16 — data is raw
            return data[0], data[1]

        # Dequantize using KVQuantizer
        from .kv_quantization import KVQuantizer

        quantizer = KVQuantizer()
        key = quantizer.dequantize(data[0], meta["key_meta"])
        value = quantizer.dequantize(data[1], meta["val_meta"])

        return key, value

    def get_stats(self) -> AdaptiveQuantStats:
        """Return quantization statistics."""
        return self._stats

    def get_layer_config(self, layer_idx: int) -> LayerQuantConfig:
        """Get the quantization config for a specific layer."""
        if not self._configured:
            raise RuntimeError("Not configured")
        return self._layer_configs[layer_idx]

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def configured(self) -> bool:
        return self._configured


# ═══════════════════════════════════════════════════════════════════════════════
# 2. KVEvictionPredictor
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BlockAccessRecord:
    """Tracks access history for a single KV block."""
    block_id: str
    # Exponential moving average of access frequency
    ema_frequency: float = 0.0
    # Exponential moving average of recency (distance from current step)
    ema_recency: float = 0.0
    # Total number of accesses observed
    total_accesses: int = 0
    # Step of last access
    last_access_step: int = 0
    # Step of first access
    first_access_step: int = 0
    # Per-request attention weight contribution
    request_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class EvictionPrediction:
    """Prediction result for a block."""
    block_id: str
    predicted_access_prob: float
    should_evict: bool
    confidence: float  # 0.0 to 1.0


class KVEvictionPredictor:
    """Predicts which KV blocks will be needed in the near future.

    Uses attention pattern analysis from recent steps to predict access
    probabilities. Blocks with high predicted access probability are
    retained; blocks with low probability are evicted first.

    The predictor maintains exponential moving averages of:
    - Access frequency: how often a block is accessed
    - Recency: how recently a block was last accessed
    - Attention weight: how much attention weight a block receives

    This is more intelligent than LRU/LFU because:
    - A block accessed frequently in the past but not recently (LFU would keep)
      is predicted based on its attention pattern
    - A block accessed recently but with very low attention weight (LRU would keep)
      is predicted to have low future access probability
    """

    def __init__(
        self,
        ema_decay: float = _EMA_DECAY,
        eviction_threshold: float = 0.1,
        min_observations: int = _MIN_OBSERVATIONS,
    ) -> None:
        self._ema_decay = ema_decay
        self._eviction_threshold = eviction_threshold
        self._min_observations = min_observations

        # Per-block access records
        self._records: dict[str, BlockAccessRecord] = {}

        # Current step counter
        self._step: int = 0

        # Per-request last accessed blocks
        self._request_blocks: dict[str, set[str]] = {}

        # Prediction stats
        self._total_predictions: int = 0
        self._total_evictions_recommended: int = 0
        self._correct_predictions: int = 0

    def train_step(
        self,
        attention_weights: dict[str, dict[str, float]] | None = None,
        accessed_blocks: dict[str, list[str]] | None = None,
    ) -> None:
        """Learn access patterns from a single step.

        Updates EMA frequency and recency for all observed blocks.

        Args:
            attention_weights: Optional mapping of
                request_id → {block_id: weight} from attention analysis.
            accessed_blocks: Mapping of request_id → [block_ids] that
                were accessed during this step.
        """
        self._step += 1

        # Process accessed blocks
        if accessed_blocks:
            for request_id, block_ids in accessed_blocks.items():
                # Track per-request blocks
                if request_id not in self._request_blocks:
                    self._request_blocks[request_id] = set()
                self._request_blocks[request_id].update(block_ids)

                for block_id in block_ids:
                    record = self._get_or_create_record(block_id)

                    # Update EMA frequency
                    record.ema_frequency = (
                        self._ema_decay * record.ema_frequency + (1 - self._ema_decay) * 1.0
                    )

                    # Update EMA recency (lower = more recent)
                    record.ema_recency = (
                        self._ema_decay * record.ema_recency
                        + (1 - self._ema_decay) * (self._step - record.last_access_step)
                    )

                    record.last_access_step = self._step
                    record.total_accesses += 1
                    if record.first_access_step == 0:
                        record.first_access_step = self._step

        # Process attention weights
        if attention_weights:
            for request_id, weights in attention_weights.items():
                for block_id, weight in weights.items():
                    record = self._get_or_create_record(block_id)
                    # Update per-request weight (keep the max for this request)
                    old_weight = record.request_weights.get(request_id, 0.0)
                    record.request_weights[request_id] = max(old_weight, weight)

        # Decay EMA frequency for all blocks (not just accessed ones)
        for block_id, record in list(self._records.items()):
            if accessed_blocks is None or block_id not in {
                bid for bids in accessed_blocks.values() for bid in bids
            }:
                record.ema_frequency = self._ema_decay * record.ema_frequency

    def predict_next_access(
        self,
        request_id: str,
        current_blocks: list[str],
    ) -> list[EvictionPrediction]:
        """Predict next access probability for each block.

        Returns blocks ranked by predicted access probability (highest first).

        Args:
            request_id: The requesting request ID (for per-request weighting).
            current_blocks: Block IDs to evaluate.

        Returns:
            List of EvictionPrediction, sorted by predicted_access_prob descending.
        """
        predictions = []
        for block_id in current_blocks:
            record = self._records.get(block_id)
            if record is None:
                # Unknown block — low prediction confidence
                predictions.append(EvictionPrediction(
                    block_id=block_id,
                    predicted_access_prob=0.0,
                    should_evict=True,
                    confidence=0.0,
                ))
                continue

            # Compute predicted access probability from EMA signals
            prob = self._compute_access_probability(record, request_id)

            # Confidence based on number of observations
            confidence = min(1.0, record.total_accesses / (self._min_observations * 2))

            should_evict = (
                prob < self._eviction_threshold
                and record.total_accesses >= self._min_observations
            )

            self._total_predictions += 1
            if should_evict:
                self._total_evictions_recommended += 1

            predictions.append(EvictionPrediction(
                block_id=block_id,
                predicted_access_prob=prob,
                should_evict=should_evict,
                confidence=confidence,
            ))

        # Sort by predicted access probability (highest first)
        predictions.sort(key=lambda p: p.predicted_access_prob, reverse=True)
        return predictions

    def should_evict(self, block_id: str) -> bool:
        """Prediction-based eviction decision for a single block.

        Returns True if the predictor recommends evicting this block.
        """
        record = self._records.get(block_id)
        if record is None:
            return True  # Unknown block — safe to evict

        if record.total_accesses < self._min_observations:
            return False  # Not enough data — keep it

        prob = self._compute_access_probability(record, None)
        return prob < self._eviction_threshold

    def _compute_access_probability(
        self,
        record: BlockAccessRecord,
        request_id: str | None,
    ) -> float:
        """Compute predicted access probability for a block.

        Combines three signals:
        1. EMA frequency (how often accessed)
        2. EMA recency (how recently accessed)
        3. Attention weight contribution (how important)
        """
        # Frequency signal: normalize to [0, 1]
        freq_signal = min(1.0, record.ema_frequency * 2.0)

        # Recency signal: blocks accessed recently have higher probability
        if self._step > 0:
            recency_raw = record.ema_recency
            recency_signal = 1.0 / (1.0 + recency_raw)
        else:
            recency_signal = 0.0

        # Attention weight signal: blocks receiving high attention are important
        weight_signal = 0.0
        if record.request_weights:
            if request_id and request_id in record.request_weights:
                weight_signal = min(1.0, record.request_weights[request_id] * 2.0)
            else:
                max_weight = max(record.request_weights.values())
                weight_signal = min(1.0, max_weight * 2.0)

        # Weighted combination
        prob = 0.4 * freq_signal + 0.35 * recency_signal + 0.25 * weight_signal
        return min(1.0, max(0.0, prob))

    def _get_or_create_record(self, block_id: str) -> BlockAccessRecord:
        """Get or create an access record for a block."""
        if block_id not in self._records:
            self._records[block_id] = BlockAccessRecord(block_id=block_id)
        return self._records[block_id]

    def record_eviction_outcome(
        self,
        block_id: str,
        was_needed: bool,
    ) -> None:
        """Record whether an evicted block was actually needed.

        Used for self-calibration of the eviction threshold.
        """
        self._total_predictions += 1
        if not was_needed:
            self._correct_predictions += 1
        else:
            # Evicted a block that was needed — threshold may be too aggressive
            # Nudge threshold up slightly
            self._eviction_threshold = min(
                0.5, self._eviction_threshold * 1.05
            )

    def get_stats(self) -> dict:
        """Return predictor statistics."""
        total = self._total_predictions
        return {
            "step": self._step,
            "tracked_blocks": len(self._records),
            "tracked_requests": len(self._request_blocks),
            "total_predictions": total,
            "total_evictions_recommended": self._total_evictions_recommended,
            "eviction_threshold": round(self._eviction_threshold, 4),
            "correct_predictions": self._correct_predictions,
            "prediction_accuracy": round(
                self._correct_predictions / total * 100, 1
            ) if total > 0 else 0.0,
        }

    def reset(self) -> None:
        """Reset all predictor state."""
        self._records.clear()
        self._request_blocks.clear()
        self._step = 0
        self._total_predictions = 0
        self._total_evictions_recommended = 0
        self._correct_predictions = 0


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ChunkedPrefillOptimizer
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ChunkInfo:
    """Information about a computed chunk."""
    start_token: int
    end_token: int
    num_tokens: int
    importance: float = 1.0
    request_id: str = ""
    chunk_index: int = 0


class ChunkedPrefillOptimizer:
    """Optimizes chunk boundaries for chunked prefill.

    Instead of splitting prompts at arbitrary token boundaries, this optimizer:
    1. Identifies semantically meaningful split points (sentence ends,
       paragraph breaks, etc.)
    2. Avoids splitting mid-token or at positions that break semantic units
    3. Prioritizes chunks by importance (system prompts first, then context)
    4. Fairly interleaves chunks from different requests

    Semantic boundary detection (without requiring a tokenizer for most cases):
    - Uses token importance heuristics: first chunk gets highest importance
      (system prompt), importance decreases for later chunks
    - Respects sentence boundaries when detected
    - Falls back to fixed-size chunks when no boundaries are detected
    """

    def __init__(
        self,
        sentence_end_tokens: set[int] | None = None,
    ) -> None:
        self._sentence_end_tokens = sentence_end_tokens or set()

    def compute_optimal_chunks(
        self,
        tokens: list[int],
        max_chunk_tokens: int,
        tokenizer: Any = None,
    ) -> list[ChunkInfo]:
        """Compute optimal chunk boundaries for a token sequence.

        Splits at semantic boundaries when possible, falls back to
        fixed-size chunks otherwise. Importance is assigned based on
        position (earlier chunks are more important for system prompts).

        Args:
            tokens: Full token ID sequence for the prompt.
            max_chunk_tokens: Maximum tokens per chunk.
            tokenizer: Optional tokenizer for semantic boundary detection.

        Returns:
            List of ChunkInfo describing each chunk.
        """
        if not tokens:
            return []

        if max_chunk_tokens <= 0:
            raise ValueError("max_chunk_tokens must be > 0")

        total = len(tokens)
        if total <= max_chunk_tokens:
            return [ChunkInfo(
                start_token=0,
                end_token=total,
                num_tokens=total,
                importance=1.0,
                chunk_index=0,
            )]

        # Find semantic split points
        split_points = self._find_split_points(tokens, max_chunk_tokens, tokenizer)

        # Build chunks from split points
        chunks = []
        prev = 0
        for idx, split in enumerate(split_points):
            if split > prev:
                # Importance decays with chunk index
                importance = max(0.1, 1.0 - idx * 0.15)
                chunks.append(ChunkInfo(
                    start_token=prev,
                    end_token=split,
                    num_tokens=split - prev,
                    importance=importance,
                    chunk_index=idx,
                ))
                prev = split

        # Handle remaining tokens
        if prev < total:
            importance = max(0.1, 1.0 - len(chunks) * 0.15)
            chunks.append(ChunkInfo(
                start_token=prev,
                end_token=total,
                num_tokens=total - prev,
                importance=importance,
                chunk_index=len(chunks),
            ))

        return chunks

    def _find_split_points(
        self,
        tokens: list[int],
        max_chunk_tokens: int,
        tokenizer: Any = None,
    ) -> list[int]:
        """Find semantic split points in the token sequence.

        Strategy:
        1. Walk through tokens in max_chunk_tokens-sized windows
        2. Within each window, search backward for a sentence boundary
        3. If no sentence boundary found, split at the window boundary

        Args:
            tokens: Token IDs.
            max_chunk_tokens: Maximum tokens per chunk.
            tokenizer: Optional tokenizer for boundary detection.

        Returns:
            List of token indices where chunks should end.
        """
        split_points: list[int] = []
        pos = 0

        while pos < len(tokens):
            # Target end of this chunk
            target_end = min(pos + max_chunk_tokens, len(tokens))

            if target_end >= len(tokens):
                # Last chunk — take everything remaining
                break

            # Search backward from target_end for a semantic boundary
            best_split = target_end
            search_start = max(pos + max_chunk_tokens // 2, pos + 1)

            for i in range(target_end, search_start - 1, -1):
                if self._is_sentence_boundary(tokens, i, tokenizer):
                    best_split = i
                    break

            split_points.append(best_split)
            pos = best_split

        return split_points

    def _is_sentence_boundary(
        self,
        tokens: list[int],
        idx: int,
        tokenizer: Any = None,
    ) -> bool:
        """Check if position idx is a sentence boundary.

        Uses:
        1. Known sentence-end token IDs (if configured)
        2. Tokenizer decode (if available) to check for '.', '!', '?', etc.
        3. Falls back to checking common tokenizer patterns
        """
        if idx <= 0 or idx >= len(tokens):
            return False

        # Check known sentence-end tokens
        if tokens[idx - 1] in self._sentence_end_tokens:
            return True

        # If tokenizer available, decode surrounding tokens to detect boundaries
        if tokenizer is not None:
            try:
                # Decode a small window around the position
                window_start = max(0, idx - 3)
                window_end = min(len(tokens), idx + 3)
                text = tokenizer.decode(tokens[window_start:window_end])

                # Check for paragraph/sentence end markers
                # Look for '. ', '.\n', '!\n', '?\n', '\n\n'
                for marker in ['. ', '.\n', '!\n', '?\n', '\n\n', '。\n', '。 ']:
                    if marker in text:
                        return True
            except Exception:
                logger.debug("operation failed", exc_info=True)
                pass

        return False

    def interleave_chunks(
        self,
        requests_chunks: dict[str, list[ChunkInfo]],
    ) -> list[ChunkInfo]:
        """Produce a fair interleaving schedule for multiple requests.

        Round-robin across requests, taking one chunk from each per round.
        Within a request, chunks are ordered by importance (high → low).

        This prevents a single long request from starving others during
        chunked prefill, ensuring fair prefill progress across all requests.

        Args:
            requests_chunks: Mapping of request_id → list of ChunkInfo.

        Returns:
            Flattened list of ChunkInfo in execution order, with request_id set.
        """
        if not requests_chunks:
            return []

        # Assign request_id to each chunk and sort by importance within each request
        request_queues: dict[str, list[ChunkInfo]] = {}
        for req_id, chunks in requests_chunks.items():
            sorted_chunks = sorted(chunks, key=lambda c: c.importance, reverse=True)
            for chunk in sorted_chunks:
                chunk.request_id = req_id
            request_queues[req_id] = list(sorted_chunks)

        # Round-robin interleaving
        schedule: list[ChunkInfo] = []
        active_ids = list(request_queues.keys())

        while active_ids:
            next_active = []
            for req_id in active_ids:
                queue = request_queues.get(req_id, [])
                if queue:
                    chunk = queue.pop(0)
                    schedule.append(chunk)
                    if queue:
                        next_active.append(req_id)
            active_ids = next_active

        return schedule


# ═══════════════════════════════════════════════════════════════════════════════
# 4. KVBlockCompactor
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class KVBlock:
    """Represents a single KV cache block."""
    block_id: str
    request_id: str
    tokens: list[int] = field(default_factory=list)
    # Number of valid slots in this block (0 to block_size)
    num_valid: int = 0
    # Total capacity
    block_size: int = 64
    # Whether this block is actively in use
    is_active: bool = True

    @property
    def utilization(self) -> float:
        """Fraction of block capacity used."""
        if self.block_size <= 0:
            return 0.0
        return self.num_valid / self.block_size

    @property
    def is_partial(self) -> bool:
        """Whether this block is partially filled."""
        return self.num_valid < self.block_size


@dataclass
class CompactionResult:
    """Result of a compaction pass."""
    blocks_before: int = 0
    blocks_after: int = 0
    blocks_freed: int = 0
    partial_blocks_merged: int = 0
    utilization_before: float = 0.0
    utilization_after: float = 0.0


@dataclass
class FragmentationStats:
    """Statistics about block fragmentation."""
    total_blocks: int = 0
    active_blocks: int = 0
    free_blocks: int = 0
    partial_blocks: int = 0
    full_blocks: int = 0
    total_valid_slots: int = 0
    total_capacity: int = 0
    overall_utilization: float = 0.0
    waste_pct: float = 0.0
    fragmentation_ratio: float = 0.0


class KVBlockCompactor:
    """Periodically compacts KV cache by merging partially-used blocks.

    After preemption or eviction, KV cache blocks may be partially filled
    (a block with capacity 64 but only 10 valid tokens). This wastes memory
    and reduces effective cache capacity.

    The compactor merges adjacent partial blocks belonging to the same request
    to free complete blocks. For example:
    - Block A (10/64 valid) + Block B (20/64 valid) → Block A (30/64 valid) + freed block

    Compaction runs periodically (every N steps) as a background optimization.
    It only merges blocks from the same request to maintain correctness.
    """

    def __init__(
        self,
        block_size: int = 64,
        compact_interval: int = 50,
        min_utilization_threshold: float = 0.5,
    ) -> None:
        """Initialize the compactor.

        Args:
            block_size: Number of tokens per KV block.
            compact_interval: Run compaction every N steps.
            min_utilization_threshold: Only compact blocks below this utilization.
        """
        self._block_size = block_size
        self._compact_interval = compact_interval
        self._min_utilization = min_utilization_threshold

        # Internal block storage (simulated — real system uses KVCacheManager)
        self._blocks: dict[str, KVBlock] = {}

        # Step counter
        self._step: int = 0

        # Compaction stats
        self._total_compactions: int = 0
        self._total_blocks_freed: int = 0

    @property
    def block_size(self) -> int:
        return self._block_size

    def add_block(self, block: KVBlock) -> None:
        """Add a block to the compactor's tracking."""
        self._blocks[block.block_id] = block

    def remove_block(self, block_id: str) -> KVBlock | None:
        """Remove and return a block."""
        return self._blocks.pop(block_id, None)

    def compact(
        self,
        cache_blocks: dict[str, KVBlock] | None = None,
        block_size: int | None = None,
    ) -> CompactionResult:
        """Merge partial blocks to free complete blocks.

        Strategy:
        1. Group blocks by request_id
        2. Within each group, find adjacent partial blocks
        3. Merge tokens from later blocks into earlier blocks
        4. Free blocks that become empty after merging

        Args:
            cache_blocks: Optional block dict to compact (uses internal if None).
            block_size: Override block size for this compaction pass.

        Returns:
            CompactionResult with before/after statistics.
        """
        blocks = cache_blocks if cache_blocks is not None else self._blocks
        bs = block_size or self._block_size

        if not blocks:
            return CompactionResult()

        # Compute before stats
        stats_before = self._compute_fragmentation(blocks, bs)

        # Group blocks by request
        request_blocks: dict[str, list[KVBlock]] = {}
        for block in blocks.values():
            if not block.is_active or not block.is_partial:
                continue
            if block.request_id not in request_blocks:
                request_blocks[block.request_id] = []
            request_blocks[block.request_id].append(block)

        blocks_merged = 0
        blocks_freed = 0
        blocks_to_remove: list[str] = []

        for req_id, partial_blocks in request_blocks.items():
            if len(partial_blocks) < 2:
                continue

            # Sort by block_id to get deterministic ordering
            partial_blocks.sort(key=lambda b: b.block_id)

            # Merge strategy: move tokens from later blocks to fill earlier ones
            # Walk through pairs and merge
            merged_in_group = 0
            i = 0
            while i < len(partial_blocks):
                current = partial_blocks[i]
                available = bs - current.num_valid

                # Try to fill current block from subsequent blocks
                j = i + 1
                while j < len(partial_blocks) and available > 0:
                    donor = partial_blocks[j]
                    can_take = min(available, donor.num_valid)

                    if can_take > 0:
                        # Transfer tokens
                        current.tokens.extend(donor.tokens[:can_take])
                        donor.tokens = donor.tokens[can_take:]
                        donor.num_valid -= can_take
                        current.num_valid += can_take
                        available -= can_take

                    if donor.num_valid <= 0:
                        # Donor is empty — mark for removal
                        donor.is_active = False
                        blocks_to_remove.append(donor.block_id)
                        blocks_freed += 1
                        merged_in_group += 1
                        j += 1
                    else:
                        # Donor still has tokens
                        break

                i += 1

            blocks_merged += merged_in_group

        # Remove freed blocks
        for block_id in blocks_to_remove:
            if block_id in blocks:
                del blocks[block_id]

        # Compute after stats
        stats_after = self._compute_fragmentation(blocks, bs)

        # Update internal counters
        self._total_compactions += 1
        self._total_blocks_freed += blocks_freed

        return CompactionResult(
            blocks_before=stats_before.total_blocks,
            blocks_after=stats_after.total_blocks,
            blocks_freed=blocks_freed,
            partial_blocks_merged=blocks_merged,
            utilization_before=stats_before.overall_utilization,
            utilization_after=stats_after.overall_utilization,
        )

    def _compute_fragmentation(
        self,
        blocks: dict[str, KVBlock],
        block_size: int,
    ) -> FragmentationStats:
        """Compute fragmentation statistics for the current block set."""
        total_blocks = len(blocks)
        active_blocks = sum(1 for b in blocks.values() if b.is_active)
        partial_blocks = sum(1 for b in blocks.values() if b.is_active and b.is_partial)
        full_blocks = sum(1 for b in blocks.values() if b.is_active and not b.is_partial)
        total_valid = sum(b.num_valid for b in blocks.values() if b.is_active)
        total_capacity = active_blocks * block_size

        overall_util = total_valid / total_capacity if total_capacity > 0 else 0.0
        waste = 1.0 - overall_util if total_capacity > 0 else 0.0
        frag_ratio = partial_blocks / active_blocks if active_blocks > 0 else 0.0

        return FragmentationStats(
            total_blocks=total_blocks,
            active_blocks=active_blocks,
            free_blocks=total_blocks - active_blocks,
            partial_blocks=partial_blocks,
            full_blocks=full_blocks,
            total_valid_slots=total_valid,
            total_capacity=total_capacity,
            overall_utilization=round(overall_util, 4),
            waste_pct=round(waste * 100, 1),
            fragmentation_ratio=round(frag_ratio, 4),
        )

    def get_fragmentation_stats(
        self,
        cache_blocks: dict[str, KVBlock] | None = None,
    ) -> FragmentationStats:
        """Measure block utilization and waste.

        Args:
            cache_blocks: Optional block dict (uses internal if None).

        Returns:
            FragmentationStats with utilization and waste metrics.
        """
        blocks = cache_blocks if cache_blocks is not None else self._blocks
        return self._compute_fragmentation(blocks, self._block_size)

    def maybe_compact(self, step: int) -> CompactionResult | None:
        """Run compaction if the step interval has been reached.

        Called from the scheduler step loop. Returns None if compaction
        was not triggered, otherwise returns the CompactionResult.

        Args:
            step: Current scheduler step counter.
        """
        if step <= 0 or step % self._compact_interval != 0:
            return None

        result = self.compact()
        if result.blocks_freed > 0:
            logger.info(
                f"KV block compaction (step {step}): "
                f"freed {result.blocks_freed} blocks, "
                f"util {result.utilization_before:.1%} → {result.utilization_after:.1%}"
            )
        return result

    def get_stats(self) -> dict:
        """Return compactor statistics."""
        frag = self.get_fragmentation_stats()
        return {
            "block_size": self._block_size,
            "compact_interval": self._compact_interval,
            "total_compactions": self._total_compactions,
            "total_blocks_freed": self._total_blocks_freed,
            "fragmentation": {
                "total_blocks": frag.total_blocks,
                "active_blocks": frag.active_blocks,
                "partial_blocks": frag.partial_blocks,
                "full_blocks": frag.full_blocks,
                "utilization": frag.overall_utilization,
                "waste_pct": frag.waste_pct,
                "fragmentation_ratio": frag.fragmentation_ratio,
            },
        }

    @property
    def blocks(self) -> dict[str, KVBlock]:
        """Access internal block storage."""
        return self._blocks
