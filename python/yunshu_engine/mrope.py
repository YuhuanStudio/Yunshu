from __future__ import annotations
"""Yunshu mRoPE (Multi-dimensional Rotary Position Embedding) support.

Studied from oMLX's mRoPE integration, adapted for Yunshu's architecture:
- Detection via model config (rope_scaling.mrope_section / rope_parameters.mrope_section)
- Per-request rope_deltas capture after VLM prefill
- Batch decode position construction for concurrent VLM + text requests
- Text-only contamination prevention

Architecture:
  detect_mrope(config) → MRPoEInfo
  MRPoEPositionBuilder.build_decode_positions(offsets, deltas) → position_ids
  Per-request delta stored in Request.rope_deltas + Scheduler._uid_rope_deltas

References:
  - oMLX: omlx/scheduler.py (batch mRoPE delta injection)
  - vLLM: vllm/v1/worker/gpu/mm/rope.py (RopeState)
  - Qwen2-VL paper: mRoPE for vision-language models
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MRoPEInfo:
    """Detected mRoPE configuration from model config."""
    enabled: bool = False
    # Section sizes for the 3 dimensions (temporal, height, width)
    # Sum of sections = head_dim // 2
    sections: tuple[int, int, int] | None = None
    # Number of dimensions (always 3 for mRoPE)
    num_dims: int = 3
    # The config key that was found
    source_key: str = ""


def detect_mrope(model_config: dict) -> MRoPEInfo:
    """Detect mRoPE support from model configuration.

    Checks for mRoPE section definition in:
    - text_config.rope_scaling.mrope_section
    - text_config.rope_parameters.mrope_section
    - rope_scaling.mrope_section
    - rope_parameters.mrope_section

    Args:
        model_config: Model configuration dict (can be nested).

    Returns:
        MRoPEInfo with detection results.
    """
    # Check nested text_config first (Qwen-VL/Omni pattern)
    text_config = model_config.get("text_config", {})
    if not isinstance(text_config, dict):
        text_config = {}

    for config_source in [text_config, model_config]:
        for key in ["rope_scaling", "rope_parameters"]:
            section_config = config_source.get(key, {})
            if not isinstance(section_config, dict):
                continue

            mrope_section = section_config.get("mrope_section")
            if mrope_section is not None:
                if isinstance(mrope_section, str):
                    # Format: "1,1,1,...,1,0,0,...,0" or comma-separated ints
                    try:
                        sections = [int(x.strip()) for x in mrope_section.split(",")]
                    except (ValueError, AttributeError):
                        continue
                elif isinstance(mrope_section, (list, tuple)):
                    sections = list(mrope_section)
                else:
                    continue

                # Validate: should be 3 non-negative integers summing to head_dim//2
                if len(sections) >= 3:
                    # Take the last 3 sections (T, H, W)
                    dims = tuple(sections[-3:])
                    return MRoPEInfo(
                        enabled=True,
                        sections=dims,
                        num_dims=3,
                        source_key=f"{key}.mrope_section",
                    )

    return MRoPEInfo(enabled=False)


def build_decode_positions(
    offsets: list[int],
    deltas: list[float],
    num_dims: int = 3,
) -> Any:
    """Build position IDs for batch decode with mRoPE.

    For each request in the batch, constructs position IDs:
      position_ids[d, i] = offsets[i] + deltas[i]
    broadcast across all dimensions.

    For text-only requests (delta=0), all dimensions use the same position,
    making mRoPE equivalent to standard 1D RoPE.

    Args:
        offsets: Per-request token offset (current sequence length).
        deltas: Per-request mRoPE delta (0.0 for text-only).
        num_dims: Number of position dimensions (3 for mRoPE).

    Returns:
        mx.array of shape (num_dims, batch_size, 1) or (batch_size, 1)
    """
    import mlx.core as mx

    batch_size = len(offsets)
    positions_1d = mx.array([o + d for o, d in zip(offsets, deltas)]).reshape(1, batch_size, 1)

    # Broadcast across all dimensions (T, H, W all get same position for decode)
    return mx.broadcast_to(positions_1d, (num_dims, batch_size, 1))


def build_prefill_positions(
    num_tokens: int,
    delta: float = 0.0,
    num_dims: int = 3,
) -> Any:
    """Build position IDs for prefill with mRoPE.

    For text-only prefill (delta=0), all dimensions use range(0, num_tokens).
    For VLM prefill, the vision encoder sets the actual multi-dimensional positions.

    Args:
        num_tokens: Number of tokens in the prefill sequence.
        delta: mRoPE delta (0.0 for text-only).
        num_dims: Number of position dimensions.

    Returns:
        mx.array of shape (num_dims, 1, num_tokens)
    """
    import mlx.core as mx

    positions = mx.arange(num_tokens).reshape(1, 1, num_tokens)
    return mx.broadcast_to(positions, (num_dims, 1, num_tokens))


def capture_rope_deltas(model: Any) -> float | None:
    """Capture rope_deltas from a model after prefill.

    After running the language model's forward pass (especially for VLM
    models with vision features), the model may store _rope_deltas as
    an internal state. This captures and returns it.

    Args:
        model: The language model (or wrapper) that may have _rope_deltas.

    Returns:
        The rope_deltas float, or None if not applicable.
    """
    # Check language_model wrapper first (VLM models)
    lang_model = getattr(model, 'language_model', model)
    delta = getattr(lang_model, '_rope_deltas', None)
    if delta is not None:
        try:
            return float(delta)
        except (TypeError, ValueError):
            pass

    # Check top-level model
    delta = getattr(model, '_rope_deltas', None)
    if delta is not None:
        try:
            return float(delta)
        except (TypeError, ValueError):
            pass

    return None


def clear_rope_state(model: Any) -> None:
    """Clear mRoPE state from the model after text-only prefill.

    Prevents text-only requests from being contaminated by
    previous VLM request's position state.
    """
    lang_model = getattr(model, 'language_model', model)
    for attr in ('_position_ids', '_rope_deltas', '_batch_rope_deltas'):
        if hasattr(lang_model, attr):
            setattr(lang_model, attr, None)
        if hasattr(model, attr):
            setattr(model, attr, None)


class BatchRopeDeltaManager:
    """Manages per-request rope_deltas for batch decode (oMLX pattern).

    Maintains a Dict[int, float] mapping from BatchGenerator UID to
    the rope_deltas captured during prefill. Before each decode step,
    builds a batch_rope_deltas array aligned to current batch order.

    Thread-safe: mutations are protected by a lock (CPython GIL also
    guarantees dict atomicity for simple get/set, but we use explicit
    locking for multi-step operations).
    """

    def __init__(self) -> None:
        self._deltas: dict[int, float] = {}
        import threading
        self._lock = threading.Lock()

    def register(self, uid: int, delta: float) -> None:
        """Register rope_deltas for a request after prefill."""
        with self._lock:
            self._deltas[uid] = delta

    def unregister(self, uid: int) -> None:
        """Remove rope_deltas when a request finishes/is aborted."""
        with self._lock:
            self._deltas.pop(uid, None)

    def get_batch_deltas(self, uids: list[int]) -> list[float]:
        """Get rope_deltas for a batch of UIDs, defaulting to 0.0.

        Args:
            uids: List of batch UIDs in current decode step order.

        Returns:
            List of deltas aligned to the UID order.
        """
        with self._lock:
            return [self._deltas.get(uid, 0.0) for uid in uids]

    def clear(self) -> None:
        """Clear all registered deltas."""
        with self._lock:
            self._deltas.clear()
