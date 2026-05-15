from __future__ import annotations
"""TeaCache — Timestep Embedding Aware Cache for diffusion acceleration.

TeaCache accelerates diffusion inference by reusing transformer block
computations when consecutive timestep embeddings are similar. When the
accumulated relative L1 distance between modulated inputs falls below a
threshold, the cached residual from the previous step is reused instead of
running the full transformer.

Reference:
  - vllm-omni: vllm_omni/diffusion/cache/teacache/
  - Paper: "TeaCache: Timestep Embedding Aware Cache for Tuning-Free
    Acceleration of DiT-based Image and Video Generation"

Architecture:
  - TeaCacheConfig: Threshold + polynomial coefficients per model
  - TeaCacheState: Per-run counter, accumulated distance, cached residual
  - TeaCacheHook: Intercepts transformer forward, decides cache vs compute

Integration with Z-Image:
  The hook wraps the ZImageTransformer forward pass. At each denoising step:
  1. Extract modulated input from the first transformer block (t_emb → adaLN)
  2. Compare with previous step's modulated input (relative L1 distance)
  3. Apply polynomial rescaling (model-specific coefficients)
  4. If accumulated distance < threshold: reuse cached residual (fast path)
  5. If accumulated distance >= threshold: run full transformer, cache residual
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# Model-specific polynomial coefficients for rescaling L1 distances.
# Source: vllm-omni TeaCache config.py
_MODEL_COEFFICIENTS = {
    "ZImageTransformer": [
        -4.50000000e02,
        2.80000000e02,
        -4.50000000e01,
        3.20000000e00,
        -2.00000000e-02,
    ],
    "QwenImageTransformer2DModel": [
        -4.50000000e02,
        2.80000000e02,
        -4.50000000e01,
        3.20000000e00,
        -2.00000000e-02,
    ],
    "FluxTransformer2DModel": [
        4.98651651e02,
        -2.83781631e02,
        5.58554382e01,
        -3.82021401e00,
        2.64230861e-01,
    ],
}


@dataclass
class TeaCacheConfig:
    """Configuration for TeaCache diffusion acceleration.

    Args:
        rel_l1_thresh: Threshold for accumulated relative L1 distance.
            Below this → reuse cached residual. Typical values:
            - 0.2: ~1.5x speedup, minimal quality loss
            - 0.4: ~1.8x speedup, slight quality loss
            - 0.6: ~2.0x speedup, noticeable quality loss
        coefficients: Polynomial coefficients for rescaling L1 distance.
            If None, auto-selected based on model_type.
        model_type: Transformer class name for coefficient lookup.
    """
    rel_l1_thresh: float = 0.2
    coefficients: list[float] | None = None
    model_type: str = "ZImageTransformer"

    def __post_init__(self):
        if self.rel_l1_thresh <= 0:
            raise ValueError(f"rel_l1_thresh must be positive, got {self.rel_l1_thresh}")
        if self.coefficients is None:
            if self.model_type in _MODEL_COEFFICIENTS:
                self.coefficients = _MODEL_COEFFICIENTS[self.model_type]
            else:
                self.coefficients = _MODEL_COEFFICIENTS["ZImageTransformer"]
        if len(self.coefficients) != 5:
            raise ValueError(f"coefficients must have 5 elements, got {len(self.coefficients)}")


class TeaCacheState:
    """Per-run state for TeaCache caching across denoising steps."""

    def __init__(self):
        self.cnt: int = 0
        self.accumulated_rel_l1_distance: float = 0.0
        self.previous_modulated_input = None  # mx.array or None
        self.previous_residual = None  # mx.array or None

    def reset(self):
        self.cnt = 0
        self.accumulated_rel_l1_distance = 0.0
        self.previous_modulated_input = None
        self.previous_residual = None


class TeaCacheHook:
    """TeaCache hook that wraps a transformer forward pass with caching.

    Usage:
        config = TeaCacheConfig(rel_l1_thresh=0.2)
        hook = TeaCacheHook(config)
        hook.reset()

        for t in range(num_steps):
            output = hook.forward(transformer, x, timestep, sigmas, cap_feats)
    """

    def __init__(self, config: TeaCacheConfig):
        self.config = config
        self.rescale_func = np.poly1d(config.coefficients)
        self.state = TeaCacheState()
        self._cache_hits = 0
        self._cache_misses = 0

    def reset(self):
        self.state.reset()
        self._cache_hits = 0
        self._cache_misses = 0

    def should_compute(self, modulated_input) -> bool:
        """Determine whether to run full transformer or reuse cache.

        Args:
            modulated_input: Current timestep's modulated input (mx.array).

        Returns:
            True → compute full transformer, False → reuse cached residual.
        """
        import mlx.core as mx

        # First step: always compute
        if self.state.cnt == 0:
            self.state.accumulated_rel_l1_distance = 0.0
            return True

        if self.state.previous_modulated_input is None:
            return True

        # Compute relative L1 distance
        prev = self.state.previous_modulated_input
        diff = mx.abs(modulated_input - prev).mean()
        ref = mx.abs(prev).mean() + 1e-8
        rel_distance = float((diff / ref).item())

        # Apply polynomial rescaling
        rescaled = float(self.rescale_func(rel_distance))
        self.state.accumulated_rel_l1_distance += abs(rescaled)

        # Decision
        if self.state.accumulated_rel_l1_distance < self.config.rel_l1_thresh:
            return False  # Cache hit
        else:
            self.state.accumulated_rel_l1_distance = 0.0
            return True  # Cache miss, recompute

    def forward(self, transformer, x, timestep, sigmas, cap_feats):
        """Run transformer forward with TeaCache caching.

        Args:
            transformer: The diffusion transformer (e.g., ZImageTransformer).
            x: Current noise latents.
            timestep: Current timestep.
            sigmas: Sigma schedule.
            cap_feats: Text conditioning features.

        Returns:
            Transformer output (noise prediction).
        """
        import mlx.core as mx

        # Extract modulated input for cache decision
        # Use the timestep embedding as the modulated input proxy
        if not isinstance(timestep, mx.array):
            if isinstance(timestep, int):
                sigma_t = sigmas[timestep].reshape((1,))
                timestep = mx.ones_like(sigma_t) - sigma_t
            else:
                timestep = mx.array(timestep, dtype=mx.float32)
        if timestep.ndim == 0:
            timestep = timestep.reshape((1,))

        t_emb = transformer.t_embedder(timestep.astype(mx.float32) * transformer.t_scale)
        modulated_input = t_emb

        should_run = self.should_compute(modulated_input)

        if not should_run and self.state.previous_residual is not None:
            # Fast path: reuse cached residual
            self._cache_hits += 1
            return self.state.previous_residual
        else:
            # Slow path: full transformer computation
            self._cache_misses += 1
            output = transformer(
                x=x,
                timestep=timestep,
                sigmas=sigmas,
                cap_feats=cap_feats,
            )
            mx.eval(output)

            # Cache residual for next step
            self.state.previous_residual = output
            self.state.previous_modulated_input = modulated_input
            mx.eval(self.state.previous_modulated_input)

        self.state.cnt += 1
        return output

    def get_stats(self) -> dict:
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / max(total, 1)
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "accumulated_distance": self.state.accumulated_rel_l1_distance,
        }
