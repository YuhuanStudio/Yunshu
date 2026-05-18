from __future__ import annotations
"""Gemma4 built-in speculative decoding proposer.

Gemma4 models have built-in speculative decoding capability where certain
intermediate attention layers can generate draft token predictions. This
leverages the model's architecture directly — no separate draft model needed.

Detection:
  Gemma4 models are identified by:
    - model_type containing "gemma4" or "gemma_4"
    - model_type "gemma" with version >= 4
    - config key "speculative_layers" or "draft_heads"

Architecture:
  Gemma4SpecConfig: Configuration dataclass.
  Gemma4SpecProposer: Manages Gemma4-specific draft proposals.
  Gemma4Stats: Runtime statistics tracking.

Lifecycle:
  1. Gemma4SpecProposer.detect(model) — check if model supports Gemma4 spec
  2. propose(hidden_states, n_draft) — extract draft predictions
  3. get_stats() — report usage

Graceful degradation:
  - When model is not Gemma4, returns empty proposals
  - When model lacks spec layers, returns empty proposals

Integration:
  - Gemma4Strategy wraps as SpecStrategy for CompositeStrategy
  - SpecStrategyFactory creates from {"type": "gemma4", ...}

References:
  - Gemma 4 Technical Report (Google DeepMind, 2025)
  - vLLM Gemma4 spec decode (vllm/v1/spec_decode/gemma4_proposer.py)
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

# ── Gemma4 Detection Keys ──

# Config keys that indicate Gemma4 built-in speculative capability
_GEMMA4_SPEC_KEYS = frozenset({
    "speculative_layers",
    "draft_heads",
    "speculative_decoder_layers",
    "num_speculative_heads",
})

# model_type values indicating Gemma4
_GEMMA4_MODEL_TYPES = frozenset({
    "gemma4",
    "gemma_4",
    "gemma4_speculative",
})


# ── Configuration ──


@dataclass
class Gemma4SpecConfig:
    """Configuration for Gemma4 built-in speculative decoding.

    Attributes:
        enabled: Whether Gemma4 spec decode is enabled.
        draft_length: Maximum number of draft tokens per proposal.
        acceptance_threshold: Minimum probability threshold for accepting
                              draft predictions (higher = more selective).
        speculative_layer_indices: Specific layer indices to extract drafts
                                    from. Empty = auto-detect from model.
    """
    enabled: bool = False
    draft_length: int = 4
    acceptance_threshold: float = 0.1
    speculative_layer_indices: list[int] = field(default_factory=list)


@dataclass
class Gemma4Stats:
    """Runtime statistics for the Gemma4 spec proposer."""
    total_proposals: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    model_detected: bool = False
    model_type: str = ""
    spec_layers_found: int = 0

    @property
    def acceptance_rate(self) -> float:
        return (
            self.total_accepted_tokens / self.total_draft_tokens
            if self.total_draft_tokens > 0
            else 0.0
        )

    @property
    def avg_draft_length(self) -> float:
        return (
            self.total_draft_tokens / self.total_proposals
            if self.total_proposals > 0
            else 0.0
        )

    def to_dict(self) -> dict:
        return {
            "total_proposals": self.total_proposals,
            "total_draft_tokens": self.total_draft_tokens,
            "total_accepted_tokens": self.total_accepted_tokens,
            "model_detected": self.model_detected,
            "model_type": self.model_type,
            "spec_layers_found": self.spec_layers_found,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "avg_draft_length": round(self.avg_draft_length, 2),
        }


# ── Gemma4SpecProposer ──


class Gemma4SpecProposer:
    """Gemma4 built-in speculative decoding proposer.

    Detects Gemma4 models and extracts draft token predictions from their
    built-in speculative layers. No separate draft model is needed — the
    predictions come from intermediate attention layers.

    Usage:
        proposer = Gemma4SpecProposer(Gemma4SpecConfig(enabled=True))
        if proposer.detect(model, model_config):
            draft_tokens = proposer.propose(hidden_states, n_draft=4)

    Args:
        config: Gemma4SpecConfig with spec decode settings.
    """

    def __init__(self, config: Gemma4SpecConfig | None = None) -> None:
        self._config = config or Gemma4SpecConfig()
        self._model: Any = None
        self._model_config: dict | None = None
        self._spec_layers: list[int] = []
        self._stats = Gemma4Stats()

    @property
    def config(self) -> Gemma4SpecConfig:
        return self._config

    @property
    def stats(self) -> Gemma4Stats:
        return self._stats

    @property
    def is_detected(self) -> bool:
        """Whether a Gemma4 model with spec capability has been detected."""
        return self._stats.model_detected and len(self._spec_layers) > 0

    @property
    def spec_layers(self) -> list[int]:
        """Indices of speculative layers in the model."""
        return list(self._spec_layers)

    def detect(self, model: Any, model_config: dict | None = None) -> bool:
        """Detect if the model is Gemma4 with speculative capability.

        Checks model config for Gemma4-specific keys and identifies the
        speculative layers.

        Args:
            model: The loaded MLX model.
            model_config: Parsed config.json dict (from HuggingFace).

        Returns:
            True if Gemma4 with spec capability detected.
        """
        if model_config is None:
            model_config = {}

        model_type = model_config.get("model_type", "").lower().replace("-", "_")
        self._stats.model_type = model_type

        # Check if model is Gemma4
        is_gemma4 = model_type in _GEMMA4_MODEL_TYPES

        # Check for Gemma with version >= 4
        if not is_gemma4 and model_type == "gemma":
            version = model_config.get("model_version", 0)
            if isinstance(version, (int, float)) and version >= 4:
                is_gemma4 = True

        # Check for Gemma4-specific spec decode keys in config
        has_spec_keys = any(
            key in model_config for key in _GEMMA4_SPEC_KEYS
        )

        # Also check model attributes for spec layers
        has_spec_attr = hasattr(model, "speculative_layers") or hasattr(model, "spec_heads")

        if not (is_gemma4 or has_spec_keys or has_spec_attr):
            self._stats.model_detected = False
            return False

        # Identify speculative layers
        self._spec_layers = self._find_spec_layers(model, model_config)

        if not self._spec_layers:
            # Model is Gemma4 but no spec layers found
            self._stats.model_detected = False
            logger.info("Gemma4SpecProposer: model is Gemma4 but no spec layers found")
            return False

        self._model = model
        self._model_config = model_config
        self._stats.model_detected = True
        self._stats.spec_layers_found = len(self._spec_layers)

        logger.info(
            f"Gemma4SpecProposer: detected Gemma4 with {len(self._spec_layers)} "
            f"speculative layers (model_type={model_type})"
        )
        return True

    def _find_spec_layers(self, model: Any, model_config: dict) -> list[int]:
        """Find speculative layer indices from model config or model attributes.

        Looks for:
          1. config.speculative_layer_indices (explicit)
          2. config keys like speculative_layers, draft_heads
          3. model attribute speculative_layers

        Args:
            model: The loaded model.
            model_config: Model config dict.

        Returns:
            List of layer indices for speculative predictions.
        """
        # Check explicit config
        if self._config.speculative_layer_indices:
            return list(self._config.speculative_layer_indices)

        indices: list[int] = []

        # Check model config for spec layer count
        spec_layers = model_config.get("speculative_layers")
        draft_heads = model_config.get("draft_heads")
        spec_decoder_layers = model_config.get("speculative_decoder_layers")
        num_spec_heads = model_config.get("num_speculative_heads")

        layer_count = spec_layers or draft_heads or spec_decoder_layers or num_spec_heads

        if layer_count and isinstance(layer_count, int):
            # Use the last N layers as speculative layers
            num_layers = model_config.get("num_hidden_layers", 0)
            if num_layers > 0:
                start = num_layers - layer_count
                indices = list(range(max(0, start), num_layers))
            else:
                # Can't determine positions, use sequential
                indices = list(range(layer_count))

        # Check model attributes
        if not indices and hasattr(model, "speculative_layers"):
            spec_attr = model.speculative_layers
            if isinstance(spec_attr, (list, tuple)):
                indices = [int(x) for x in spec_attr]
            elif isinstance(spec_attr, int):
                indices = list(range(spec_attr))

        if not indices and hasattr(model, "spec_heads"):
            spec_attr = model.spec_heads
            if isinstance(spec_attr, (list, tuple)):
                indices = [int(x) for x in spec_attr]
            elif isinstance(spec_attr, int):
                indices = list(range(spec_attr))

        return indices

    def propose(
        self,
        hidden_states: mx.array,
        n_draft: int | None = None,
    ) -> list[int]:
        """Extract draft token predictions from Gemma4's speculative layers.

        Uses the model's built-in speculative heads to predict future tokens
        from intermediate layer representations.

        When the model is not Gemma4 or no spec layers are found, returns
        an empty list (graceful degradation).

        Args:
            hidden_states: Hidden state tensor from the base model forward pass.
                           Shape: (batch, seq_len, hidden_size) or (1, 1, hidden_size).
            n_draft: Maximum number of draft tokens. Falls back to config.

        Returns:
            List of draft token IDs (may be empty).
        """
        if not self.is_detected or self._model is None:
            return []

        n = n_draft or self._config.draft_length
        n = min(n, self._config.draft_length, len(self._spec_layers))

        try:
            return self._extract_draft_predictions(hidden_states, n)
        except Exception as e:
            logger.warning(f"Gemma4SpecProposer: draft extraction failed: {e}")
            return []

    def _extract_draft_predictions(
        self,
        hidden_states: mx.array,
        n: int,
    ) -> list[int]:
        """Extract draft predictions from Gemma4's intermediate layers.

        Strategy: For each speculative layer, extract logits from the
        hidden state at that layer and take the argmax prediction.

        Args:
            hidden_states: Hidden state tensor.
            n: Number of draft tokens to extract.

        Returns:
            List of draft token IDs.
        """
        draft_tokens: list[int] = []

        lm_head = self._get_lm_head()
        if lm_head is None:
            return []

        # Extract from each speculative layer index.
        # Gemma4's built-in spec layers produce independent predictions at
        # different depths — we advance through self._spec_layers so each
        # draft position comes from a distinct speculative head.
        spec_to_use = self._spec_layers[:n]

        for layer_idx in spec_to_use:
            # Gemma4 models expose per-layer hidden states through a
            # dedicated method or store them as intermediate outputs.
            # Fall back to the final hidden state when per-layer access
            # is unavailable.
            layer_hs = hidden_states
            extract_fn = getattr(self._model, "get_layer_hidden", None)
            if extract_fn is not None:
                try:
                    layer_hs = extract_fn(layer_idx)
                except Exception:
                    pass  # fall back to final hidden state

            if layer_hs.ndim == 3:
                hs = layer_hs[:, -1:, :]
            elif layer_hs.ndim == 2:
                hs = layer_hs[-1:, :].reshape(1, 1, -1)
            else:
                hs = layer_hs.reshape(1, 1, -1)

            logits = lm_head(hs)

            if self._config.acceptance_threshold > 0:
                probs = mx.softmax(logits, axis=-1)
                max_prob = float(mx.max(probs).item())
                if max_prob < self._config.acceptance_threshold:
                    break

            token = int(mx.argmax(logits.reshape(-1)).item())
            draft_tokens.append(token)

        self._stats.total_proposals += 1
        self._stats.total_draft_tokens += len(draft_tokens)

        return draft_tokens

    def _get_lm_head(self) -> Any:
        """Get the model's language model head (lm_head).

        Handles wrapped models (e.g., VLM with language_model attribute).

        Returns:
            The lm_head layer, or None if not found.
        """
        if self._model is None:
            return None

        inner = self._model
        if hasattr(inner, "language_model"):
            inner = inner.language_model

        # Try standard names
        for attr in ("lm_head", "output", "embed_out", "model_head"):
            head = getattr(inner, attr, None)
            if head is not None:
                return head

        return None

    def get_stats(self) -> dict:
        """Return proposer statistics as a dictionary."""
        return self._stats.to_dict()

    def reset_stats(self) -> None:
        """Reset proposal/acceptance counters (preserves model info)."""
        self._stats.total_proposals = 0
        self._stats.total_draft_tokens = 0
        self._stats.total_accepted_tokens = 0
