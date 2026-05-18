from __future__ import annotations
"""LLM-based speculative decoding proposer — uses a smaller LLM as draft model.

Unlike cross-model spec decode (EAGLE-3 pattern) which tightly couples draft
and target models, this proposer provides a general interface for using any
smaller LLM as a draft model. The draft model is loaded independently and
generates token proposals that are verified against the target model.

Architecture:
  LLMProposerConfig: Configuration dataclass for the LLM proposer.
  LLMProposer: Manages draft model loading, proposal generation, and memory.
  LLMStats: Runtime statistics tracking.

Lifecycle:
  1. LLMProposer(config) — create with configuration
  2. load_draft_model("Qwen2.5-0.5B") — load smaller model
  3. propose(context_ids, n_draft=5) — generate N draft tokens
  4. unload_draft_model() — release memory when done

Graceful degradation:
  - If draft model can't be loaded (OOM), returns empty proposals
  - If draft model fails during proposal, returns empty proposals
  - Memory usage tracked and reported for monitoring

Integration:
  - LLMStrategy wraps as SpecStrategy for CompositeStrategy
  - SpecStrategyFactory creates from {"type": "llm", ...}
  - Compatible with ngram + medusa via CompositeStrategy

References:
  - vLLM LLM-based proposer (vllm/v1/spec_decode/llm_proposer.py)
  - "Fast Inference from Transformers via Speculative Decoding"
    (Leviathan et al., 2023) — https://arxiv.org/abs/2211.17192
"""

import logging
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)


# ── Configuration ──


@dataclass
class LLMProposerConfig:
    """Configuration for LLM-based speculative decoding proposer.

    Attributes:
        draft_model_name: Name/path of the smaller model to use as drafter.
        max_draft_length: Maximum number of draft tokens per proposal.
        temperature: Sampling temperature (0.0 = greedy for deterministic
                     matching, which improves acceptance rate).
        top_k: Top-K for sampling (1 = greedy).
        compiled: Whether to use mx.compile() for the draft model forward pass.
    """
    draft_model_name: str = ""
    max_draft_length: int = 5
    temperature: float = 0.0
    top_k: int = 1
    compiled: bool = False


@dataclass
class LLMStats:
    """Runtime statistics for the LLM proposer."""
    total_proposals: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_failed_proposals: int = 0
    draft_model_loaded: bool = False
    draft_model_name: str = ""
    draft_model_memory_mb: float = 0.0

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
            "total_failed_proposals": self.total_failed_proposals,
            "draft_model_loaded": self.draft_model_loaded,
            "draft_model_name": self.draft_model_name,
            "draft_model_memory_mb": round(self.draft_model_memory_mb, 2),
            "acceptance_rate": round(self.acceptance_rate, 4),
            "avg_draft_length": round(self.avg_draft_length, 2),
        }


# ── LLMProposer ──


class LLMProposer:
    """LLM-based speculative decoding proposer.

    Loads a separate smaller model as the draft model. The draft model
    generates token proposals that are verified against the target model.

    Supports mx.compile() for draft model speedup if available.

    Usage:
        proposer = LLMProposer(LLMProposerConfig(
            draft_model_name="Qwen2.5-0.5B-Instruct-4bit",
            max_draft_length=5,
            temperature=0.0,
        ))
        proposer.load_draft_model("Qwen2.5-0.5B-Instruct-4bit")
        draft_tokens = proposer.propose(context_ids, n_draft=5)
        proposer.unload_draft_model()

    Args:
        config: LLMProposerConfig with draft model settings.
    """

    def __init__(self, config: LLMProposerConfig | None = None) -> None:
        self._config = config or LLMProposerConfig()
        self._model: Any = None
        self._tokenizer: Any = None
        self._cache: list | None = None
        self._compiled_fn: Any = None
        self._stats = LLMStats()

    @property
    def config(self) -> LLMProposerConfig:
        return self._config

    @property
    def stats(self) -> LLMStats:
        return self._stats

    @property
    def is_loaded(self) -> bool:
        """Whether a draft model is currently loaded."""
        return self._model is not None

    @property
    def draft_model(self) -> Any:
        """Access the loaded draft model (may be None)."""
        return self._model

    @property
    def draft_tokenizer(self) -> Any:
        """Access the draft tokenizer (may be None)."""
        return self._tokenizer

    def load_draft_model(self, model_name: str | None = None) -> bool:
        """Load a smaller model as the draft model.

        Args:
            model_name: Model name/path. Falls back to config.draft_model_name.

        Returns:
            True if model loaded successfully, False on failure (e.g., OOM).
        """
        name = model_name or self._config.draft_model_name
        if not name:
            logger.warning("LLMProposer: no draft model name provided")
            return False

        try:
            from mlx_lm.utils import load as load_model

            logger.info(f"LLMProposer: loading draft model '{name}'...")
            self._model, self._tokenizer = load_model(name)

            # Record memory usage
            self._stats.draft_model_memory_mb = _estimate_model_memory(self._model)

            # Optionally compile the draft model's forward pass
            if self._config.compiled:
                try:
                    self._compiled_fn = mx.compile(self._model)
                    logger.info("LLMProposer: compiled draft model forward pass")
                except Exception as e:
                    logger.warning(f"LLMProposer: mx.compile() failed: {e}")
                    self._compiled_fn = None

            self._stats.draft_model_loaded = True
            self._stats.draft_model_name = name

            logger.info(
                f"LLMProposer: loaded draft model '{name}' "
                f"(memory: {self._stats.draft_model_memory_mb:.1f} MB)"
            )
            return True

        except MemoryError:
            logger.warning(
                f"LLMProposer: OOM loading draft model '{name}', "
                f"falling back to no proposals"
            )
            self._model = None
            self._tokenizer = None
            self._stats.draft_model_loaded = False
            return False

        except Exception as e:
            logger.error(f"LLMProposer: failed to load draft model '{name}': {e}", exc_info=True)
            self._model = None
            self._tokenizer = None
            self._stats.draft_model_loaded = False
            return False

    def unload_draft_model(self) -> None:
        """Release draft model memory.

        Clears all references to allow GC to reclaim Metal buffers.
        """
        self._model = None
        self._tokenizer = None
        self._cache = None
        self._compiled_fn = None
        self._stats.draft_model_loaded = False
        self._stats.draft_model_memory_mb = 0.0
        logger.info("LLMProposer: unloaded draft model")

    def propose(
        self,
        context_ids: list[int],
        n_draft: int | None = None,
    ) -> list[int]:
        """Generate N draft tokens given the current context.

        Runs the draft model forward pass to propose tokens. Uses greedy
        decoding (temperature=0.0) for deterministic matching against the
        target model.

        Graceful degradation: returns empty list if draft model is not loaded
        or if any error occurs during proposal generation.

        Args:
            context_ids: Current token IDs (prompt + generated tokens so far).
            n_draft: Number of draft tokens to generate. Falls back to
                     config.max_draft_length.

        Returns:
            List of proposed draft token IDs (may be empty).
        """
        if not self.is_loaded:
            return []

        n = n_draft or self._config.max_draft_length
        n = min(n, self._config.max_draft_length)

        try:
            return self._generate_draft(context_ids, n)
        except Exception as e:
            logger.warning(f"LLMProposer: draft generation failed: {e}")
            self._stats.total_failed_proposals += 1
            return []

    def _generate_draft(self, context_ids: list[int], n: int) -> list[int]:
        """Internal draft generation using the draft model.

        Args:
            context_ids: Full context token IDs.
            n: Number of draft tokens to generate.

        Returns:
            List of draft token IDs.
        """
        try:
            from mlx_lm.models.cache import make_prompt_cache
            # Create fresh cache for this proposal
            self._cache = make_prompt_cache(self._model)
        except Exception:
            logger.debug("operation failed", exc_info=True)
            # Model may not have standard layers (e.g., test mocks)
            self._cache = None

        # Prefill: process the context
        input_ids = mx.array([context_ids])
        forward_fn = self._compiled_fn or self._model
        output = forward_fn(input_ids, cache=self._cache)
        logits = output.logits[:, -1, :] if hasattr(output, 'logits') else output[:, -1, :]

        # Generate n tokens autoregressively
        draft_tokens: list[int] = []
        current_logits = logits

        for _ in range(n):
            if self._config.temperature == 0.0 or self._config.top_k == 1:
                # Greedy: argmax for deterministic matching
                next_token = int(mx.argmax(current_logits, axis=-1).item())
            else:
                # Sample with temperature
                next_token = self._sample_token(current_logits)

            draft_tokens.append(next_token)

            # Feed back for next step
            next_input = mx.array([[next_token]])
            output = forward_fn(next_input, cache=self._cache)
            current_logits = output.logits[:, -1, :] if hasattr(output, 'logits') else output[:, -1, :]

        self._stats.total_proposals += 1
        self._stats.total_draft_tokens += len(draft_tokens)

        return draft_tokens

    def _sample_token(self, logits: mx.array) -> int:
        """Sample a token from logits with temperature and top-k.

        Args:
            logits: Logits array of shape (1, vocab_size) or (vocab_size,).

        Returns:
            Sampled token ID.
        """
        if logits.ndim > 1:
            logits = logits.squeeze(0)

        # Apply temperature
        if self._config.temperature > 0:
            logits = logits / self._config.temperature

        # Apply top-k
        if self._config.top_k > 0:
            k = min(self._config.top_k, logits.shape[0])
            top_k_indices = mx.argsort(logits)[-k:]
            mask = mx.full(logits.shape, float('-inf'))
            mask[top_k_indices] = logits[top_k_indices]
            logits = mask

        # Sample
        probs = mx.softmax(logits)
        token = mx.random.categorical(logits.reshape(1, -1), axis=-1)
        return int(token.item())

    def get_stats(self) -> dict:
        """Return proposer statistics as a dictionary."""
        return self._stats.to_dict()

    def reset_stats(self) -> None:
        """Reset proposal/acceptance counters (preserves model info)."""
        self._stats.total_proposals = 0
        self._stats.total_draft_tokens = 0
        self._stats.total_accepted_tokens = 0
        self._stats.total_failed_proposals = 0


def _estimate_model_memory(model: Any) -> float:
    """Estimate model memory usage in MB.

    Sums all array parameters recursively via MLX tree utilities.

    Args:
        model: An MLX model (nn.Module).

    Returns:
        Estimated memory in megabytes.
    """
    total_bytes = 0
    try:
        from mlx.utils import tree_flatten_with_path
        for path, leaf in tree_flatten_with_path(model.parameters()):
            if hasattr(leaf, 'nbytes'):
                total_bytes += leaf.nbytes
    except ImportError:
        # Fallback: tree_flatten (older MLX)
        try:
            from mlx.utils import tree_flatten
            flat = tree_flatten(model.parameters())
            for name, leaf in flat:
                if hasattr(leaf, 'nbytes'):
                    total_bytes += leaf.nbytes
        except Exception:
            logger.debug("operation failed", exc_info=True)
    except Exception:
        logger.debug("operation failed", exc_info=True)
    return total_bytes / (1024 * 1024)
