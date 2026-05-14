"""Medusa speculative decoding proposer — multi-head prediction on base model hidden state.

Medusa adds K independent prediction heads on top of the base model's last hidden
state. Each head predicts the token at position offset+i (i=1..K). Unlike draft-model
spec decode, no separate model is needed — the heads share the backbone's forward pass.

Architecture:
  MedusaHead: Single linear layer mapping hidden_state -> logits for a future position.
              Initialized from the base model's lm_head weights as a starting point.
  MedusaProposer: Manages K Medusa heads, proposes draft tokens via tree-based
                  verification (Medusa tree with multiple candidate paths).
  MedusaConfig: Configuration dataclass for Medusa parameters.

Tree-based verification:
  Instead of a single greedy path (head_i picks argmax), Medusa evaluates multiple
  candidate paths (top-K from each head). A "Medusa tree" accumulates candidate
  sequences. During verification, the target model scores all candidates at once
  and accepts the longest matching prefix.

Integration:
  - SpecStrategyFactory creates MedusaStrategy from {"type": "medusa", ...}
  - YUNSHU_MEDUSA=1 env var enables Medusa in BatchedEngine
  - Compatible with ngram + cross-model via CompositeStrategy

References:
  - "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"
    (Tianle Cai et al., 2024) — https://arxiv.org/abs/2401.10774
  - vLLM Medusa implementation (vllm/v1/spec_decode/medusa_proposer.py)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


# ── Configuration ──


@dataclass
class MedusaConfig:
    """Configuration for Medusa speculative decoding.

    Attributes:
        num_heads: Number of Medusa prediction heads (each predicts offset+i).
        tree_size: Number of candidate paths to evaluate in tree verification.
        enabled: Whether Medusa is enabled (also set via YUNSHU_MEDUSA=1).
        top_k_per_head: Top-K candidates per head for tree construction.
        residual_length: Number of recent tokens to include in hidden state
                         projection (0 = use full hidden state).
    """
    num_heads: int = 4
    tree_size: int = 5
    enabled: bool = False
    top_k_per_head: int = 5
    residual_length: int = 0


@dataclass
class MedusaStats:
    """Runtime statistics for Medusa proposer."""
    total_proposals: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_tree_nodes: int = 0
    heads_attached: int = 0

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
            "total_tree_nodes": self.total_tree_nodes,
            "heads_attached": self.heads_attached,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "avg_draft_length": round(self.avg_draft_length, 2),
        }


# ── MedusaHead ──


class MedusaHead(nn.Module):
    """A single linear prediction head that maps hidden_state -> logits.

    Each head predicts the token at a different future offset from the
    base model's last hidden state. Initialized as a copy of lm_head
    weights, then optionally fine-tuned with Medusa-specific weights.

    Args:
        hidden_size: Dimension of the input hidden state.
        vocab_size: Size of the output vocabulary.
    """

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        # Residual projection: hidden -> hidden (optional bottleneck)
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(self, hidden_state: mx.array) -> mx.array:
        """Project hidden state to vocabulary logits.

        Args:
            hidden_state: Shape (batch, seq_len, hidden_size) or (1, 1, hidden_size).

        Returns:
            Logits array of shape (batch, seq_len, vocab_size).
        """
        return self.linear(hidden_state)


# ── Medusa Tree Node ──


class _MedusaTreeNode:
    """A node in the Medusa candidate tree.

    Each node represents a candidate token sequence. The tree is built
    by expanding top-K candidates from each head, then pruned to the
    tree_size best paths.

    Attributes:
        token_id: The token ID at this node.
        parent: Parent node (None for root).
        children: Child nodes.
        cumulative_logprob: Sum of log probabilities from root to this node.
        depth: Depth in the tree (0 = root).
    """
    __slots__ = (
        "token_id", "parent", "children",
        "cumulative_logprob", "depth",
    )

    def __init__(
        self,
        token_id: int = -1,
        parent: Optional[_MedusaTreeNode] = None,
        cumulative_logprob: float = 0.0,
        depth: int = 0,
    ) -> None:
        self.token_id = token_id
        self.parent = parent
        self.children: list[_MedusaTreeNode] = []
        self.cumulative_logprob = cumulative_logprob
        self.depth = depth

    def path_tokens(self) -> list[int]:
        """Return the token sequence from root to this node."""
        tokens = []
        node: Optional[_MedusaTreeNode] = self
        while node is not None and node.token_id >= 0:
            tokens.append(node.token_id)
            node = node.parent
        tokens.reverse()
        return tokens

    def add_child(
        self,
        token_id: int,
        logprob: float,
    ) -> _MedusaTreeNode:
        """Add a child node with the given token and log probability."""
        child = _MedusaTreeNode(
            token_id=token_id,
            parent=self,
            cumulative_logprob=self.cumulative_logprob + logprob,
            depth=self.depth + 1,
        )
        self.children.append(child)
        return child


# ── MedusaProposer ──


class MedusaProposer:
    """Medusa speculative decoding proposer with tree-based verification.

    Manages K Medusa heads, each predicting a token at a different future
    position. Proposes multiple candidate paths via a Medusa tree, then
    verifies against the target model's logits.

    Usage:
        proposer = MedusaProposer(MedusaConfig(num_heads=4))
        proposer.attach(model)  # Adds Medusa heads to model
        draft_tokens = proposer.propose(hidden_states, n_draft=5)
        accepted = proposer.verify(target_logits, draft_tokens)

    Args:
        config: MedusaConfig with head count, tree size, etc.
    """

    def __init__(self, config: MedusaConfig | None = None) -> None:
        self._config = config or MedusaConfig()
        self._heads: list[MedusaHead] = []
        self._attached = False
        self._model: Any = None
        self._stats = MedusaStats()
        self._vocab_size: int = 0
        self._hidden_size: int = 0

    @property
    def config(self) -> MedusaConfig:
        return self._config

    @property
    def stats(self) -> MedusaStats:
        return self._stats

    @property
    def is_attached(self) -> bool:
        return self._attached

    @property
    def heads(self) -> list[MedusaHead]:
        return list(self._heads)

    def attach(self, model: Any) -> None:
        """Attach Medusa heads to a loaded model.

        Extracts the hidden_size and vocab_size from the model's lm_head,
        then creates K Medusa heads initialized from lm_head weights.

        Args:
            model: An MLX model with an lm_head attribute (nn.Linear).
        """
        if self._attached:
            logger.warning("Medusa heads already attached, skipping")
            return

        # Resolve the inner model (handles wrapped models like VLM)
        inner = model
        if hasattr(model, "language_model"):
            inner = model.language_model

        # Find lm_head
        lm_head = getattr(inner, "lm_head", None)
        if lm_head is None:
            # Try alternative names
            for attr in ("output", "embed_out", "model_head"):
                lm_head = getattr(inner, attr, None)
                if lm_head is not None:
                    break

        if lm_head is None:
            logger.error("Cannot find lm_head on model, cannot attach Medusa heads")
            return

        # Extract dimensions
        # MLX nn.Linear weight shape is (out_features, in_features)
        if isinstance(lm_head, nn.Linear):
            self._vocab_size = lm_head.weight.shape[0]
            self._hidden_size = lm_head.weight.shape[1]
        else:
            # Try to infer from weight shape
            try:
                w = lm_head.weight
                self._vocab_size = w.shape[0]
                self._hidden_size = w.shape[1]
            except (AttributeError, IndexError):
                logger.error("Cannot infer dimensions from lm_head")
                return

        # Create K heads, initialized from lm_head weights
        self._heads = []
        for i in range(self._config.num_heads):
            head = MedusaHead(self._hidden_size, self._vocab_size)
            # Initialize from lm_head weights as starting point
            if hasattr(lm_head, "weight"):
                head.linear.weight = mx.array(lm_head.weight)
            self._heads.append(head)

        self._model = model
        self._attached = True
        self._stats.heads_attached = len(self._heads)

        logger.info(
            f"Medusa: attached {len(self._heads)} heads "
            f"(hidden={self._hidden_size}, vocab={self._vocab_size})"
        )

    def detach(self) -> None:
        """Detach Medusa heads and release memory."""
        self._heads = []
        self._attached = False
        self._model = None
        self._stats.heads_attached = 0

    def propose(
        self,
        hidden_states: mx.array,
        n_draft: int = 5,
    ) -> list[list[int]]:
        """Propose draft token sequences using Medusa tree.

        Runs each head on the hidden state, builds a candidate tree from
        top-K logits per head, prunes to tree_size best paths, and returns
        the candidate sequences.

        Args:
            hidden_states: Last hidden state from the base model forward pass.
                           Shape: (batch, seq_len, hidden_size) or (1, 1, hidden_size).
            n_draft: Maximum number of draft tokens per candidate path.

        Returns:
            List of candidate token sequences (sorted by cumulative logprob,
            best first). May be empty if heads are not attached.
        """
        if not self._attached or not self._heads:
            return []

        self._stats.total_proposals += 1

        # Get logits from each head
        head_logits = []
        for head in self._heads:
            logits = head(hidden_states)  # (batch, seq, vocab)
            head_logits.append(logits[:, -1, :])  # (batch, vocab)

        # Build Medusa tree
        candidates = self._build_tree(head_logits, n_draft)

        if candidates:
            self._stats.total_draft_tokens += sum(len(c) for c in candidates)
            self._stats.total_tree_nodes += sum(len(c) for c in candidates)

        return candidates

    def _build_tree(
        self,
        head_logits: list[mx.array],
        max_depth: int,
    ) -> list[list[int]]:
        """Build Medusa candidate tree from head logits.

        For each head, takes top-K tokens. Expands the tree level by level,
        keeping only the best tree_size paths based on cumulative log probability.

        Args:
            head_logits: List of (batch, vocab) logits, one per head.
            max_depth: Maximum tree depth (= number of draft tokens).

        Returns:
            List of candidate token sequences, best first.
        """
        tree_size = self._config.tree_size
        top_k = self._config.top_k_per_head

        # Root node
        root = _MedusaTreeNode(token_id=-1, depth=0)

        # Current frontier: nodes to expand at each level
        frontier: list[_MedusaTreeNode] = [root]

        num_levels = min(len(head_logits), max_depth)

        for level in range(num_levels):
            logits = head_logits[level]  # (batch, vocab)
            log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)

            # Take top-K from this head's logits
            # logits shape: (1, vocab) or (vocab,) after squeezing
            if logits.ndim > 1:
                logits_1d = logits.reshape(-1)
                log_probs_1d = log_probs.reshape(-1)
            else:
                logits_1d = logits
                log_probs_1d = log_probs

            top_k_actual = min(top_k, logits_1d.shape[0])
            top_indices = mx.argsort(logits_1d)[-top_k_actual:][::-1]
            top_logprobs = log_probs_1d[top_indices]

            # Expand frontier
            new_frontier: list[_MedusaTreeNode] = []
            for node in frontier:
                for j in range(top_k_actual):
                    child = node.add_child(
                        token_id=int(top_indices[j].item()),
                        logprob=float(top_logprobs[j].item()),
                    )
                    new_frontier.append(child)

            # Prune: keep only tree_size best nodes by cumulative logprob
            if len(new_frontier) > tree_size:
                new_frontier.sort(
                    key=lambda n: n.cumulative_logprob,
                    reverse=True,
                )
                new_frontier = new_frontier[:tree_size]

            frontier = new_frontier

            if not frontier:
                break

        # Extract paths from frontier nodes
        candidates = []
        for node in frontier:
            tokens = node.path_tokens()
            if tokens:
                candidates.append(tokens)

        # Sort by cumulative logprob (descending)
        candidates.sort(
            key=lambda t: sum(
                float(top_logprobs[k].item()) if k < len(top_logprobs) else 0.0
                for k in range(len(t))
            ),
            reverse=True,
        )

        return candidates

    def verify(
        self,
        target_logits: mx.array,
        draft_tokens: list[list[int]],
    ) -> tuple[int, list[int]]:
        """Verify draft token candidates against target model logits.

        For each candidate sequence, checks how many tokens match the target
        model's greedy predictions. Returns the best matching sequence and
        the number of accepted tokens.

        Args:
            target_logits: Target model logits for the draft positions.
                           Shape: (batch, num_draft_tokens, vocab) or
                           list of per-position logits.
            draft_tokens: List of candidate token sequences from propose().

        Returns:
            Tuple of (accepted_count, best_match_tokens).
            accepted_count is the number of verified tokens from the best path.
        """
        if not draft_tokens:
            return 0, []

        best_count = 0
        best_tokens: list[int] = []

        for candidate in draft_tokens:
            accepted = self._verify_single_path(target_logits, candidate)
            if accepted > best_count:
                best_count = accepted
                best_tokens = candidate[:accepted]

        self._stats.total_accepted_tokens += best_count

        return best_count, best_tokens

    def _verify_single_path(
        self,
        target_logits: mx.array,
        candidate: list[int],
    ) -> int:
        """Verify a single candidate path against target logits.

        Checks each position: if target model's argmax matches the candidate
        token, count as accepted. Stop at first mismatch.

        Args:
            target_logits: Target model logits. Shape depends on how the
                           verification is performed. Can be:
                           (1, num_positions, vocab) — batch of position logits
            candidate: Single candidate token sequence.

        Returns:
            Number of accepted tokens (0 to len(candidate)).
        """
        if not candidate:
            return 0

        if target_logits.ndim == 3:
            # (batch, positions, vocab) -> take first batch
            logits_2d = target_logits[0]
        elif target_logits.ndim == 2:
            logits_2d = target_logits
        else:
            logits_2d = target_logits.reshape(1, -1)

        num_positions = min(logits_2d.shape[0], len(candidate))
        accepted = 0

        for i in range(num_positions):
            target_token = int(mx.argmax(logits_2d[i]).item())
            if target_token == candidate[i]:
                accepted += 1
            else:
                break

        return accepted

    def load_medusa_weights(self, path: str | Path) -> bool:
        """Load pre-trained Medusa head weights from a directory.

        Expected directory structure:
          medusa_head_0.safetensors
          medusa_head_1.safetensors
          ...
          config.json (optional, overrides MedusaConfig)

        Args:
            path: Directory containing Medusa head weight files.

        Returns:
            True if all heads loaded successfully.
        """
        if not self._attached:
            logger.error("Cannot load Medusa weights: heads not attached")
            return False

        path = Path(path)
        if not path.is_dir():
            logger.error(f"Medusa weights directory not found: {path}")
            return False

        import json

        # Load config if present
        config_path = path / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    medusa_config = json.load(f)
                num_heads = medusa_config.get("num_heads", self._config.num_heads)
                if num_heads != len(self._heads):
                    logger.warning(
                        f"Config says {num_heads} heads but {len(self._heads)} attached"
                    )
            except Exception as e:
                logger.warning(f"Failed to load Medusa config.json: {e}")

        # Load weights for each head
        loaded = 0
        for i, head in enumerate(self._heads):
            weight_file = path / f"medusa_head_{i}.safetensors"
            if not weight_file.exists():
                weight_file = path / f"medusa_lm_head_{i}.safetensors"
            if not weight_file.exists():
                logger.warning(f"Weight file for head {i} not found in {path}")
                continue

            try:
                from mlx.utils import tree_flatten
                weights = mx.load(str(weight_file))
                # Expect a single "weight" or "linear.weight" key
                w = None
                for key in ("weight", "linear.weight"):
                    if key in weights:
                        w = weights[key]
                        break
                if w is None:
                    # Take the first weight tensor
                    flat = tree_flatten(weights)
                    if flat:
                        w = flat[0][1]

                if w is not None:
                    if w.shape == head.linear.weight.shape:
                        head.linear.weight = mx.array(w)
                        loaded += 1
                    else:
                        logger.warning(
                            f"Head {i}: weight shape mismatch "
                            f"(got {w.shape}, expected {head.linear.weight.shape})"
                        )
            except Exception as e:
                logger.error(f"Failed to load weights for head {i}: {e}")

        logger.info(f"Medusa: loaded weights for {loaded}/{len(self._heads)} heads from {path}")
        return loaded == len(self._heads)

    def get_stats(self) -> dict:
        """Return proposer statistics as a dictionary."""
        return self._stats.to_dict()

    def reset_stats(self) -> None:
        """Reset all accumulated statistics."""
        self._stats = MedusaStats(heads_attached=len(self._heads))

    def forward_heads(self, hidden_states: mx.array) -> list[mx.array]:
        """Run all heads on hidden states and return per-head logits.

        This is used by the engine to get Medusa logits alongside the
        main model's forward pass, without extra overhead.

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            List of (batch, seq_len, vocab_size) logits, one per head.
        """
        if not self._attached:
            return []
        return [head(hidden_states) for head in self._heads]

    def greedy_propose(self, hidden_states: mx.array) -> list[int]:
        """Simple greedy proposal: each head picks argmax.

        Faster than tree-based propose() but only produces one path.
        Useful for latency-sensitive scenarios.

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            List of draft token IDs, one per head.
        """
        if not self._attached:
            return []

        tokens = []
        for head in self._heads:
            logits = head(hidden_states)[:, -1, :]
            tokens.append(int(mx.argmax(logits).item()))

        self._stats.total_proposals += 1
        self._stats.total_draft_tokens += len(tokens)
        return tokens
