"""Gemma-4 MTP assistant drafter — centroid-based sparse logit masking (MLX).

Port of vLLM's ``Gemma4MTPMaskedEmbedder`` (see
``reference/vllm/vllm/model_executor/models/gemma4_mtp.py``) to MLX.

The assistant drafter predicts the next token without materializing a
full-vocabulary logit tensor. It projects the hidden state to a small set
of *centroids*, selects the top-K centroids, and only scores the tokens
that belong to those centroids' clusters. For a 256k-vocab model with
2048 centroids and top_k=32 this scores ~4096 tokens instead of 256k —
a 64x reduction in the lm_head GEMM.

Only the pure-tensor-math embedder is implemented here; it is fully
testable offline with random tensors (no real weights required). The
decoder layers and weight loading depend on the real Gemma-4 checkpoint
and are intentionally deferred until the model drive is available — see
the module-level note below.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# --- DEFERRED (genuinely needs the real Gemma-4 checkpoint + model drive) ---
# Only two things below truly require the drive:
# 1. load_weights() — mapping safetensors keys (incl. gate/up stacking and
# the embed_tokens<->lm_head tie) onto these modules.
# 2. Numerical-accuracy validation against the real Gemma-4 outputs, and
# wiring the Q-only attention's K/V to the *target* model's live KV cache.
# The module architecture itself (MLP, Q-only attention, decoder layer, the
# pre/post-projection predictor pipeline) is ported below and is structurally
# testable offline with random weights + synthetic external K/V — see
# tests/unit/test_gemma4_assistant.py.


class Gemma4MTPMaskedEmbedder(nn.Module):
    """Sparse logit computation via centroid-based vocabulary masking.

    Instead of computing logits against the full vocabulary, projects the
    hidden state to ``num_centroids`` centroid scores, selects the top-K
    centroids, and computes logits only for the
    ``top_k * (vocab_size / num_centroids)`` tokens belonging to those
    centroids.

    ``token_ordering`` is a vocab-length permutation buffer that groups
    token IDs into contiguous per-centroid clusters; reshaping it to
    ``(num_centroids, vocab_size_per_centroid)`` yields each centroid's
    member token IDs.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_centroids: int,
        centroid_intermediate_top_k: int,
    ) -> None:
        super().__init__()
        if vocab_size % num_centroids != 0:
            raise ValueError(
                f"vocab_size ({vocab_size}) must be divisible by "
                f"num_centroids ({num_centroids})"
            )
        if not 0 < centroid_intermediate_top_k <= num_centroids:
            raise ValueError(
                f"centroid_intermediate_top_k ({centroid_intermediate_top_k}) "
                f"must be in (0, num_centroids={num_centroids}]"
            )
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_centroids = num_centroids
        self.centroid_intermediate_top_k = centroid_intermediate_top_k
        self.vocab_size_per_centroid = vocab_size // num_centroids
        self.num_selected = centroid_intermediate_top_k * self.vocab_size_per_centroid

        self.centroids = nn.Linear(hidden_size, num_centroids, bias=False)
        # Permutation buffer; populated from the checkpoint at load time.
        # Default identity ordering keeps the module usable offline.
        self.token_ordering = mx.arange(vocab_size, dtype=mx.int32)

    def _select_and_score(
        self,
        hidden_states: mx.array,
        lm_head_weight: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Centroid selection + sparse dot product.

        Args:
            hidden_states: ``(num_tokens, hidden_size)``.
            lm_head_weight: ``(vocab_size, hidden_size)`` output embeddings.

        Returns:
            logits: ``(num_tokens, num_selected)`` sparse logits.
            indices: ``(num_tokens, num_selected)`` corresponding vocab IDs.
        """
        num_tokens = hidden_states.shape[0]
        scores = self.centroids(hidden_states)  # (num_tokens, num_centroids)
        # Top-K centroid indices. Order within the top-K is irrelevant since
        # every selected centroid's full cluster is scored and the final
        # argmax is taken over the union.
        k = self.centroid_intermediate_top_k
        top_k_indices = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]

        clusters = self.token_ordering.reshape(
            self.num_centroids, self.vocab_size_per_centroid
        )
        # (num_tokens, k, vocab_size_per_centroid)
        selected = clusters[top_k_indices]
        # (num_tokens * num_selected, hidden) -> (num_tokens, num_selected, hidden)
        embeddings = lm_head_weight[selected.reshape(-1)].reshape(
            num_tokens, self.num_selected, self.hidden_size
        )
        # einsum("td,tsd->ts"): per-token dot product against each candidate.
        logits = (hidden_states[:, None, :] * embeddings).sum(axis=-1)
        return logits, selected.reshape(num_tokens, -1)

    def __call__(
        self,
        hidden_states: mx.array,
        lm_head_weight: mx.array,
    ) -> mx.array:
        """Full-vocab logits with non-selected positions masked to -inf."""
        logits, indices = self._select_and_score(hidden_states, lm_head_weight)
        num_tokens = hidden_states.shape[0]
        neg_inf = mx.finfo(hidden_states.dtype).min
        flat = mx.full((num_tokens * self.vocab_size,), neg_inf, dtype=logits.dtype)
        rows = mx.arange(num_tokens, dtype=indices.dtype)[:, None]
        flat_idx = (rows * self.vocab_size + indices).reshape(-1)
        flat[flat_idx] = logits.reshape(-1)
        return flat.reshape(num_tokens, self.vocab_size)

    def get_top_tokens(
        self,
        hidden_states: mx.array,
        lm_head_weight: mx.array,
    ) -> mx.array:
        """Sparse argmax — returns vocab token IDs without a full-vocab tensor."""
        logits, indices = self._select_and_score(hidden_states, lm_head_weight)
        best = mx.argmax(logits, axis=-1)  # (num_tokens,)
        return mx.take_along_axis(indices, best[:, None], axis=-1).squeeze(-1)


class GemmaRMSNorm(nn.Module):
    """RMSNorm for the Gemma-4 *assistant drafter*: standard ``weight``
    multiply (NOT the Gemma2/3 ``1 + weight`` convention).

    The assistant checkpoint stores full-gain norm weights (e.g. model.norm
    mean ≈ 9.8, not centered at 0), and mlx-lm's gemma4 uses a plain
    ``nn.RMSNorm``. Using ``1 + weight`` here corrupts every norm and was the
    cause of 0% draft acceptance until verified against real weights — a
    convention bug random-tensor shape tests cannot catch.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        f = x.astype(mx.float32)
        normed = f * mx.rsqrt(mx.mean(f * f, axis=-1, keepdims=True) + self.eps)
        return (normed * self.weight.astype(mx.float32)).astype(x.dtype)


class Gemma4MLP(nn.Module):
    """Gated MLP with gelu-tanh activation (``gelu_pytorch_tanh``)."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class Gemma4MTPAttention(nn.Module):
    """Q-only attention for the MTP drafter.

    The drafter projects only the query; the key/value come from the
    *target* model's KV cache (KV sharing). Offline, that cache is modelled
    as the ``k``/``v`` arguments to ``__call__`` — shape
    ``(1, num_kv_heads, T, head_dim)`` — which is exactly the interface the
    live integration will feed. GQA (``num_kv_heads < num_heads``) is handled
    natively by MLX's scaled_dot_product_attention.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_base: float = 10000.0,
        scaling: float = 1.0,
        eps: float = 1e-6,
        rope_dims: int | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = scaling
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = GemmaRMSNorm(head_dim, eps)
        # rope_dims < head_dim => partial rotary (Gemma-4 full-attention layers
        # rotate only partial_rotary_factor * head_dim; nn.RoPE passes the rest
        # through unchanged).
        self.rope = nn.RoPE(
            rope_dims if rope_dims is not None else head_dim,
            traditional=False,
            base=rope_base,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        k: mx.array,
        v: mx.array,
        mask: str | mx.array = "causal",
        offset: int = 0,
    ) -> mx.array:
        t = hidden_states.shape[0]
        q = self.q_proj(hidden_states).reshape(t, self.num_heads, self.head_dim)
        q = self.q_norm(q)  # per-head RMSNorm
        # (T, H, D) -> (1, H, T, D) for RoPE + SDPA
        q = q.transpose(1, 0, 2)[None]
        # offset = query's absolute start position, so its RoPE matches the
        # target's cached-K RoPE when drafting from a single later position.
        q = self.rope(q, offset=offset)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scaling, mask=mask
        )
        # (1, H, T, D) -> (T, H*D)
        out = out[0].transpose(1, 0, 2).reshape(t, self.num_heads * self.head_dim)
        return self.o_proj(out)


class Gemma4MTPDecoderLayer(nn.Module):
    """Single MTP decoder layer: pre/post norms around attention + MLP, with
    Gemma's residual-then-scalar structure (``hidden * layer_scalar``)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_base: float = 10000.0,
        scaling: float = 1.0,
        eps: float = 1e-6,
        rope_dims: int | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = Gemma4MTPAttention(
            hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_base,
            scaling,
            eps,
            rope_dims,
        )
        self.mlp = Gemma4MLP(hidden_size, intermediate_size)
        self.input_layernorm = GemmaRMSNorm(hidden_size, eps)
        self.post_attention_layernorm = GemmaRMSNorm(hidden_size, eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(hidden_size, eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(hidden_size, eps)
        self.layer_scalar = mx.ones((1,))

    def __call__(
        self,
        hidden_states: mx.array,
        k: mx.array,
        v: mx.array,
        mask: str | mx.array = "causal",
        offset: int = 0,
    ) -> mx.array:
        residual = hidden_states
        h = self.input_layernorm(residual)
        h = self.self_attn(h, k, v, mask, offset)
        h = self.post_attention_layernorm(h)
        h = h + residual
        residual = h
        h = self.pre_feedforward_layernorm(h)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        h = h + residual
        return h * self.layer_scalar


class Gemma4MultiTokenPredictor(nn.Module):
    """MTP drafter body: ``cat(inputs_embeds, hidden)`` → pre_projection →
    decoder layers → norm → (draft_hidden, post_projection→backbone_hidden).

    ``forward`` takes ``inputs_embeds`` and ``hidden_states`` directly (both
    backbone-dim). In the live system ``inputs_embeds`` comes from the target
    model's shared embedding and ``hidden_states`` is the target's last hidden;
    that embedding-sharing + K/V-sharing wiring is the drive-blocked part. The
    projection→layers→norm pipeline below is fully shape-testable offline.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        backbone_hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_base: float = 10000.0,
        scaling: float = 1.0,
        eps: float = 1e-6,
        head_dims: list[int] | None = None,
        rope_bases: list[float] | None = None,
        rope_dims: list[int | None] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.backbone_hidden_size = backbone_hidden_size
        # Replaced by the target's backbone-dim embedding after sharing.
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.pre_projection = nn.Linear(
            2 * backbone_hidden_size, hidden_size, bias=False
        )
        self.post_projection = nn.Linear(hidden_size, backbone_hidden_size, bias=False)
        # Per-layer overrides let sliding (head_dim=256) and full-attention
        # (global_head_dim=512, partial rotary) layers coexist, matching the
        # real Gemma-4 drafter. Falls back to uniform values when not given.
        hd = head_dims or [head_dim] * num_layers
        rb = rope_bases or [rope_base] * num_layers
        rd = rope_dims or [None] * num_layers
        self.layers = [
            Gemma4MTPDecoderLayer(
                hidden_size,
                intermediate_size,
                num_heads,
                num_kv_heads,
                hd[i],
                rb[i],
                scaling,
                eps,
                rd[i],
            )
            for i in range(num_layers)
        ]
        self.norm = GemmaRMSNorm(hidden_size, eps)
        self.normalizer = float(backbone_hidden_size**0.5)

    @classmethod
    def from_hf_config(cls, cfg: dict) -> Gemma4MultiTokenPredictor:
        """Build from a Gemma-4 assistant ``config.json`` dict (per-layer
        head_dim + rope derived from ``layer_types`` / ``rope_parameters``)."""
        tcfg = cfg["text_config"]
        n = tcfg["num_hidden_layers"]
        head_dim = tcfg["head_dim"]
        global_head_dim = tcfg.get("global_head_dim", head_dim)
        layer_types = tcfg["layer_types"]
        rope_params = tcfg["rope_parameters"]
        head_dims, rope_bases, rope_dims = [], [], []
        for i in range(n):
            lt = layer_types[i]
            is_full = lt == "full_attention"
            hd = global_head_dim if is_full else head_dim
            rp = rope_params.get(lt, {})
            head_dims.append(hd)
            rope_bases.append(float(rp.get("rope_theta", 10000.0)))
            prf = rp.get("partial_rotary_factor")
            rope_dims.append(int(hd * prf) if prf else None)
        return cls(
            vocab_size=tcfg["vocab_size"],
            hidden_size=tcfg["hidden_size"],
            backbone_hidden_size=cfg["backbone_hidden_size"],
            intermediate_size=tcfg["intermediate_size"],
            num_layers=n,
            num_heads=tcfg["num_attention_heads"],
            num_kv_heads=tcfg["num_key_value_heads"],
            head_dim=head_dim,
            eps=tcfg.get("rms_norm_eps", 1e-6),
            head_dims=head_dims,
            rope_bases=rope_bases,
            rope_dims=rope_dims,
        )

    def embed_input_ids(self, input_ids: mx.array) -> mx.array:
        return self.embed_tokens(input_ids) * self.normalizer

    def __call__(
        self,
        inputs_embeds: mx.array,
        hidden_states: mx.array,
        kv_per_layer: list[tuple[mx.array, mx.array]],
        mask: str | mx.array = "causal",
        offset: int = 0,
    ) -> tuple[mx.array, mx.array]:
        if len(kv_per_layer) != len(self.layers):
            raise ValueError(
                f"kv_per_layer has {len(kv_per_layer)} entries but model has "
                f"{len(self.layers)} layers"
            )
        combined = mx.concatenate([inputs_embeds, hidden_states], axis=-1)
        h = self.pre_projection(combined)
        for layer, (k, v) in zip(self.layers, kv_per_layer, strict=True):
            h = layer(h, k, v, mask, offset)
        draft_hidden = self.norm(h)
        backbone_hidden = self.post_projection(draft_hidden)
        return draft_hidden, backbone_hidden


def load_assistant_drafter(
    model_dir: str | Path,
) -> tuple[Gemma4MultiTokenPredictor, Gemma4MTPMaskedEmbedder, dict]:
    """Load the real gemma-4-*-assistant checkpoint into the ported modules.

    Returns ``(predictor, masked_embedder, config)``. Uses strict weight
    loading so every checkpoint key must map onto a module parameter (and
    vice-versa) — a missing/extra/mis-shaped key raises, which is exactly the
    validation we want for the port.

    The checkpoint namespaces ``embed_tokens``/``layers``/``norm`` under a
    ``model.`` prefix while keeping ``pre_projection``/``post_projection``/
    ``masked_embedding`` at the top level; this remaps accordingly.
    """
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    tcfg = cfg["text_config"]

    predictor = Gemma4MultiTokenPredictor.from_hf_config(cfg)
    embedder = Gemma4MTPMaskedEmbedder(
        hidden_size=tcfg["hidden_size"],
        vocab_size=tcfg["vocab_size"],
        num_centroids=cfg["num_centroids"],
        centroid_intermediate_top_k=cfg["centroid_intermediate_top_k"],
    )

    raw = mx.load(str(model_dir / "model.safetensors"))
    predictor_w: list[tuple[str, mx.array]] = []
    embedder_w: list[tuple[str, mx.array]] = []
    for key, arr in raw.items():
        if key.startswith("masked_embedding."):
            embedder_w.append((key[len("masked_embedding.") :], arr))
        elif key.startswith("model."):
            predictor_w.append((key[len("model.") :], arr))
        else:  # pre_projection / post_projection — already top-level
            predictor_w.append((key, arr))

    predictor.load_weights(predictor_w, strict=True)
    embedder.load_weights(embedder_w, strict=True)
    mx.eval(predictor.parameters(), embedder.parameters())
    return predictor, embedder, cfg


class Gemma4AssistantProposer:
    """Speculative-decode proposer using the dual-load Gemma-4 assistant drafter.

    Validated end-to-end at 90.1% draft acceptance against the real target.
    This wraps the per-step draft primitive so the
    serving loop can use it: given the target's last-layer hidden state, the
    embedding of the most-recent token (target-shared embedding, backbone-dim),
    and the target's KV cache, it proposes the next token via the centroid
    MaskedEmbedder sparse argmax.

    KV sharing: each draft layer reads K/V from the target's last *non*-shared
    layer of the matching attention type (sliding/full), per vLLM's
    ``_setup_gemma4_kv_sharing``. The two target layer indices are resolved
    once from the target config and passed in as ``sliding_kv_layer`` /
    ``full_kv_layer``.

    The drafter layer types come from its own config; this proposer assumes the
    gemma-4-E4B assistant layout [sliding, sliding, sliding, full]. ``propose``
    returns the single most-likely next token (the drafter proposes 1 token per
    step; multi-token speculation chains these with the target verifying each).
    """

    def __init__(
        self,
        predictor: Gemma4MultiTokenPredictor,
        embedder: Gemma4MTPMaskedEmbedder,
        target_embed_weight: mx.array,
        target_embed_scale: float,
        sliding_kv_layer: int,
        full_kv_layer: int,
    ) -> None:
        self.predictor = predictor
        self.embedder = embedder
        # lm_head is the drafter's tied draft-dim embedding (centroid path uses
        # it as the candidate-token embedding table).
        self._lm_head = predictor.embed_tokens.weight
        # Target's embedding (backbone-dim) — after embedding-sharing the drafter
        # consumes the TARGET's token embedding, not its own.
        self._target_embed_weight = target_embed_weight
        self._target_embed_scale = target_embed_scale
        self._layer_types = [
            "sliding" if i < len(predictor.layers) - 1 else "full"
            for i in range(len(predictor.layers))
        ]
        self.sliding_kv_layer = sliding_kv_layer
        self.full_kv_layer = full_kv_layer

    @classmethod
    def from_paths(
        cls,
        drafter_dir: str | Path,
        target_embed_weight: mx.array,
        target_embed_scale: float,
        target_config: dict,
    ) -> Gemma4AssistantProposer:
        """Build from the drafter checkpoint + the already-loaded target's
        embedding. Resolves the KV-share layer indices from the target config
        (last non-shared sliding / full layer)."""
        predictor, embedder, _ = load_assistant_drafter(drafter_dir)
        tcfg = target_config.get("text_config", target_config)
        layer_types = tcfg["layer_types"]
        n_shared = tcfg.get("num_kv_shared_layers", 0)
        non_shared = layer_types[: len(layer_types) - n_shared]
        last_by_type: dict[str, int] = {}
        for i, lt in enumerate(non_shared):
            last_by_type[lt] = i
        return cls(
            predictor=predictor,
            embedder=embedder,
            target_embed_weight=target_embed_weight,
            target_embed_scale=target_embed_scale,
            sliding_kv_layer=last_by_type.get("sliding_attention", 0),
            full_kv_layer=last_by_type.get("full_attention", 0),
        )

    def _kv_for_layers(
        self, kv_by_target_layer: dict[int, tuple[mx.array, mx.array]]
    ) -> list[tuple[mx.array, mx.array]]:
        """Map the drafter's per-layer KV from the target's shared layers."""
        sk = kv_by_target_layer[self.sliding_kv_layer]
        fk = kv_by_target_layer[self.full_kv_layer]
        return [sk if t == "sliding" else fk for t in self._layer_types]

    def _embed(self, token_id: int, dtype: mx.Dtype) -> mx.array:
        return (
            self._target_embed_weight[token_id].astype(dtype)
            * self._target_embed_scale
        )[None, :]  # (1, backbone)

    def propose_chain(
        self,
        last_token_id: int,
        target_last_hidden: mx.array,
        kv_by_target_layer: dict[int, tuple[mx.array, mx.array]],
        offset: int,
        k: int = 1,
        return_logits: bool = False,
    ) -> list[int] | tuple[list[int], list[mx.array]]:
        """Propose up to ``k`` draft tokens (EAGLE/MTP chain) from a single
        position ``offset`` (vLLM ``constant_draft_positions``).

        All draft steps stay at the same position and use the same target KV
        (0..offset, which exists); the chain advances via the drafter's own
        ``backbone_hidden`` feedback. The drafter's attention reads only the
        external target KV, so a single-row forward is exact (no need to carry
        the full context) once the RoPE ``offset`` is supplied.

        Args:
            last_token_id: most-recent accepted token (drafter's first input).
            target_last_hidden: target hidden at ``offset``, shape (1, backbone)
                or (backbone,).
            kv_by_target_layer: {target_layer_idx: (keys, values)}.
            offset: absolute position of the query (= context length - 1).
            k: number of draft tokens to chain.
        """
        h = target_last_hidden
        if h.ndim == 1:
            h = h[None]
        kv = self._kv_for_layers(kv_by_target_layer)
        prev = last_token_id
        drafts: list[int] = []
        logits_out: list[mx.array] = []
        lm = self._lm_head.astype(mx.float32)
        for _ in range(k):
            ie = self._embed(prev, h.dtype)  # (1, backbone)
            # mask=None: the single query attends to all of the target's cached
            # context (position `offset` legitimately sees 0..offset).
            draft_hidden, backbone_hidden = self.predictor(
                ie, h, kv, mask=None, offset=offset
            )
            dh = draft_hidden.astype(mx.float32)
            if return_logits:
                lg = self.embedder(dh, lm)[0]  # (vocab,) full-vocab, masked to -inf
                logits_out.append(lg)
                d = int(mx.argmax(lg).item())
            else:
                top = self.embedder.get_top_tokens(dh, lm)
                mx.eval(top)
                d = int(top[0].item())
            drafts.append(d)
            h = backbone_hidden  # feedback for the next chained draft
            prev = d
        if return_logits:
            return drafts, logits_out
        return drafts

    def propose(
        self,
        last_token_id: int,
        target_hidden: mx.array,
        kv_by_target_layer: dict[int, tuple[mx.array, mx.array]],
    ) -> int:
        """Propose a single next token. Accepts either the full-context hidden
        ``(T, backbone)`` (uses the last row at offset T-1) or a single row."""
        if target_hidden.ndim == 2 and target_hidden.shape[0] > 1:
            offset = target_hidden.shape[0] - 1
            last = target_hidden[-1:]
        else:
            offset = 0
            last = target_hidden if target_hidden.ndim == 2 else target_hidden[None]
        return self.propose_chain(
            last_token_id, last, kv_by_target_layer, offset, k=1
        )[0]

    def spec_decode_generate(
        self,
        target_inner,
        lm_head,
        cache: list,
        prompt_ids: list[int],
        max_tokens: int,
        k: int = 4,
        eos_ids: set[int] | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> list[int]:
        """Draft-verify-rollback spec-decode loop (validated 2.08×).

        The single reusable serving primitive promoted from the validation
        script. Each step: take the free target token t1, chain ``k`` drafts from
        the drafter at the constant position, verify [t1, *drafts] in ONE target
        forward, accept a prefix, and trim the target KV for the rejected tail.

        ``temperature == 0`` (default): greedy — accept the longest *exact* prefix
        where draft_j == target argmax_j. Output is exactly the target's greedy
        sequence (the validated 1.98-2.08× path).

        ``temperature > 0``: distribution-correct **speculative sampling**
        (Leviathan et al. 2023). t1 is sampled from the target; each draft_j is
        accepted with prob min(1, p_target_j(d_j)/q_draft_j(d_j)); on the first
        rejection a token is resampled from norm(relu(p_target − q_draft)) and the
        chain stops; if all k accept, a bonus token is sampled from the target.
        The emitted distribution equals plain target sampling at the same temp.

        Args:
            target_inner: the target's inner transformer, callable
                ``target_inner(ids, cache=) -> hidden (1, T, backbone)``.
            lm_head: callable ``lm_head(hidden) -> logits`` (e.g.
                ``embed_tokens.as_linear``).
            cache: the target's KV cache (``make_prompt_cache(target_wrapper)``);
                must expose per-layer ``.offset``, ``.state`` and ``.trim(n)``.
            prompt_ids: prompt token ids.
            max_tokens: number of tokens to generate.
            k: draft chain length per step.
            eos_ids: optional stop-token ids (generation halts after one).
            temperature: 0 = greedy; >0 = speculative sampling at that temperature.
            seed: optional RNG seed for reproducible sampling.
        """
        if seed is not None:
            mx.random.seed(seed)
        sl, fl = self.sliding_kv_layer, self.full_kv_layer
        eos = eos_ids or set()
        h = target_inner(mx.array(prompt_ids)[None], cache=cache)
        mx.eval(h)
        hidden_last = h[0, -1:]  # (1, backbone) at position P
        out: list[int] = []
        greedy = temperature <= 0.0
        while len(out) < max_tokens:
            offset = cache[fl].offset - 1
            t1_logits = lm_head(hidden_last[None])[0, -1]
            t1 = (int(mx.argmax(t1_logits)) if greedy
                  else int(mx.random.categorical(t1_logits / temperature).item()))
            # Sliding-window safety: the verify forward advances every
            # layer by 1+k. If that pushes the sliding RotatingKVCache (gemma window
            # = 512) past max_size it rotates, and a later trim of rejected positions
            # is UNSOUND — RotatingKVCache.trim() blindly decrements _idx into the
            # rotated ring buffer, misordering the sliding-layer KV. Once the context
            # nears the window, fall back to a plain single-token step (advance by 1,
            # accept it, no over-advance, no trim) for the rest of the generation.
            _sl_cache = cache[sl]
            if (hasattr(_sl_cache, "max_size")
                    and (_sl_cache.offset + 1 + k) >= _sl_cache.max_size):
                out.append(t1)
                if t1 in eos:
                    break
                _bh = target_inner(mx.array([[t1]]), cache=cache)
                mx.eval(_bh)
                hidden_last = _bh[0, -1:]
                continue
            kv = {sl: cache[sl].state, fl: cache[fl].state}
            if greedy:
                drafts = self.propose_chain(t1, hidden_last, kv, offset, k=k)
            else:
                drafts, draft_logits = self.propose_chain(
                    t1, hidden_last, kv, offset, k=k, return_logits=True
                )
            vhid = target_inner(mx.array([[t1, *drafts]]), cache=cache)
            mx.eval(vhid)
            tlogits = lm_head(vhid)[0]  # (1+k, vocab) target dist after each pos
            mx.eval(tlogits)
            emitted = [t1]
            if greedy:
                targ = mx.argmax(tlogits, axis=-1)
                a = 0
                for j in range(k):
                    if drafts[j] == int(targ[j].item()):
                        a += 1
                    else:
                        break
                emitted.extend(drafts[:a])
                hidden_carry_idx = a
            else:
                a = 0
                bonus: int | None = None
                for j in range(k):
                    p = mx.softmax(tlogits[j] / temperature, axis=-1)
                    q = mx.softmax(draft_logits[j] / temperature, axis=-1)
                    pj = float(p[drafts[j]].item())
                    qj = float(q[drafts[j]].item())
                    if float(mx.random.uniform().item()) < min(1.0, pj / (qj + 1e-9)):
                        a += 1
                    else:
                        resid = mx.maximum(p - q, 0.0)
                        s = float(resid.sum().item())
                        resid = resid / s if s > 1e-9 else p
                        bonus = int(mx.random.categorical(mx.log(resid + 1e-20)).item())
                        break
                emitted.extend(drafts[:a])
                if bonus is None:  # all k accepted -> bonus from target at last pos
                    bonus = int(mx.random.categorical(tlogits[k] / temperature).item())
                emitted.append(bonus)
                hidden_carry_idx = a  # hidden after the last accepted draft
            # Stop at the FIRST eos in this step. The draft/verify chain can accept
            # tokens AFTER an eos (t1==eos with drafts behind it, or a draft==eos),
            # and the sampled bonus can sit past one too. Truncate here so they don't
            # leak: the caller only strips a TRAILING eos and would otherwise emit the
            # post-eos tokens as visible output with finish_reason "length" not "stop".
            _eos_pos = next((i for i, tok in enumerate(emitted) if tok in eos), None)
            if _eos_pos is not None:
                emitted = emitted[: _eos_pos + 1]
            out.extend(emitted)
            if _eos_pos is not None:
                break  # done — skip the cache trim/hidden carry (no next step)
            # trim KV the verify forward advanced past what we accepted:
            # verify added (1+k) tokens; we keep (1 + a) [+1 bonus carried via re-feed].
            keep = 1 + a
            advanced = 1 + k
            if greedy:
                if advanced - keep > 0:
                    for c in cache:
                        c.trim(advanced - keep)
                hidden_last = vhid[0, hidden_carry_idx : hidden_carry_idx + 1]
            else:
                # sampled path: the bonus token is NOT in the verify forward, so
                # re-feed it to advance the cache and get its hidden state.
                if advanced - keep > 0:
                    for c in cache:
                        c.trim(advanced - keep)
                bh = target_inner(mx.array([[emitted[-1]]]), cache=cache)
                mx.eval(bh)
                hidden_last = bh[0, -1:]
        return out[:max_tokens]
