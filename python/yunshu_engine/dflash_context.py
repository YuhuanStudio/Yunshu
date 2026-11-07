# Upstream (derived): Blaizzy/mlx-vlm (MIT) mlx_vlm/speculative/utils.py, mlx_vlm/models/qwen3_5/language.py, mlx_vlm/speculative/drafters/qwen3_dflash/dflash.py @ v0.7.3
# Patches upstream mlx-vlm symbols (see vendor.json kind=patches; `just vendor-check` flags source changes)
"""DFlash speculative decoding on Qwen3.5: lossless prefill, verify head, and
the drafter's context window.

1. (Removed with the stock runner.) A one-row ``BatchKVCache`` prefill with captured layers used to
   need a per-row routing patch of ``Qwen3_5Model.__call__``; the round driver prefills every row on
   its own plain ``KVCache`` and passes ``capture_layer_ids`` / ``hidden_sink`` straight to the model.
2. Greedy verify head. mlx-vlm's greedy DFlash round asks the target for
   ``speculative_verify_dflash_hidden`` (captured layers + final hidden) and
   takes the argmax with ``speculative_argmax_from_hidden`` -- the head MTP
   verify uses. Qwen3.5 lacks the hook, so DFlash computed T x vocab verify
   logits through the verifier's own head matmul. ``install`` adds it (as
   upstream Nemotron-H has).
3. Drafter context window. When every drafter layer is a sliding-window layer,
   the drafter attends to at most ``sliding_window - 1`` context positions.
   Upstream trims to that window inside each attention layer, but only after
   the drafter's input projection (``fc``: 5 x hidden -> hidden) and the fused
   context K/V projection ran over the whole prompt: ~48 TFLOP in the first
   speculative round at 131K tokens on Qwen3.8-27B, plus ~6.7 GiB of captured
   hidden states (5 layers x 131K x 5120 x bf16) held through prefill.
   - ``DFlashDraftModel._hidden`` trims the target hidden states to the window
     before the projections and advances each drafter cache by the skipped
     count (what the attention layers do after projecting), so draft keys keep
     their absolute RoPE positions.
   - ``SpeculativePrefill`` retains only the chunks covering the last window of
     captured hidden states. The dropped positions shift every drafter position
     by the same constant, which RoPE attention does not see.
   Drafts can differ from upstream's only by rounding (matmuls over fewer
   rows); the target verifies every draft, so the output does not depend on it.
"""

from __future__ import annotations

from typing import Any

_STATE: dict = {"installed": False}


def context_window(drafter: Any) -> int | None:
    """Context positions a DFlash drafter can attend to, or None when some
    layer is full attention (then it reads the whole prompt)."""
    config = getattr(drafter, "config", None)
    layer_types = list(getattr(config, "layer_types", None) or [])
    window = getattr(config, "sliding_window", None)
    if not layer_types or window is None:
        return None
    if any(t != "sliding_attention" for t in layer_types):
        return None
    return max(1, int(window) - 1)


def _speculative_verify_dflash_hidden(self, inputs, cache, capture_layer_ids):
    out = self(
        inputs,
        cache=cache,
        capture_layer_ids=capture_layer_ids,
        speculative_verify=True,
        return_hidden=True,
        skip_logits=True,
    )
    return out.hidden_states[:-1], out.hidden_states[-1], out.gdn_states


def neutral_context(drafter: Any):
    """One all-zero context position for a drafter that has seen no target hidden states.

    A row promoted onto the lane from the shared batch has no captured states until its first
    verify. Drafts only propose (the target verifies every one), so a neutral context costs
    acceptance for that round and never a token; later rounds carry the real states."""
    import mlx.core as mx

    width = len(drafter.config.target_layer_ids) * int(drafter.config.hidden_size)
    return mx.zeros((1, 1, width), dtype=mx.bfloat16)


def _masked_attention(
    self,
    x,
    x_ctx,
    rope,
    cache,
    projected_context=None,
):
    import mlx.core as mx

    batch, rows, _ = x.shape
    context_rows = x_ctx.shape[1]
    if self.is_sliding:
        if self.sliding_window is None:
            raise ValueError(
                "DFlash draft config must define sliding_window for sliding layers."
            )
        keep_ctx = self.sliding_window - 1
        if context_rows > keep_ctx:
            skip = context_rows - keep_ctx
            x_ctx = x_ctx[:, skip:]
            if projected_context is not None:
                projected_context = tuple(
                    value[:, skip:] for value in projected_context
                )
            context_rows = x_ctx.shape[1]
            cache.offset += skip

    # Project context and proposal separately so only context KV
    queries = self.q_proj(x)
    ctx_keys, ctx_values = (
        self._project_kv(x_ctx) if projected_context is None else projected_context
    )
    prop_keys, prop_values = self._project_kv(x)
    queries = self.q_norm(queries.reshape(batch, rows, self.n_heads, -1)).transpose(
        0, 2, 1, 3
    )
    ctx_keys = self.k_norm(
        ctx_keys.reshape(batch, context_rows, self.n_kv_heads, -1)
    ).transpose(0, 2, 1, 3)
    ctx_values = ctx_values.reshape(batch, context_rows, self.n_kv_heads, -1).transpose(
        0, 2, 1, 3
    )
    prop_keys = self.k_norm(
        prop_keys.reshape(batch, rows, self.n_kv_heads, -1)
    ).transpose(0, 2, 1, 3)
    prop_values = prop_values.reshape(batch, rows, self.n_kv_heads, -1).transpose(
        0, 2, 1, 3
    )
    queries = rope(queries, offset=cache.offset + context_rows)
    ctx_keys = rope(ctx_keys, offset=cache.offset)
    prop_keys = rope(prop_keys, offset=cache.offset + context_rows)
    keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
    # A one-row append can leave RotatingKVCache physically rotated. The TF
    # sliding mask indexes chronological keys, unlike unmasked attention.
    from mlx_vlm.models.cache import BufferedRotatingKVCache, RotatingKVCache

    if isinstance(cache, RotatingKVCache) and (
        not isinstance(cache, BufferedRotatingKVCache) or cache.keep
    ):
        keys, values = cache._temporal_order(keys), cache._temporal_order(values)
    keys = mx.concatenate([keys, prop_keys], axis=2)
    values = mx.concatenate([values, prop_values], axis=2)
    # TensorFold 1.2.0 dflash/runtime_context.zig:137-146.
    from .dflash_ring import block_mask

    context = int(keys.shape[2]) - rows
    mask = block_mask(
        mx.arange(context, dtype=mx.int32),
        mx.array(context),
        rows,
        self._yunshu_trained_block,
        self.sliding_window,
    )
    o = mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=self.scale,
        mask=mask,
        sinks=self.attention_sink_bias,
    )
    return self.o_proj(o.transpose(0, 2, 1, 3).reshape(batch, rows, -1))


def _install_context_window() -> None:
    from mlx_vlm.speculative import utils as spec_utils
    from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import (
        DFlashAttention,
        DFlashDraftModel,
    )

    # Existing and future DFlash2 layers inherit this attention entry point.
    original_attention = DFlashAttention.__call__

    def attention(self, *args, **kwargs):
        if not hasattr(self, "_yunshu_trained_block"):
            return original_attention(self, *args, **kwargs)
        return _masked_attention(self, *args, **kwargs)

    DFlashAttention.__call__ = attention
    orig_hidden = DFlashDraftModel._hidden

    def _hidden(self, inputs, target_hidden, cache):
        for layer in self.layers:
            object.__setattr__(
                layer.self_attn, "_yunshu_trained_block", int(self.config.block_size)
            )
        if target_hidden is None:
            target_hidden = neutral_context(self)
        keep = context_window(self)
        length = int(target_hidden.shape[1])
        if keep is not None and length > keep:
            skip = length - keep
            target_hidden = target_hidden[:, skip:]
            for c in cache:
                c.offset += skip
        return orig_hidden(self, inputs, target_hidden, cache)

    DFlashDraftModel._hidden = _hidden

    prefill_cls = spec_utils.SpeculativePrefill
    orig_init = prefill_cls.__init__
    orig_append = prefill_cls.append

    def __init__(self, draft_kind, drafter):
        orig_init(self, draft_kind, drafter)
        self._yunshu_keep = (
            context_window(drafter)
            if draft_kind == "dflash" and drafter is not None
            else None
        )

    def append(self, output):
        orig_append(self, output)
        keep = getattr(self, "_yunshu_keep", None)
        if keep is None or not self.chunks:
            return
        # The final chunk (``finish``) adds its own positions, so the window
        # is covered as long as the retained chunks hold ``keep`` positions.
        retained = 0
        for index in range(len(self.chunks) - 1, -1, -1):
            retained += int(self.chunks[index][0].shape[1])
            if retained >= keep:
                if index:
                    del self.chunks[:index]
                break

    prefill_cls.__init__ = __init__
    prefill_cls.append = append


def install(language_model: Any = None) -> bool:
    """Install the patches (idempotent). ``language_model``'s class gets
    the greedy DFlash verify hook when it verifies through
    ``speculative_argmax_from_hidden`` but lacks it."""
    cls = type(language_model) if language_model is not None else None
    if (
        cls is not None
        and callable(getattr(cls, "speculative_argmax_from_hidden", None))
        and not hasattr(cls, "speculative_verify_dflash_hidden")
    ):
        cls.speculative_verify_dflash_hidden = _speculative_verify_dflash_hidden
    if _STATE["installed"]:
        return True
    _install_context_window()
    try:
        from . import dflash_apc_window, settings

        dflash_apc_window.set_budget(
            int(float(settings.get("YUNSHU_VLM_APC_DRAFT_WINDOW_GB") or 0) * (1 << 30))
        )
    except Exception:  # lossless fallback: suffix-only drafter context after a hit
        import logging

        logging.getLogger(__name__).warning(
            "drafter window store not installed", exc_info=True
        )
    _STATE["installed"] = True
    return True


__all__ = ["context_window", "install"]
