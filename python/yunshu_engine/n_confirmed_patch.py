"""n_confirmed support for GatedDeltaNet — enables zero-cost reject in MTP.

When the model forward processes S=2 tokens with n_confirmed=1, the SSM
layers process tokens one-at-a-time via the ops-based implementation and
save state after the first (confirmed) token as rollback_state. This makes
reject cost = accept cost = 1 backbone forward + 1 MTP forward.

Without n_confirmed:
  Accept: 1 backbone(2tok) + 1 MTP = 1.15x
  Reject: 1 backbone(2tok) + 1 backbone(1tok restore+refeed) + 1 MTP = 2.15x

With n_confirmed=1:
  Accept: 1 backbone(2tok) + 1 MTP = 1.15x
  Reject: 1 backbone(2tok) + restore_rollback + 1 MTP = 1.15x

The speedup formula changes from:
  old: speedup = (1+p) / (1.15 + (1-p)*1.0) → negative for p < 0.735
  new: speedup = (1+p) / 1.15 → positive for any p > 0.15

Based on oMLX PR 990 pattern. First standalone implementation outside oMLX.
"""

import logging
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.gated_delta import gated_delta_update

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_n_confirmed_patch() -> bool:
    """Apply n_confirmed patches. Idempotent."""
    global _PATCHED
    if _PATCHED:
        return True

    try:
        from mlx_lm.models import qwen3_5 as q35
        from mlx_lm.models.cache import ArraysCache
    except ImportError:
        logger.debug("mlx_lm.models.qwen3_5 not importable; skipping n_confirmed patch")
        return False

    if hasattr(q35.GatedDeltaNet, "_yunshu_n_confirmed_patched"):
        _PATCHED = True
        return True

    # 1. Add rollback_state to ArraysCache
    if not hasattr(ArraysCache, "rollback_state"):
        ArraysCache.rollback_state = None
        logger.debug("Added rollback_state to ArraysCache")

    # 2. Patch GatedDeltaNet.__call__
    _patch_gated_delta(q35)

    # 3. Patch Qwen3_5TextModel.__call__ to pass n_confirmed through
    _patch_text_model(q35)

    _PATCHED = True
    logger.info("n_confirmed patch applied")
    return True


def _patch_gated_delta(q35: Any) -> None:
    cls = q35.GatedDeltaNet
    if "_yunshu_n_confirmed_patched" in cls.__dict__:
        return

    original_call = cls.__call__

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        n_confirmed: int = 0,
    ) -> mx.array:
        B, S, _ = inputs.shape

        # Fast path: no n_confirmed, use original implementation
        if n_confirmed <= 0 or n_confirmed >= S or S <= 1:
            return original_call(self, inputs, mask=mask, cache=cache)

        # n_confirmed path: process tokens one-at-a-time, saving rollback state
        # Only used for SSM (linear attention) layers with ArraysCache
        is_ssm = cache is not None and hasattr(cache, "cache") and isinstance(getattr(cache, "cache", None), list)
        if not is_ssm:
            return original_call(self, inputs, mask=mask, cache=cache)

        return _forward_n_confirmed(self, inputs, mask, cache, n_confirmed, B, S)

    cls.__call__ = __call__
    cls._yunshu_n_confirmed_patched = True


def _forward_n_confirmed(
    self,
    inputs: mx.array,
    mask: Optional[mx.array],
    cache: Any,
    n_confirmed: int,
    B: int,
    S: int,
) -> mx.array:
    """GatedDeltaNet forward with n_confirmed support.

    Splits processing into two chunks:
    1. Confirmed chunk (tokens 0..n_confirmed-1): processes with its own conv state
    2. Draft chunk (tokens n_confirmed..S-1): processes from confirmed chunk's output state

    After confirmed chunk, saves (conv_c, ssm_c) as rollback_state.
    On accept: caller clears rollback_state.
    On reject: caller restores rollback_state, reverting to confirmed state.

    Based on oMLX's _process_chunk pattern.
    """
    from mlx_lm.models.gated_delta import gated_delta_update

    if self.sharding_group is not None:
        from mlx_lm.models.gated_delta import sum_gradients
        inputs = sum_gradients(self.sharding_group)(inputs)

    # Projections over all S tokens
    qkv = self.in_proj_qkv(inputs)
    z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
    b = self.in_proj_b(inputs)
    a = self.in_proj_a(inputs)

    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)

    # Get initial states
    conv_state = cache[0] if cache is not None and cache[0] is not None else mx.zeros(
        (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype,
    )
    ssm_state = cache[1] if cache else None

    # Split qkv, a, b, mask into confirmed and draft chunks
    qkv_c = qkv[:, :n_confirmed]
    qkv_d = qkv[:, n_confirmed:]
    a_c = a[:, :n_confirmed]
    a_d = a[:, n_confirmed:]
    b_c = b[:, :n_confirmed]
    b_d = b[:, n_confirmed:]
    mask_c = mask[:, :n_confirmed] if mask is not None else None
    mask_d = mask[:, n_confirmed:] if mask is not None else None

    # --- Process confirmed chunk ---
    out_c, conv_c, ssm_c = _process_chunk(
        self, qkv_c, a_c, b_c, conv_state, ssm_state, mask_c,
    )

    # Save rollback state after confirmed chunk
    if cache is not None:
        cache.rollback_state = (conv_c, ssm_c)

    # --- Process draft chunk from confirmed chunk's state ---
    out_d, conv_f, ssm_f = _process_chunk(
        self, qkv_d, a_d, b_d, conv_c, ssm_c, mask_d,
    )

    # Concatenate outputs and update cache
    out = mx.concatenate([out_c, out_d], axis=1)
    z_full = z  # already [B, S, Hv, Dv]

    if cache is not None:
        cache[0] = conv_f
        cache[1] = ssm_f
        cache.advance(S)

    out = self.norm(out, z_full)
    out = self.out_proj(out.reshape(B, S, -1))

    if self.sharding_group is not None:
        out = mx.distributed.all_sum(out, group=self.sharding_group)

    return out


def _process_chunk(
    self,
    qkv_chunk: mx.array,
    a_chunk: mx.array,
    b_chunk: mx.array,
    conv_state: mx.array,
    ssm_state: Optional[mx.array],
    ssm_mask: Optional[mx.array],
) -> tuple:
    """Process a chunk of tokens through conv + SSM, returning output and states."""
    B, S_chunk = qkv_chunk.shape[:2]

    # Conv state handling
    conv_in = mx.concatenate([conv_state, qkv_chunk], axis=1)
    n_keep = self.conv_kernel_size - 1
    new_conv_state = mx.contiguous(conv_in[:, -n_keep:, :])

    conv_out = nn.silu(self.conv1d(conv_in))

    q, k, v = [
        t.reshape(B, S_chunk, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
            [self.num_k_heads, self.num_k_heads, self.num_v_heads],
            [self.head_k_dim, self.head_k_dim, self.head_v_dim],
        )
    ]

    inv_scale = k.shape[-1] ** -0.5
    q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

    out, new_ssm_state = gated_delta_update(
        q, k, v, a_chunk, b_chunk, self.A_log, self.dt_bias,
        ssm_state, ssm_mask, use_kernel=True,
    )
    return out, new_conv_state, new_ssm_state


def _patch_text_model(q35: Any) -> None:
    """Patch Qwen3_5TextModel and DecoderLayer to pass n_confirmed through."""
    # Patch DecoderLayer first so it accepts n_confirmed kwarg
    _patch_decoder_layer(q35)

    # Then patch Qwen3_5TextModel to pass n_confirmed through
    cls = q35.Qwen3_5TextModel
    if "_yunshu_n_confirmed_patched" in cls.__dict__:
        return

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings=None,
        n_confirmed: int = 0,
    ):
        if input_embeddings is not None:
            hidden_states = input_embeddings
        else:
            hidden_states = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = q35.create_attention_mask(hidden_states, cache[self.fa_idx])
        ssm_mask = q35.create_ssm_mask(hidden_states, cache[self.ssm_idx])

        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            hidden_states = layer(hidden_states, mask=mask, cache=c, n_confirmed=n_confirmed)

        return self.norm(hidden_states)

    cls.__call__ = __call__
    cls._yunshu_n_confirmed_patched = True


def _patch_decoder_layer(q35: Any) -> None:
    """Patch DecoderLayer to pass n_confirmed to linear attention layers."""
    cls = q35.DecoderLayer
    if "_yunshu_n_confirmed_patched" in cls.__dict__:
        return

    original_call = cls.__call__

    def __call__(self, x, mask=None, cache=None, n_confirmed: int = 0):
        if self.is_linear:
            r = self.linear_attn(
                self.input_layernorm(x), mask, cache, n_confirmed=n_confirmed,
            )
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        return h + self.mlp(self.post_attention_layernorm(h))

    cls.__call__ = __call__
    cls._yunshu_n_confirmed_patched = True


def clear_rollback(cache: list) -> None:
    """Clear all rollback states after an accepted draft."""
    for c in cache:
        if hasattr(c, "rollback_state") and c.rollback_state is not None:
            c.rollback_state = None


def restore_rollback(cache: list) -> bool:
    """Restore all rollback states after a rejected draft.

    For SSM layers: restore conv_state and ssm_state from rollback snapshot.
    For KV layers: trim by 1.
    Returns False if any layer can't be rolled back.
    """
    success = True
    for c in cache:
        if hasattr(c, "rollback_state") and c.rollback_state is not None:
            conv_snap, ssm_snap = c.rollback_state
            c[0] = conv_snap
            c[1] = ssm_snap
            c.rollback_state = None
            if hasattr(c, "lengths") and c.lengths is not None:
                c.lengths = c.lengths - 1
            continue
        if hasattr(c, "is_trimmable") and c.is_trimmable():
            c.trim(1)
            continue
        success = False
    return success
