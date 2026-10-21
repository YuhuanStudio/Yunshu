"""DFlash2 drafter context in fixed-width rings, so a round's draft needs no host numbers.

The vendor drafter appends each round's accepted target rows to a rotating KV cache whose
length, rope offsets and window crop are host integers: the next draft cannot be built until
the verify is read back. Here every layer keeps its context keys/values in a ring of ``CAP``
slots and a per-slot absolute position; a round writes a fixed ``W`` rows at the GPU-held
position, marks only the first ``count`` of them valid, and the proposal rows attend to the
ring through a mask built on the GPU (the last ``keep`` valid context rows, as the vendor's
sliding window keeps). With the accepted path, its count and the bonus token computed on the
GPU (``dflash_plan.walk_gpu``), the next round's draft can be queued behind the verify.

Drafts only propose (the target verifies every row), so this path changes which tokens are
proposed at most through rounding, never an emitted token.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

INVALID = -(1 << 30)


class Ring:
    """Context keys/values of every drafter layer, ``cap`` slots each."""

    def __init__(self, drafter: Any, cap: int, keep: int):
        cfg = drafter.config
        kvh, hd = int(cfg.num_key_value_heads), int(cfg.head_dim)
        self.cap, self.keep = int(cap), int(keep)
        if self.keep >= self.cap:
            raise ValueError("ring must hold more slots than the window it keeps")
        self.keys = [
            mx.zeros((1, kvh, self.cap, hd), dtype=mx.bfloat16) for _ in drafter.layers
        ]
        self.values = [
            mx.zeros((1, kvh, self.cap, hd), dtype=mx.bfloat16) for _ in drafter.layers
        ]
        self.slot_pos = mx.full((self.cap,), INVALID, dtype=mx.int32)
        self.pos = mx.zeros((1,), dtype=mx.int32)  # context rows appended so far
        self.empty = True  # the first append may fill every slot: no live row to keep

    def append(self, drafter: Any, target_hidden: mx.array, count: mx.array) -> None:
        """Append the first ``count`` of ``target_hidden``'s rows ([1, W, D]) as context."""
        width = int(target_hidden.shape[1])
        if width > (self.cap if self.empty else self.cap - self.keep):
            raise ValueError(f"{width} rows do not fit beside a {self.keep}-row window")
        self.empty = False
        h_ctx = drafter.hidden_norm(drafter.fc(target_hidden))
        offsets = mx.arange(width, dtype=mx.int32)
        absolute = self.pos + offsets
        slots = absolute % self.cap
        valid = offsets < count.reshape(())
        for i, layer in enumerate(drafter.layers):
            attn = layer.self_attn
            k, v = attn._project_kv(h_ctx)
            k = attn.k_norm(k.reshape(1, width, attn.n_kv_heads, -1)).transpose(
                0, 2, 1, 3
            )
            v = v.reshape(1, width, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
            k = drafter.rope(k, offset=self.pos.reshape(()))
            # rows past ``count`` land in slots no valid row holds yet; the next round
            # overwrites them from its own position
            self.keys[i][:, :, slots, :] = k.astype(mx.bfloat16)
            self.values[i][:, :, slots, :] = v.astype(mx.bfloat16)
        self.slot_pos[slots] = mx.where(valid, absolute, INVALID)
        self.pos = self.pos + count.reshape(1).astype(mx.int32)

    def mask(self, rows: int) -> mx.array:
        """Additive mask [1, 1, 1, cap + rows]: the last ``keep`` context rows and every
        proposal row (the drafter's attention is not causal)."""
        live = (self.slot_pos >= self.pos - self.keep) & (self.slot_pos >= 0)
        ring = mx.where(live, 0.0, -mx.inf).astype(mx.bfloat16)
        props = mx.zeros((rows,), dtype=mx.bfloat16)
        return mx.concatenate([ring, props]).reshape(1, 1, 1, -1)


def _attend(attn: Any, x: mx.array, keys: mx.array, values: mx.array, rope, pos, mask):
    """The vendor DFlashAttention for the proposal rows over the ring's context."""
    b, n, _ = x.shape
    q = attn.q_proj(x)
    pk, pv = attn._project_kv(x)
    q = attn.q_norm(q.reshape(b, n, attn.n_heads, -1)).transpose(0, 2, 1, 3)
    pk = attn.k_norm(pk.reshape(b, n, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    pv = pv.reshape(b, n, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    q = rope(q, offset=pos)
    pk = rope(pk, offset=pos)
    k = mx.concatenate([keys, pk.astype(keys.dtype)], axis=2)
    v = mx.concatenate([values, pv.astype(values.dtype)], axis=2)
    o = mx.fast.scaled_dot_product_attention(
        q.astype(k.dtype),
        k,
        v,
        scale=attn.scale,
        mask=mask,
        sinks=attn.attention_sink_bias,
    )
    return attn.o_proj(o.transpose(0, 2, 1, 3).reshape(b, n, -1))


def draft_hidden(drafter: Any, ring: Ring, bonus: mx.array, positions: int) -> Any:
    """The proposal rows' hidden states ([1, positions, H]) after ``bonus`` ([1] token)."""
    cfg = drafter.config
    masks = mx.full((positions,), int(cfg.mask_token_id), dtype=mx.int32)
    inputs = mx.concatenate([bonus.reshape(1).astype(mx.int32), masks])[None]
    h = drafter._embed_input_tokens(inputs)
    rows = int(inputs.shape[1])
    pos = ring.pos.reshape(())
    mask = ring.mask(rows)
    for i, layer in enumerate(drafter.layers):
        h = _layer(layer, h, ring.keys[i], ring.values[i], drafter.rope, pos, mask)
    return drafter.norm(h)[:, 1:]


def _layer(layer, x, keys, values, rope, pos, mask):
    compiled = getattr(layer, "_compiled", None)
    if compiled is not None:  # dflash_tree.compiled_layer_class: compiled halves
        pre, mid, post = compiled()
        prepared, kernel = pre(x)
        attended = _attend(layer.self_attn, prepared, keys, values, rope, pos, mask)
        x1, prepared2, kernel2 = mid(x, attended, kernel)
        return post(x1, layer.mlp(prepared2), kernel2)
    residual = x
    prepared, kernel = layer.attention_conv.prepare(layer.input_layernorm(x))
    attended = _attend(layer.self_attn, prepared, keys, values, rope, pos, mask)
    x = residual + layer.attention_conv.finish(attended, kernel)
    residual = x
    prepared, kernel = layer.mlp_conv.prepare(layer.post_attention_layernorm(x))
    return residual + layer.mlp_conv.finish(layer.mlp(prepared), kernel)
