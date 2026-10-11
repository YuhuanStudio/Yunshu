"""Metal dispatches per op inside one DFlash drafter layer (run with the metalrt interposer loaded)."""

import json
import sys
from pathlib import Path

DRAFTER = "/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2"


def main() -> int:
    import mlx.core as mx
    import mlx.nn as nn

    sys.path.insert(0, "python")
    from mlx_vlm.speculative.drafters.qwen3_dflash import dflash as v
    from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import (
        DFlashConfig,
        DFlashDraftModel,
    )

    from yunshu_engine import gpu_window
    from yunshu_engine.dflash_tree import (
        quantize_drafter,
        share_masks,
        stack_projections,
    )

    if gpu_window._lib() is None:
        print("interposer not loaded")
        return 2
    cfg = DFlashConfig.from_dict(
        json.loads((Path(DRAFTER) / "config.json").read_text())
    )
    m = DFlashDraftModel(cfg)
    m.embed_tokens = nn.Embedding(1000, cfg.hidden_size)
    mx.eval(m.parameters())
    quantize_drafter(m, 8)
    share_masks()
    stack_projections(m)
    cache = m.make_cache()
    ids = mx.array([[5] + [3] * 15], dtype=mx.int32)
    ctx = mx.random.normal((1, 12, len(cfg.target_layer_ids) * cfg.hidden_size)).astype(
        mx.bfloat16
    )
    mx.eval(ids, ctx)
    for _ in range(3):  # warm: fills caches, kernels
        mx.eval(m._hidden(ids, ctx, cache))
    out = {}

    def step(name, *arrays):
        mx.eval(arrays)
        out[name] = out.get(name, 0) + gpu_window.stage_end()

    layer, attn = m.layers[0], m.layers[0].self_attn
    gpu_window.stage_begin()
    x = m._embed_input_tokens(ids)
    step("embed", x)
    h_ctx = m.hidden_norm(m.fc(ctx))
    step("fc+hidden_norm", h_ctx)
    normed = layer.input_layernorm(x)
    step("input_norm", normed)
    B, L, _ = normed.shape
    S = h_ctx.shape[1]
    q = attn.q_proj(normed)
    step("q_proj", q)
    ck, cv = attn._project_kv(h_ctx)
    step("kv_ctx", ck, cv)
    pk, pv = attn._project_kv(normed)
    step("kv_prop", pk, pv)
    q = attn.q_norm(q.reshape(B, L, attn.n_heads, -1)).transpose(0, 2, 1, 3)
    step("q_norm", q)
    ck = attn.k_norm(ck.reshape(B, S, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    cv = cv.reshape(B, S, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    step("k_norm_ctx", ck, cv)
    pk = attn.k_norm(pk.reshape(B, L, attn.n_kv_heads, -1)).transpose(0, 2, 1, 3)
    pv = pv.reshape(B, L, attn.n_kv_heads, -1).transpose(0, 2, 1, 3)
    step("k_norm_prop", pk, pv)
    off = cache[0].offset
    q = m.rope(q, offset=off + S)
    ck = m.rope(ck, offset=off)
    pk = m.rope(pk, offset=off + S)
    step("rope", q, ck, pk)
    keys, values = cache[0].update_and_fetch(ck, cv)
    step("cache_update", keys, values)
    keys = mx.concatenate([keys, pk], axis=2)
    values = mx.concatenate([values, pv], axis=2)
    step("concat_kv", keys, values)
    mask = v.create_causal_mask(
        L, offset=keys.shape[2] - L, window_size=attn.sliding_window
    )
    step("mask", *([mask] if hasattr(mask, "shape") else []))
    o = mx.fast.scaled_dot_product_attention(
        q, keys, values, scale=attn.scale, mask=mask
    )
    step("sdpa", o)
    o = attn.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, -1))
    step("o_proj", o)
    h = x + o
    step("residual1", h)
    n2 = layer.post_attention_layernorm(h)
    step("post_norm", n2)
    mlp_out = layer.mlp(n2)
    step("mlp", mlp_out)
    step("residual2", h + mlp_out)
    out["total"] = sum(out.values())
    for rows in (4, 12):
        c = mx.random.normal(
            (1, rows, len(cfg.target_layer_ids) * cfg.hidden_size)
        ).astype(mx.bfloat16)
        mx.eval(c)
        mx.eval(m._hidden(ids, c, cache))  # warm this shape
        mx.eval(c)
        gpu_window.stage_begin()
        mx.eval(m._hidden(ids, c, cache))
        out[f"whole_hidden_ctx{rows}"] = gpu_window.stage_end()
    from yunshu_engine import dflash_fast

    private = dflash_fast.prepare(m)
    private.bind(True)
    for rows in (4,):
        c = mx.random.normal(
            (1, rows, len(cfg.target_layer_ids) * cfg.hidden_size)
        ).astype(mx.bfloat16)
        mx.eval(c)
        mx.eval(m._hidden(ids, c, cache))
        mx.eval(c)
        gpu_window.stage_begin()
        mx.eval(m._hidden(ids, c, cache))
        out[f"whole_hidden_lane_ctx{rows}"] = gpu_window.stage_end()
    private.bind(False)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
