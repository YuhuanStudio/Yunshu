"""Which op of layer 0 / layer 3 breaks span invariance? Compares the first
512 tokens of a span-512 run with a span-2048 run, op by op."""

import random
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def eq(name, a, b):
    print(f"{name}: equal={bool(mx.array_equal(a, b).item())}", flush=True)


def main():
    from mlx_vlm import load as vlm_load

    from yunshu_engine.kernels import lane_linear
    from yunshu_engine.round_driver import forward as F

    model, _ = vlm_load(sys.argv[1])
    lm = model.language_model
    lane_linear.convert(lm)
    rng = random.Random(0)
    ids = mx.array([rng.randrange(1000, 20000) for _ in range(2048)], dtype=mx.int32)
    inner = lm.model
    for li in (0, 3):
        layer = inner.layers[li]
        x = inner.embed_tokens(ids)[None]
        xn = layer.input_layernorm(x)
        lin = (
            layer.linear_attn.in_proj_qkv if layer.is_linear else layer.self_attn.q_proj
        )
        big = lin.prefill([xn])[0]
        small = lin.prefill([xn[:, :512]])[0]
        mx.eval(big, small)
        eq(f"L{li} proj stock 2048 vs 512 (first 512)", big[:, :512], small)
        eq(
            f"L{li} proj stock 2048 vs 1024",
            big[:, :512],
            lin.prefill([xn[:, :1024]])[0][:, :512],
        )
        eq(
            f"L{li} proj 512@512 vs in-2048",
            lin.prefill([xn[:, 512:1024]])[0],
            big[:, 512:1024],
        )
        if layer.is_linear:
            g = layer.linear_attn

            def gdn(S, g=g, li=li, xn=xn):
                cache = lm.make_cache()[li]

                def p(m):
                    return m.prefill([xn[:, :S]])[0]

                qkv, z, b, a = (
                    p(g.in_proj_qkv),
                    p(g.in_proj_z),
                    p(g.in_proj_b),
                    p(g.in_proj_a),
                )
                out = F._gdn_mix(g, qkv, z, b, a, cache)
                mx.eval(out)
                return out, qkv

            def parts(S, g=g, xn=xn):
                p = lambda m: m.prefill([xn[:, :S]])[0]  # noqa: E731
                qkv, z, b, a = (
                    p(g.in_proj_qkv),
                    p(g.in_proj_z),
                    p(g.in_proj_b),
                    p(g.in_proj_a),
                )
                zc = mx.zeros((1, g.conv_kernel_size - 1, g.conv_dim), dtype=qkv.dtype)
                ci = mx.concatenate([zc, qkv], axis=1)
                co = nn.silu(g.conv1d(ci))
                q, k, v = [
                    t.reshape(1, S, h, d)
                    for t, h, d in zip(
                        mx.split(co, [g.key_dim, 2 * g.key_dim], -1),
                        [g.num_k_heads, g.num_k_heads, g.num_v_heads],
                        [g.head_k_dim, g.head_k_dim, g.head_v_dim],
                        strict=True,
                    )
                ]
                inv = k.shape[-1] ** -0.5
                q = (inv**2) * mx.fast.rms_norm(q, None, 1e-6)
                k = inv * mx.fast.rms_norm(k, None, 1e-6)
                from mlx_vlm.models.qwen3_5.gated_delta import gated_delta_update

                out, st = gated_delta_update(
                    q, k, v, a, b, g.A_log, g.dt_bias, use_kernel=True, cache=None
                )
                zz = z.reshape(1, S, -1, g.head_v_dim)
                fin = g.norm(out, zz)
                mx.eval(co, q, k, v, out, fin)
                return co, q, v, out, fin

            for nm in ("in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
                m = getattr(g, nm)
                xin = xn if nm != "out_proj" else xn[..., : m.input_dims]
                eq(
                    f"   {nm} {tuple(m.weight.shape)} 2048 vs 512",
                    m.prefill([xin])[0][:, :512],
                    m.prefill([xin[:, :512]])[0],
                )
            A, B = parts(2048), parts(512)
            for nm, x1, x2 in zip(("conv", "q", "v", "delta out", "norm"), A, B):
                eq(f"   {nm}", x1[:, :512], x2)
            o1, q1 = gdn(2048)
            o2, q2 = gdn(512)
            eq(f"L{li} gdn mix 2048 vs 512 (first 512)", o1[:, :512], o2)
            eq("   input qkv", q1[:, :512], q2)
        else:
            at = layer.self_attn

            def att(S, at=at, li=li, xn=xn):
                cache = lm.make_cache()[li]
                q, k, v = (
                    m.prefill([xn[:, :S]])[0] for m in (at.q_proj, at.k_proj, at.v_proj)
                )
                out = F._attention_mix(at, q, k, v, F.Segment([], ids[:S]), cache)
                mx.eval(out)
                return out, q

            def ap(S, at=at, xn=xn):
                q, k, v = (
                    m.prefill([xn[:, :S]])[0] for m in (at.q_proj, at.k_proj, at.v_proj)
                )
                from mlx_lm.models.cache import KVCache

                c = KVCache()
                queries, keys, values, gate, _ = at._prepare_projected_qkv(
                    q, k, v, c, None, None, None
                )
                o = mx.fast.scaled_dot_product_attention(
                    queries, keys, values, scale=at.scale, mask="causal"
                )
                mx.eval(queries, keys, o)
                return queries, keys, o

            A, B = ap(2048), ap(512)
            eq("   queries (rope)", A[0][:, :, :512], B[0])
            eq("   keys", A[1][:, :, :512], B[1])
            eq("   sdpa", A[2][:, :, :512], B[2])
            o1, q1 = att(2048)
            o2, q2 = att(512)
            eq(f"L{li} attn mix 2048 vs 512 (first 512)", o1[:, :512], o2)
            eq("   input q", q1[:, :512], q2)
    print("done")


main()
