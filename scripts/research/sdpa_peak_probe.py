"""Probe transient allocation of mx.fast.scaled_dot_product_attention (prefill shapes)."""

import json

import mlx.core as mx
from mlx_vlm.models.cache import create_causal_mask


def probe(D, qL, kL, kind, H=24, Hk=4, left_pad=False):
    q = mx.random.normal((1, H, qL, D)).astype(mx.bfloat16)
    k = mx.random.normal((1, Hk, kL, D)).astype(mx.bfloat16)
    v = mx.random.normal((1, Hk, kL, D)).astype(mx.bfloat16)
    if kind == "causal":
        mask = "causal"
    elif kind == "bool":
        mask = create_causal_mask(qL, offset=kL - qL)
    elif kind == "bool_lp":
        mask = create_causal_mask(qL, offset=kL - qL, left_padding=mx.array([0]))
    elif kind == "none":
        mask = None
    mx.eval(q, k, v, *([mask] if isinstance(mask, mx.array) else []))
    mx.clear_cache()
    base = mx.get_active_memory()
    mx.reset_peak_memory()
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5, mask=mask)
    mx.eval(o)
    peak = mx.get_peak_memory() - base - o.nbytes
    return dict(
        D=D,
        qL=qL,
        kL=kL,
        mask=kind,
        mask_dtype=str(getattr(mask, "dtype", None)),
        mask_shape=list(getattr(mask, "shape", [])),
        extra_gib=round(peak / 2**30, 3),
    )


if __name__ == "__main__":
    out = []
    for D in (128, 256):
        for kind in ("causal", "bool", "bool_lp", "none"):
            for klen in (8192, 32768, 98304):
                r = probe(D, 2048, klen, kind)
                print(json.dumps(r), flush=True)
                out.append(r)
