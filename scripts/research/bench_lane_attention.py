"""Speculative-lane attention: decode (T=1) and verify (T>1) over one row.

Times the attention of one decode/verify step across ``--layers`` layers
(default 16, Qwen3.8-27B's full-attention count) with 27B heads (24 query / 4
KV, head_dim 256) for:

- tile: Yunshu's token-tile kernel (what ``ragged_kv.set_dense_lane`` runs;
  every token x head of a KV head in one pass, same bits for any T);
- key_parallel: the batch kernel (one query token per threadgroup);
- stock: what the lane runs without ragged KV — MLX SDPA for T=1, oMLX's
  verify attention (``kernels/omlx``) for T>1.

    python scripts/research/bench_lane_attention.py --keys 1024,8192,32768,131072 \
        --output runs/lane-attention.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from yunshu_engine.kernels import omlx  # noqa: E402
from yunshu_engine.kernels.ragged_attention import (  # noqa: E402
    ragged_decode_attention,
    tile_ready,
)


def bench(fn, n):
    for _ in range(3):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        mx.eval(fn())
    mx.synchronize()
    return round((time.perf_counter() - t0) / n * 1000, 3)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--keys", default="1024,8192,32768,131072")
    ap.add_argument("--tokens", default="1,3,6")
    ap.add_argument("--heads", default="24,4")
    ap.add_argument("--layers", type=int, default=16)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    omlx.apply()
    from mlx_vlm.models.cache import KVCache
    from mlx_vlm.models.qwen3_5 import language as lang

    H, HKV = (int(x) for x in a.heads.split(","))
    D = 256
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("a") as f:
        for n in (int(x) for x in a.keys.split(",")):
            cap = n + 256
            ks = [
                mx.random.normal((1, HKV, cap, D)).astype(mx.bfloat16)
                for _ in range(a.layers)
            ]
            vs = [
                mx.random.normal((1, HKV, cap, D)).astype(mx.bfloat16)
                for _ in range(a.layers)
            ]
            mx.eval(ks, vs)
            dense = KVCache()  # no left_padding: the oMLX verify seam claims it
            for T in (int(x) for x in a.tokens.split(",")):
                q = mx.random.normal((1, H, T, D)).astype(mx.bfloat16)
                lengths = mx.array([n], dtype=mx.int32)

                def ragged(impl, q=q, lengths=lengths, n=n):
                    return [
                        ragged_decode_attention(
                            q,
                            k,
                            v,
                            lengths,
                            D**-0.5,
                            max_length=n,
                            row_lengths=(n,),
                            impl=impl,
                        )
                        for k, v in zip(ks, vs, strict=True)
                    ]

                def stock(q=q, n=n, T=T):
                    outs = []
                    for k, v in zip(ks, vs, strict=True):
                        kk, vv = k[:, :, :n], v[:, :, :n]
                        o = None
                        if T > 1:
                            o = lang._qwen3_5_left_padded_attention(
                                q, kk, vv, cache=dense, scale=D**-0.5, mask=None
                            )
                        if o is None:
                            o = mx.fast.scaled_dot_product_attention(
                                q,
                                kk,
                                vv,
                                scale=D**-0.5,
                                mask="causal" if T > 1 else None,
                            )
                        outs.append(o)
                    return outs

                row = {"keys": n, "T": T, "layers": a.layers, "heads": a.heads}
                if tile_ready():
                    row["tile_ms"] = bench(lambda: ragged("tile"), a.iters)
                row["key_parallel_ms"] = bench(lambda: ragged("key_parallel"), a.iters)
                row["stock_ms"] = bench(stock, a.iters)
                f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            ks = vs = None
            mx.clear_cache()


if __name__ == "__main__":
    main()
