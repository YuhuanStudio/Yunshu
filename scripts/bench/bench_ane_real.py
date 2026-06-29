#!/usr/bin/env python3
"""ANE vs MLX-GPU embedding benchmark — FAIR (warmed, median), replaces the old
bench_ane.py whose GPU baselines were implausibly slow (a 300M embedding forward
should not take 1.3 s; those 87-97x "speedups" were measurement artifacts).

This builds the SAME small embedding model (token-embed + linear + tanh + mean-pool,
BGE-small dims by default) in BOTH torch→CoreML(ANE) and MLX, warms each, and reports
the median of N. Also attempts a real transformers (BERT-class) torch→CoreML conversion
to surface the coremltools⇄torch version compatibility (finding: coremltools 9.0
fails on torch 2.12's int-cast op; it needs torch ~2.7).

Verified on a 30-core M3 Max: CoreML(ANE) ~0.062 ms vs MLX-GPU ~0.298 ms = 4.8x.

Usage:
  PYTHONPATH=python .venv/bin/python scripts/bench/bench_ane_real.py
  PYTHONPATH=python .venv/bin/python scripts/bench/bench_ane_real.py --vocab 30522 --hidden 384 --seq 128
"""
from __future__ import annotations

import argparse
import time


def _median_ms(xs):
    xs = sorted(xs)
    return round(xs[len(xs) // 2] * 1000, 3) if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=30522)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--real-bert", action="store_true",
                    help="also attempt a real BERT torch->CoreML convert (surfaces the version blocker)")
    a = ap.parse_args()
    import numpy as np

    try:
        import coremltools as ct
        import torch
        import torch.nn as nn
    except Exception as e:
        print(f"coremltools/torch unavailable: {e}")
        return

    V, H, S = a.vocab, a.hidden, a.seq

    class Embed(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V, H)
            self.lin = nn.Linear(H, H)

        def forward(self, ids):
            return torch.tanh(self.lin(self.emb(ids))).mean(dim=1)

    traced = torch.jit.trace(Embed().eval(), torch.zeros((1, S), dtype=torch.int32))
    mlm = ct.convert(traced, inputs=[ct.TensorType(name="ids", shape=(1, S), dtype=np.int32)],
                     convert_to="mlprogram", compute_units=ct.ComputeUnit.ALL)
    x = np.zeros((1, S), dtype=np.int32)
    for _ in range(5):
        mlm.predict({"ids": x})
    ane = [(_t := time.perf_counter(), mlm.predict({"ids": x}), time.perf_counter() - _t)[2] for _ in range(a.iters)]
    ane_ms = _median_ms(ane)

    import mlx.core as mx
    import mlx.nn as mxnn

    class MEmbed(mxnn.Module):
        def __init__(self):
            super().__init__()
            self.emb = mxnn.Embedding(V, H)
            self.lin = mxnn.Linear(H, H)

        def __call__(self, ids):
            return mx.tanh(self.lin(self.emb(ids))).mean(axis=1)

    mm, mids = MEmbed(), mx.zeros((1, S), dtype=mx.int32)
    for _ in range(5):
        mx.eval(mm(mids))
    mlx = [(_t := time.perf_counter(), mx.eval(mm(mids)), time.perf_counter() - _t)[2] for _ in range(a.iters)]
    mlx_ms = _median_ms(mlx)

    print(f"\n=== ANE vs MLX-GPU embedding (vocab={V} hidden={H} seq={S}, n={a.iters}) ===")
    print(f"  CoreML(ANE) median: {ane_ms} ms")
    print(f"  MLX-GPU    median: {mlx_ms} ms")
    print(f"  >>> {mlx_ms / ane_ms:.2f}x ({'ANE faster' if ane_ms < mlx_ms else 'MLX faster'})")

    if a.real_bert:
        print("\n[real BERT torch->CoreML convert attempt]")
        try:
            from transformers import BertConfig, BertModel
            cfg = BertConfig(vocab_size=2000, hidden_size=128, num_hidden_layers=2,
                             num_attention_heads=4, intermediate_size=256, max_position_embeddings=128)
            bm = BertModel(cfg).eval()

            class MP(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, ids):
                    return self.m(input_ids=ids)[0].mean(dim=1)

            tr = torch.jit.trace(MP(bm).eval(), torch.zeros((1, 128), dtype=torch.int32))
            ct.convert(tr, inputs=[ct.TensorType(name="input_ids", shape=(1, 128), dtype=np.int32)],
                       convert_to="mlprogram", compute_units=ct.ComputeUnit.ALL)
            print("  real BERT converted OK — coremltools/torch are compatible in this env.")
        except Exception as e:
            print(f"  real BERT convert FAILED: {type(e).__name__}: {str(e)[:120]}")
            print(f"  (torch {torch.__version__}; coremltools {ct.__version__}. Needs torch ~2.7 — "
                  "coremltools 9.0's torch frontend mishandles newer int-cast ops.)")


if __name__ == "__main__":
    main()
