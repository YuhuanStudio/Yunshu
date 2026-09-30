"""Decode forward cost of the round driver by window shape and by part.

B rows decode (prompts of ``--context`` tokens, no drafting); then the decode
batch's forward is timed (evaluated, plus the LM head over every window
position, like a step) for windows of T tokens (uniform) and a mixed set, and
with parts stubbed to zeros (GDN mixers, attention layers, MLPs, LM head) to
see which part grows with T.

    python scripts/research/probe_driver_window.py <ckpt> --rows 4 --context 1024
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep_round_driver import PROMPTS, encode, load  # noqa: E402


class Zero(nn.Module):
    def __call__(self, x):
        return mx.zeros_like(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt")
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    model, processor, drafter, _ = load(a.ckpt, a.quantize)
    from yunshu_engine.round_driver import batch as rb
    from yunshu_engine.round_driver.driver import Request, RoundDriver
    from yunshu_engine.round_driver.forward import logits

    tok = processor.tokenizer
    prompts = [encode(tok, p, a.context) for p in PROMPTS]
    d = RoundDriver(model, drafter=None, stop_tokens=set())
    for i in range(a.rows):
        d.add(Request(prompts[i % len(prompts)], 10_000, handle=i))
    while any(r.pending is None for r in d.rows):
        d.step()
    lm = d.lm
    B = a.rows

    def run(windows, head=True):
        ts = []
        for _ in range(a.reps + 2):
            t0 = time.perf_counter()
            hidden = d.batch.forward(windows)
            out = logits(lm, hidden) if head else hidden
            mx.eval(out)
            ts.append((time.perf_counter() - t0) * 1e3)
            d.batch.commit([len(w) for w in windows])
            for r in d.batch.rows:  # keep the context steady
                r.n -= 0
        return round(statistics.median(ts[2:]), 2)

    def windows(lens):
        return [[7 + j for j in range(n)] for n in lens]

    shapes = {
        "T1": [1] * B,
        "T2": [2] * B,
        "T4": [4] * B,
        "T6": [6] * B,
        "T8": [8] * B,
        "mixed": [6, 3, 1, 5, 2, 7, 4, 8][:B],
    }
    rec = {"rows": B, "context": a.context, "full": {}}
    for name, lens in shapes.items():
        rec["full"][name] = run(windows(lens))
    print(json.dumps(rec), flush=True)

    orig_attend, orig_gdn = rb.attend, rb.DecodeBatch._gdn
    mlps = [layer.mlp for layer in lm.model.layers]

    def stub_attend(at, xn, slots, i, plan, pack=None):
        return mx.zeros_like(xn)

    def stub_gdn(self, g, xn, i, lens, lens_arr, pack, hist):
        return mx.zeros_like(xn)

    abl = {}
    for what in ("no_attention", "no_gdn", "no_mlp"):
        if what == "no_attention":
            rb.attend = stub_attend
        elif what == "no_gdn":
            rb.DecodeBatch._gdn = stub_gdn
        else:
            for layer in lm.model.layers:
                layer.mlp = Zero()
        abl[what] = {}
        for name in ("T1", "T6", "mixed"):
            try:
                abl[what][name] = run(windows(shapes[name]))
            except Exception as e:  # a stub may break a later step's commit
                abl[what][name] = repr(e)[:80]
        rb.attend, rb.DecodeBatch._gdn = orig_attend, orig_gdn
        for layer, m in zip(lm.model.layers, mlps, strict=True):
            layer.mlp = m
    abl["no_head"] = {n: run(windows(shapes[n]), head=False) for n in ("T1", "T6")}
    rec["ablate"] = abl
    print(json.dumps(rec), flush=True)
    if a.output:
        with a.output.open("a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
