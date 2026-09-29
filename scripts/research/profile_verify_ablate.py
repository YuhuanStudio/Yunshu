"""In-context cost of each part of the speculative lane's verify forward, by ablation.

A barrier per part inflates every number (a sync costs ~150 us), and isolated
kernel timings overstate parts that overlap in the real dependent chain. This
instead runs upstream's ``_mtp_verify_target`` (the served verify) on a prefilled
cache with one part of every layer replaced by a constant (zeros), and reports the
step time; the difference from the full step is that part's in-context cost.

Parts: the GDN mixer, the attention mixer, the MLP (gate/up, swiglu, down) and its
halves. ``--tgs`` re-times the full step for packed-kernel threadgroup targets.

    python scripts/research/profile_verify_ablate.py MODEL_DIR --context 1024 \
        --rows 1 6 --output runs/verify-ablate.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

FILLER = "".join(
    f"Log line {i}: sensor {i % 17} reading {i * 37 % 1000}.\n" for i in range(40000)
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("--context", type=int, nargs="+", default=[1024])
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 6])
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument(
        "--tgs",
        type=int,
        nargs="*",
        default=[],
        help="packed threadgroup targets to time",
    )
    ap.add_argument(
        "--lane-layers", action="store_true", help="fused add+norm and flush"
    )
    ap.add_argument(
        "--lane-linear",
        action="store_true",
        help="TensorFold lane matmul for every projection",
    )
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    import mlx_vlm.speculative.mtp as mtp
    from mlx_vlm import load
    from mlx_vlm.models.qwen3_5 import speculative_verifier as sv

    from yunshu_engine.kernels import omlx, ragged_kv
    from yunshu_engine.kernels.batch_invariant import install as install_invariant
    from yunshu_engine.kernels.batch_invariant import set_active
    from yunshu_engine.kernels.omlx import qwen35_packed_linear as pl

    omlx.apply(row_exact=False)
    model, processor = load(a.model_dir)
    lm = model.language_model
    if a.lane_linear:
        from yunshu_engine.kernels import lane_linear

        print(
            json.dumps({"lane_linear": lane_linear.convert(lm)["converted"]}),
            flush=True,
        )
    install_invariant(
        lm, model=model, packed=omlx.is_nax_available() and not a.lane_linear
    )
    set_active(True)
    ragged_kv.install()
    ragged_kv.set_dense_lane(True)
    if a.lane_layers:
        from yunshu_engine.kernels import lane_layers

        lane_layers.install()
    tok = processor.tokenizer
    ids_all = tok.encode(FILLER, add_special_tokens=False)
    cls = sv.Qwen3_5BatchInvariantForward
    orig = {n: getattr(cls, n) for n in ("_gated_delta", "_attention", "_feed_forward")}

    def zeros_mixer(self, mixer, x, *rest):
        return mx.zeros_like(x)

    def ff_gate_up_only(self, ff, x):
        gate, up = self._linears((ff.gate_proj, ff.up_proj), x)
        return (gate * up)[..., : x.shape[-1]]

    def ff_down_only(self, ff, x):
        wide = mx.zeros((*x.shape[:-1], ff.down_proj.input_dims), dtype=x.dtype)
        return self._linear(ff.down_proj, wide)

    def zeros_ff(self, ff, x):
        return mx.zeros_like(x)

    variants = {
        "full": {},
        "no_gdn": {"_gated_delta": zeros_mixer},
        "no_attn": {"_attention": zeros_mixer},
        "no_mixers": {"_gated_delta": zeros_mixer, "_attention": zeros_mixer},
        "no_mlp": {"_feed_forward": zeros_ff},
        "mlp_gate_up_only": {"_feed_forward": ff_gate_up_only},
        "mlp_down_only": {"_feed_forward": ff_down_only},
        "mixers_only": {"_feed_forward": zeros_ff},
    }

    def apply(name):
        for n, fn in orig.items():
            setattr(cls, n, fn)
        for n, fn in variants[name].items():
            setattr(cls, n, fn)

    def sampler(lg):
        return mx.argmax(lg, axis=-1)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    for ctx in a.context:
        apply("full")
        cache = lm.make_cache()
        ids = mx.array(ids_all[:ctx])[None]
        for s in range(0, ctx, 2048):
            lm(ids[:, s : s + 2048], cache=cache)
            mx.eval([c.state for c in cache])
        for T in a.rows:
            toks = mx.array(ids_all[ctx : ctx + T])[None]

            def step():
                r = mtp._mtp_verify_target(lm, toks, cache, sampler)
                mx.eval(r.target_tokens)
                try:
                    r.abort()
                except RuntimeError:  # an ablated part never recorded its state
                    pass

            def timeit():
                for _ in range(3):
                    step()
                best = 1e9
                for _ in range(3):
                    mx.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(a.steps):
                        step()
                    best = min(best, (time.perf_counter() - t0) / a.steps)
                return best * 1e3

            base = None
            for name in variants:
                apply(name)
                ms = timeit()
                base = ms if name == "full" else base
                row = {
                    "context": ctx,
                    "T": T,
                    "variant": name,
                    "step_ms": round(ms, 2),
                    "vs_full_ms": round(ms - base, 2),
                }
                with a.output.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            apply("full")
            default = pl._TARGET_TGS
            for tgs in a.tgs:
                pl._TARGET_TGS = tgs
                ms = timeit()
                row = {
                    "context": ctx,
                    "T": T,
                    "variant": f"full tgs={tgs}",
                    "step_ms": round(ms, 2),
                }
                with a.output.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            pl._TARGET_TGS = default
            mx.clear_cache()


if __name__ == "__main__":
    main()
