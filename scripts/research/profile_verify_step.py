"""Cost of the speculative lane's verify forward (T rows) by layer part.

Runs upstream's ``_mtp_verify_target`` (what the MTP rounds call) on a prefilled
cache with the served lane stack: total time pipelined the way rounds run it,
the launch count of one verify graph (primitives that are not views), and a
barrier split of the verifier's parts (GDN mixer, attention mixer, feed-forward,
the rest: norms / adds / residual, head).

    python scripts/research/profile_verify_step.py MODEL_DIR --context 1024 \
        --rows 1 7 --output runs/verify-step.jsonl
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

FILLER = "".join(
    f"Log line {i}: sensor {i % 17} reading {i * 37 % 1000}.\n" for i in range(40000)
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("--context", type=int, nargs="+", default=[1024])
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 7])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--lane-linear",
        action="store_true",
        help="TensorFold lane matmul for every projection",
    )
    ap.add_argument("--packed-geometry", default=None, choices=("target", "few"))
    a = ap.parse_args()

    import mlx_vlm.speculative.mtp as mtp
    from mlx_vlm import load
    from mlx_vlm.models.qwen3_5 import speculative_verifier as sv

    from profile_decode_ops import count_graph
    from yunshu_engine.kernels import omlx, ragged_kv
    from yunshu_engine.kernels.omlx import qwen35_packed_linear

    if a.packed_geometry:
        qwen35_packed_linear.GEOMETRY = a.packed_geometry
    from yunshu_engine.kernels.batch_invariant import install as install_invariant
    from yunshu_engine.kernels.batch_invariant import set_active

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
    tok = processor.tokenizer
    ids_all = tok.encode(FILLER, add_special_tokens=False)

    acc = defaultdict(float)
    cnt = defaultdict(int)
    state = {"on": False}
    cls = sv.Qwen3_5BatchInvariantForward

    def timed(name, fn):
        def inner(self, *args, **kw):
            if not state["on"]:
                return fn(self, *args, **kw)
            out = fn(self, *args, **kw)
            first = out[0] if isinstance(out, tuple) else out
            mx.eval(first)
            t = time.perf_counter()
            acc[name] += t - state["last"]
            cnt[name] += 1
            state["last"] = t
            return out

        return inner

    def entering(name, fn):
        # barrier before the part so earlier lazy work is attributed to "between"
        def inner(self, *args, **kw):
            if state["on"]:
                x = args[1] if len(args) > 1 else None
                if isinstance(x, mx.array):
                    mx.eval(x)
                t = time.perf_counter()
                acc["between(norm/add)"] += t - state["last"]
                state["last"] = t
            return timed(name, fn)(self, *args, **kw)

        return inner

    cls._gated_delta = entering("gdn mixer", cls._gated_delta)
    cls._attention = entering("attn mixer", cls._attention)
    cls._feed_forward = entering("feed_forward", cls._feed_forward)

    def sampler(lg):
        return mx.argmax(lg, axis=-1)

    a.output.parent.mkdir(parents=True, exist_ok=True)
    for ctx in a.context:
        cache = lm.make_cache()
        ids = mx.array(ids_all[:ctx])[None]
        for s in range(0, ctx, 2048):
            lm(ids[:, s : s + 2048], cache=cache)
            mx.eval([c.state for c in cache])
        for T in a.rows:
            toks = mx.array(ids_all[ctx : ctx + T])[None]

            def verify():
                r = mtp._mtp_verify_target(lm, toks, cache, sampler)
                return r

            def once():
                r = verify()
                mx.async_eval(r.target_tokens, r.hidden)
                return r

            for _ in range(3):
                r = once()
                mx.eval(r.target_tokens)
                r.abort()
            t0 = time.perf_counter()
            for _ in range(a.steps):
                r = once()
                mx.eval(r.target_tokens)
                r.abort()
            eval_ms = (time.perf_counter() - t0) / a.steps * 1e3
            r = verify()
            graph = count_graph(r.target_tokens)
            mx.eval(r.target_tokens)
            r.abort()
            t0 = time.perf_counter()
            r = verify()
            build_ms = (time.perf_counter() - t0) * 1e3
            mx.eval(r.target_tokens)
            r.abort()
            # barrier split
            acc.clear()
            cnt.clear()
            state["on"] = True
            n = 4
            for _ in range(n):
                mx.synchronize()
                state["last"] = time.perf_counter()
                r = verify()
                mx.eval(r.target_tokens)
                acc["tail(head/argmax/rest)"] += time.perf_counter() - state["last"]
                r.abort()
            state["on"] = False
            split = {k: round(v / n * 1e3, 2) for k, v in sorted(acc.items())}
            row = {
                "context": ctx,
                "T": T,
                "eval_ms": round(eval_ms, 2),
                "build_ms": round(build_ms, 2),
                "graph": graph,
                "barrier_split_ms": split,
            }
            with a.output.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            mx.clear_cache()


if __name__ == "__main__":
    main()
