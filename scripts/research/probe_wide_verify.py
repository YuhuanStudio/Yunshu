"""Wide verify windows on the speculative lane: row invariance and cost by row count.

``invariance``: for each window width T, a T-row chain verify must give every
row the bits of a one-row verify of the same token after its predecessors
(hidden states and argmax tokens), and a tree window's row must equal the chain
verify of its root path. ``time``: the verify forward's ms by row count (chain
and tree shapes), pipelined like the rounds, with a barrier split by part.

    python scripts/research/probe_wide_verify.py MODEL_DIR --mode invariance --rows 2 8 9 16 24 32
    python scripts/research/probe_wide_verify.py MODEL_DIR --mode time --rows 1 8 16 --shapes chain heap
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

FILLER = "".join(
    f"Log line {i}: sensor {i % 17} reading {i * 37 % 1000}.\n" for i in range(40000)
)


def shape_parents(kind: str, w: int) -> list[int]:
    if kind == "chain":
        return list(range(-1, w - 1))
    if kind == "heap":  # binary heap: parent (r - 1) // 2
        return [-1] + [(r - 1) // 2 for r in range(1, w)]
    if kind == "spine":  # a chain of w // 2 with one leaf hanging off each spine row
        half = max(1, w // 2)
        par = list(range(-1, half - 1))
        par += [min(r - half, half - 1) for r in range(half, w)]
        return par[:w]
    raise ValueError(kind)


def setup(model_dir: str, lane_linear: bool = True):
    from mlx_vlm import load

    from yunshu_engine.kernels import omlx, ragged_kv
    from yunshu_engine.kernels.batch_invariant import install as install_invariant
    from yunshu_engine.kernels.batch_invariant import set_active

    omlx.apply(row_exact=False)
    model, processor = load(model_dir)
    lm = model.language_model
    if lane_linear:
        from yunshu_engine.kernels import lane_linear as ll

        ll.convert(lm)
    install_invariant(lm, model=model, packed=False)
    set_active(True)
    ragged_kv.install()
    ragged_kv.set_dense_lane(True)
    ids = processor.tokenizer.encode(FILLER, add_special_tokens=False)
    return model, lm, ids


def prefill(lm, ids, ctx):
    cache = lm.make_cache()
    arr = mx.array(ids[:ctx])[None]
    for s in range(0, ctx, 2048):
        lm(arr[:, s : s + 2048], cache=cache)
        mx.eval([c.state for c in cache])
    return cache


def chain_verify(lm, toks, cache):
    import mlx_vlm.speculative.mtp as mtp

    return mtp._mtp_verify_target(lm, toks, cache, lambda lg: mx.argmax(lg, axis=-1))


def invariance(a, lm, ids):
    from yunshu_engine import tree_verify as tv

    ok_all = True
    for T in a.rows:
        cache = prefill(lm, ids, a.context)
        toks = mx.array(ids[a.context : a.context + T])[None]
        r = chain_verify(lm, toks, cache)
        mx.eval(r.hidden, r.target_tokens)
        wide_h, wide_t = r.hidden, r.target_tokens
        r.abort()
        # one-row decode of the same tokens, one after another
        hs, ts = [], []
        for j in range(T):
            out = lm(toks[:, j : j + 1], cache=cache, return_hidden=True)
            h = out.hidden_states[-1]
            mx.eval(h, out.logits)
            hs.append(h)
            ts.append(mx.argmax(out.logits, axis=-1))
        seq_h = mx.concatenate(hs, axis=1)
        seq_t = mx.concatenate(ts, axis=1)
        bad = [
            j for j in range(T) if not mx.array_equal(wide_h[:, j], seq_h[:, j]).item()
        ]
        tok_ok = mx.array_equal(wide_t, seq_t).item()
        row = {
            "check": "chain",
            "T": T,
            "bit_equal": not bad,
            "bad_rows": bad,
            "tokens_equal": bool(tok_ok),
        }
        ok_all &= not bad and bool(tok_ok)
        print(json.dumps(row), flush=True)
        # tree windows: row r equals the chain verify of its root path
        for kind in a.shapes:
            if kind == "chain" or T < 3:
                continue
            cache = prefill(lm, ids, a.context)
            parents = shape_parents(kind, T)
            shape = tv.TreeShape(parents)
            res = tv.tree_forward(lm, toks, shape, cache)
            mx.eval(res.hidden)
            h_tree = res.hidden
            tv.tree_abort(cache, res)
            bad = []
            for rr in range(T):
                path = shape.paths[rr]
                pt = mx.array([[int(toks[0, p].item()) for p in path]])
                rc = chain_verify(lm, pt, cache)
                mx.eval(rc.hidden)
                if not mx.array_equal(rc.hidden[:, -1], h_tree[:, rr]).item():
                    bad.append(rr)
                rc.abort()
            print(
                json.dumps(
                    {
                        "check": "tree",
                        "shape": kind,
                        "T": T,
                        "bit_equal": not bad,
                        "bad_rows": bad,
                    }
                ),
                flush=True,
            )
            ok_all &= not bad
    print(json.dumps({"all_equal": ok_all}), flush=True)
    return ok_all


def timing(a, lm, ids):
    from mlx_vlm.models.qwen3_5 import speculative_verifier as sv

    from yunshu_engine import tree_verify as tv

    cls = sv.Qwen3_5BatchInvariantForward
    acc = defaultdict(float)
    state = {"on": False, "last": 0.0}

    def wrap(owner, name, label):
        fn = getattr(owner, name)

        def inner(*args, **kw):
            if not state["on"]:
                return fn(*args, **kw)
            out = fn(*args, **kw)
            first = out[0] if isinstance(out, tuple) else out
            mx.eval(first)
            t = time.perf_counter()
            acc[label] += t - state["last"]
            state["last"] = t
            return out

        setattr(owner, name, inner)

    wrap(cls, "_gated_delta", "gdn")
    wrap(cls, "_attention", "attn")
    wrap(cls, "_feed_forward", "ffn")
    wrap(tv, "_gdn_layer", "gdn")
    wrap(tv, "tree_attention", "attn")

    out_rows = []
    for ctx in a.context_list:
        cache = prefill(lm, ids, ctx)
        for T in a.rows:
            toks = mx.array(ids[ctx : ctx + T])[None]
            for kind in a.shapes:
                if kind == "chain-verifier":

                    def fwd():
                        r = chain_verify(lm, toks, cache)
                        return r.target_tokens, r.abort
                else:
                    shape = tv.TreeShape(shape_parents(kind, T))

                    def fwd(shape=shape, toks=toks):
                        res = tv.tree_forward(lm, toks, shape, cache)
                        tgt = lm.speculative_argmax_from_hidden(res.hidden)
                        return tgt, lambda: tv.tree_abort(cache, res)

                for _ in range(3):
                    t, ab = fwd()
                    mx.eval(t)
                    ab()
                t0 = time.perf_counter()
                for _ in range(a.steps):
                    t, ab = fwd()
                    mx.eval(t)
                    ab()
                ms = (time.perf_counter() - t0) / a.steps * 1e3
                t0 = time.perf_counter()
                t, ab = fwd()
                build = (time.perf_counter() - t0) * 1e3
                mx.eval(t)
                ab()
                acc.clear()
                state["on"] = True
                n = 3
                for _ in range(n):
                    mx.synchronize()
                    state["last"] = time.perf_counter()
                    t, ab = fwd()
                    mx.eval(t)
                    acc["tail+between"] += time.perf_counter() - state["last"] - 0
                    ab()
                state["on"] = False
                parts = {k: round(v / n * 1e3, 2) for k, v in sorted(acc.items())}
                row = {
                    "context": ctx,
                    "T": T,
                    "shape": kind,
                    "eval_ms": round(ms, 2),
                    "build_ms": round(build, 2),
                    "barrier_ms": parts,
                }
                print(json.dumps(row), flush=True)
                out_rows.append(row)
                mx.clear_cache()
    return out_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_dir")
    ap.add_argument("--mode", choices=("invariance", "time"), required=True)
    ap.add_argument("--rows", type=int, nargs="+", default=[2, 8, 9, 16])
    ap.add_argument("--context", type=int, default=600)
    ap.add_argument("--context-list", type=int, nargs="+", default=[1024])
    ap.add_argument("--shapes", nargs="+", default=["heap"])
    ap.add_argument("--steps", type=int, default=12)
    a = ap.parse_args()
    model, lm, ids = setup(a.model_dir)
    if a.mode == "invariance":
        sys.exit(0 if invariance(a, lm, ids) else 1)
    timing(a, lm, ids)


if __name__ == "__main__":
    main()
