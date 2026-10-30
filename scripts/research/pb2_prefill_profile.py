"""Per-layer prefill profile: round driver vs plain lane forward, same process, same prompt.

Both arms run the same language model (lane projections) over the same prompt;
a class-level wrapper evaluates every decoder layer's output and charges its
wall time to (kind, layer). The eval per layer perturbs both arms equally.

    python scripts/research/pb2_prefill_profile.py CKPT --tokens 6967 --out prof.json
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def summarize(records):
    """records: [(layer_index, kind, seconds)] -> {kind: {"total": s, "per_layer_ms": ms}}."""
    acc = defaultdict(list)
    for _i, kind, s in records:
        acc[kind].append(s)
    return {
        k: {"total": round(sum(v), 4), "calls": len(v), "mean_ms": round(1e3 * sum(v) / len(v), 3)}
        for k, v in acc.items()
    }


def main():
    import mlx.core as mx
    from mlx_vlm import load as vlm_load

    from yunshu_engine.kernels import gdn_prefill, lane_linear
    from yunshu_engine.round_driver.driver import Request, RoundDriver

    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--tokens", type=int, default=6967)
    ap.add_argument("--span", type=int, default=2048)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    model, _ = vlm_load(a.ckpt)
    lm = model.language_model
    lane_linear.convert(lm)
    if lm.args.tie_word_embeddings:
        lm._yunshu_lane_head = lane_linear.lane_head(lm.model.embed_tokens)
    import random

    rng = random.Random(0)
    ids = [rng.randrange(1000, 20000) for _ in range(a.tokens)]
    layers = list(lm.model.layers)
    kinds = {id(layer): ("gdn" if getattr(layer, "is_linear", False) else "attn") for layer in layers}
    index = {id(layer): i for i, layer in enumerate(layers)}
    records: list = []
    state = {"on": False}
    for cls in {type(layer) for layer in layers}:
        orig = cls.__call__

        def timed(self, *args, _orig=orig, **kw):
            if not state["on"]:
                return _orig(self, *args, **kw)
            mx.eval(args[0])
            t0 = time.perf_counter()
            out = _orig(self, *args, **kw)
            mx.eval(out)
            records.append((index[id(self)], kinds[id(self)], time.perf_counter() - t0))
            return out

        cls.__call__ = timed

    def plain():
        cache = lm.make_cache()
        for s in range(0, len(ids), a.span):
            with gdn_prefill.prefill_scope():
                h = lm.model(mx.array(ids[s : s + a.span], dtype=mx.int32)[None], cache=cache)
            mx.eval(h[:, -1], [c.state for c in cache])

    def driver():
        d = RoundDriver(model)
        d.add(Request(ids, 1, handle=0))
        while d.busy():
            d.step()

    out = {}
    for arm in ("warmup", "plain", "driver", "plain", "driver"):
        fn = plain if "plain" in arm else driver
        for timed_run in (False, True):
            if arm == "warmup" and timed_run:
                continue
            records.clear()
            state["on"] = timed_run
            mx.synchronize()
            t0 = time.perf_counter()
            fn()
            mx.synchronize()
            wall = time.perf_counter() - t0
            key = f"{arm}_{'layers' if timed_run else 'wall'}"
            out.setdefault(key, []).append(
                {"wall_s": round(wall, 3), **({"by_kind": summarize(records)} if timed_run else {})}
            )
            print(key, json.dumps(out[key][-1]), flush=True)
    a.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
