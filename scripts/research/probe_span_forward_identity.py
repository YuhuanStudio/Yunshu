"""Short span hidden/cache bit gate; insufficient to approve this experiment.

The 256-token HTTP comparison rejected span_forward despite this gate passing.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "python")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="/Volumes/P5Plus/models/Qwen3.5-0.8B-MLX-bf16"
    )
    parser.add_argument("--ctx", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from yunshu_engine.vlm_engine import VLMEngine

    engine = VLMEngine(args.model)
    asyncio.run(engine.start())
    from span_forward import _PLAN, install

    counts, uninstall = install()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def run():
        import mlx.core as mx
        from mlx_vlm.apc import _clone_prompt_cache_for_apc

        from yunshu_engine import cache_decode
        from yunshu_engine.kernels import batch_invariant, ragged_kv

        invariant = batch_invariant.is_installed()
        batch_invariant.set_active(invariant)
        cache_decode.set_active(invariant)
        ragged_kv.set_dense_lane(invariant)
        model = engine._model.language_model
        primed = model.make_cache()
        for begin in range(0, args.ctx, 2048):
            tokens = mx.arange(begin, min(begin + 2048, args.ctx)).reshape(1, -1) % 1024
            model(tokens, cache=primed, skip_logits=True, return_hidden=True)
            mx.eval([c.state for c in primed])
        records = []
        try:
            for prefix in (15, 66, 279):
                baseline = _clone_prompt_cache_for_apc(primed)
                candidate = _clone_prompt_cache_for_apc(primed)
                tokens = (mx.arange(prefix + 2).reshape(1, -1) + 123) % 1024
                first = model(
                    tokens[:, :prefix],
                    cache=baseline,
                    skip_logits=True,
                    return_hidden=True,
                ).hidden_states[-1]
                last = model(
                    tokens[:, prefix:],
                    cache=baseline,
                    skip_logits=True,
                    return_hidden=True,
                ).hidden_states[-1]
                mx.eval(first, last, [c.state for c in baseline])
                before = counts["forwards"]
                token = _PLAN.set((args.ctx + prefix,))
                try:
                    actual = model(
                        tokens, cache=candidate, skip_logits=True, return_hidden=True
                    ).hidden_states[-1]
                    mx.eval(actual, [c.state for c in candidate])
                finally:
                    _PLAN.reset(token)
                hidden_equal = bool(
                    mx.array_equal(mx.concatenate([first, last], axis=1), actual).item()
                )
                states_equal = all(
                    bool(mx.array_equal(a, b).item())
                    for left, right in zip(baseline, candidate, strict=True)
                    for a, b in zip(left.state, right.state, strict=True)
                )
                row = dict(
                    prefix=prefix,
                    tail=2,
                    ctx=args.ctx,
                    hidden_equal=hidden_equal,
                    states_equal=states_equal,
                    joined_forwards=counts["forwards"] - before,
                    invariant=invariant,
                    counts=dict(counts),
                )
                records.append(row)
                print(json.dumps(row), flush=True)
                if not (hidden_equal and states_equal and row["joined_forwards"]):
                    raise RuntimeError(f"span scheduling changed bits: {row}")
            return records
        finally:
            cache_decode.set_active(False)
            batch_invariant.set_active(False)
            ragged_kv.set_dense_lane(False)

    try:
        records = engine._executor.submit(run).result()
        args.out.write_text(
            "".join(json.dumps(row) + "\n" for row in records)
            + json.dumps(dict(phase="complete", success=True))
            + "\n"
        )
    finally:
        uninstall()
        asyncio.run(engine.stop())


if __name__ == "__main__":
    main()
