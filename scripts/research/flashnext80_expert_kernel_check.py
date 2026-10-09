"""gather_qmm vs dequantize+matmul on the REAL expert weights of a pack, per (bits, group_size).

A mixed-precision pack (5/6/8-bit, group 64/128 experts) can hit an MLX kernel path that is wrong for some
combination.  Run through gpuq only.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    import mlx.core as mx
    from mlx_vlm import load

    model, _ = load(a.model)
    lm = model.language_model
    idx = mx.array([0, 17, 100, 511], dtype=mx.uint32)
    worst = defaultdict(float)
    bad = []
    for path, mod in lm.named_modules():
        if (
            "switch_mlp." not in path
            or not hasattr(mod, "scales")
            or not hasattr(mod, "bits")
        ):
            continue
        in_dim = mod.weight.shape[-1] * 32 // mod.bits
        x = mx.random.normal((1, 1, 1, in_dim)).astype(mx.bfloat16)
        w, s, b = mod.weight[idx], mod.scales[idx], mod.biases[idx]
        ref = (
            x[0] @ mx.swapaxes(mx.dequantize(w, s, b, mod.group_size, mod.bits), -1, -2)
        ).astype(mx.float32)
        got = mx.gather_qmm(
            x[0],
            mod.weight,
            mod.scales,
            mod.biases,
            rhs_indices=idx[None],
            transpose=True,
            group_size=mod.group_size,
            bits=mod.bits,
        ).astype(mx.float32)
        # ref is [1,4,1,out] only if broadcasting matches; compare on flattened expert outputs
        ref = ref.reshape(4, -1)
        got = got.reshape(4, -1)
        err = float((mx.abs(ref - got).max() / (mx.abs(ref).max() + 1e-6)).item())
        key = f"{mod.bits}b/g{mod.group_size}"
        worst[key] = max(worst[key], err)
        if err > 0.05:
            bad.append([path, key, round(err, 4)])
    rows = [
        {
            "kind": "expert_kernel_check",
            "worst_rel_err": dict(worst),
            "bad_count": len(bad),
            "bad_first": bad[:10],
        }
    ]
    a.out.write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
        + json.dumps({"complete": True})
        + "\n"
    )
    print(rows[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
