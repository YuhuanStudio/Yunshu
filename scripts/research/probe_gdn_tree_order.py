"""GPU numerical parity for last-use node ordering; run only via gpuq/m3run."""

import argparse
import json
import random
import sys
from pathlib import Path

from gdn_tree_order import live_bound, order_plan, reorder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(dict(dry_run=True, cases=25)))
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    import mlx.core as mx

    from yunshu_engine import tree_verify as tv

    source = tv._TREE_GDN.replace("float states[T][NK];", "float states[LIVE][NK];")
    source = source.replace("states[par][i]", "states[read_slots[t]][i]")
    source = source.replace(
        "states[t][i] = st[i];",
        "if(write_slots[t]>=0) states[write_slots[t]][i]=st[i];",
    )
    kernel = mx.fast.metal_kernel(
        name="wide6_order_gdn",
        input_names=[
            "state_in",
            "A_log",
            "dt_bias",
            "q",
            "k",
            "v",
            "a",
            "b",
            "parents",
            "read_slots",
            "write_slots",
        ],
        output_names=["y"],
        source=source,
        header=tv.gv._HELPERS,
    )
    rng = random.Random(614)
    mx.random.seed(614)
    records = []
    for width in (2, 4, 8, 16, 32):
        for arm in range(5):
            parents = [
                -1,
                *[
                    r - 1
                    if arm == 0
                    else 0
                    if arm == 1
                    else (r - 1) // 2
                    if arm == 2
                    else rng.randrange(r)
                    for r in range(1, width)
                ],
            ]
            expected_order, expected_parents, need = order_plan(parents)
            tokens = mx.arange(1, width, dtype=mx.int32)
            new_tokens, new_parents, permutation = reorder(
                tokens, mx.array(parents, dtype=mx.int32)
            )
            mx.eval(new_tokens, new_parents, permutation)
            assert permutation.tolist() == expected_order
            assert new_parents.tolist() == expected_parents
            assert new_tokens.tolist() == expected_order[1:]
            remaining = [expected_parents.count(r) for r in range(width)]
            used, slots, reads, writes = set(), {}, [], []
            for row, parent in enumerate(expected_parents):
                reads.append(slots.get(parent, -1))
                if parent >= 0:
                    remaining[parent] -= 1
                    if remaining[parent] == 0:
                        used.remove(slots[parent])
                slot = (
                    next((s for s in range(live_bound(width)) if s not in used), -1)
                    if remaining[row]
                    else -1
                )
                assert slot >= 0 or not remaining[row]
                slots[row] = slot
                writes.append(slot)
                if slot >= 0:
                    used.add(slot)
            hk, hv, dk, dv = 4, 8, 128, 128

            def bf(shape):
                return mx.random.normal(shape).astype(mx.bfloat16)

            inputs = [
                mx.random.normal((1, hv, dv, dk)),
                mx.random.normal((hv,)),
                bf((hv,)),
                (bf((1, width, hk, dk)) / dk).astype(mx.bfloat16),
                (bf((1, width, hk, dk)) / (dk**0.5)).astype(mx.bfloat16),
                bf((1, width, hv, dv)),
                bf((1, width, hv)),
                bf((1, width, hv)),
                mx.array(parents, dtype=mx.int32),
            ]
            kw = dict(
                template=[
                    ("InT", mx.bfloat16),
                    ("Hk", hk),
                    ("Hv", hv),
                    ("Dk", dk),
                    ("Dv", dv),
                    ("T", width),
                ],
                grid=(32, dv, hv),
                threadgroup=(32, 4, 1),
                output_shapes=[(1, width, hv, dv)],
                output_dtypes=[mx.bfloat16],
            )
            reference = tv._gdn_kernel()(inputs=inputs, **kw)[0]
            permuted = [
                *inputs[:3],
                *[mx.take(x, permutation, axis=1) for x in inputs[3:8]],
                new_parents,
                mx.array(reads, dtype=mx.int32),
                mx.array(writes, dtype=mx.int32),
            ]
            kw["template"] += [("LIVE", live_bound(width))]
            candidate = kernel(inputs=permuted, **kw)[0]
            expected = mx.take(reference, permutation, axis=1)
            mx.eval(expected, candidate)
            assert bool(mx.array_equal(expected, candidate)), (width, arm)
            records.append(
                dict(
                    width=width,
                    arm=arm,
                    live_slots=need,
                    bound=live_bound(width),
                    bit_equal=True,
                )
            )
    records.append(dict(complete=True, success=True, cases=len(records)))
    args.output.write_text("".join(json.dumps(r) + "\n" for r in records))
    print(json.dumps(records[-1]), flush=True)


if __name__ == "__main__":
    main()
