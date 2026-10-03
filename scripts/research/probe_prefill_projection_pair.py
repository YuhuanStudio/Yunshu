"""Same-checkpoint projection A/B for arithmetic-preserving lane candidates."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "python")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=("barrier", "paired32"), required=True)
    args = parser.parse_args()
    import mlx.core as mx

    from yunshu_engine.kernels import lane_linear

    if args.variant == "paired32":
        from lane_paired32 import install
    else:
        from lane_final_barrier import install
    root = Path("/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp")
    cfg = json.loads((root / "config.json").read_text())["quantization"]
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    names = [
        "language_model.model.layers.60." + name
        for name in ("mlp.up_proj", "mlp.down_proj", "linear_attn.out_proj")
    ]
    names += [
        "language_model.model.layers.63.self_attn." + name
        for name in ("q_proj", "k_proj", "v_proj", "o_proj")
    ]
    layers = []
    for shard in sorted({index[name + ".weight"] for name in names}):
        loaded = mx.load(str(root / shard))
        for name in names:
            if index[name + ".weight"] != shard:
                continue
            q = cfg.get(name, cfg)
            layer = lane_linear.LaneLinear(
                loaded[name + ".weight"],
                loaded[name + ".scales"],
                loaded[name + ".biases"],
                q["bits"],
                q["group_size"],
            )
            mx.eval(layer.parameters())
            layers.append((name, layer))
        del loaded
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as out:
        for name, layer in layers:
            for rows in (66, 170, 282):
                x = mx.random.normal((rows, layer.input_dims)).astype(mx.bfloat16)
                mx.eval(x)
                expected = layer(x)
                mx.eval(expected)
                undo = install()
                actual = layer(x)
                mx.eval(actual)
                equal = bool(mx.array_equal(expected, actual).item())
                undo()
                if not equal:
                    raise RuntimeError((name, rows, "not bit-equal"))
                for rep in range(3):
                    for mode in (
                        ("main", "candidate") if rep != 1 else ("candidate", "main")
                    ):
                        undo = install() if mode == "candidate" else None
                        mx.eval(layer(x))
                        times = []
                        for _ in range(8):
                            mx.synchronize()
                            begin = time.perf_counter()
                            mx.eval(layer(x))
                            times.append(1000 * (time.perf_counter() - begin))
                        if undo:
                            undo()
                        row = dict(
                            module=name,
                            bits=layer.bits,
                            rows=rows,
                            rep=rep,
                            mode=mode,
                            variant=args.variant,
                            equal=equal,
                            ms=statistics.median(times),
                            all_ms=times,
                        )
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                        print(json.dumps(row), flush=True)
        out.write(json.dumps(dict(phase="complete", success=True)) + "\n")


if __name__ == "__main__":
    main()
