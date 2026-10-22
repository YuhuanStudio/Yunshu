"""Stream distinct real projection weights in one MLX eval graph (gpuq only).

No per-projection eval in the timed region. Reports useful weight+scale bytes,
not hardware DRAM counters. --dry-run reads only safetensors headers (no MLX).
"""

import argparse
import importlib.util
import json
import statistics
import struct
import sys
import time
from collections import defaultdict
from functools import partial
from pathlib import Path


def inventory(root, draft=False):
    groups = defaultdict(list)
    for file in sorted(Path(root).glob("*.safetensors")):
        with file.open("rb") as stream:
            header = json.loads(stream.read(struct.unpack("<Q", stream.read(8))[0]))
        for key, value in header.items():
            if not key.endswith(".weight") or len(value.get("shape", [])) != 2:
                continue
            prefix = key[:-7]
            if draft:
                if not prefix.startswith("layers.") and prefix != "fc":
                    continue
                n, k = value["shape"]
                bits = 8
            else:
                if (
                    not prefix.startswith("language_model.model.layers.")
                    or prefix + ".scales" not in header
                ):
                    continue
                n, kg = header[prefix + ".scales"]["shape"]
                k = kg * 64
                bits = value["shape"][1] * 32 // k
            name = (
                (
                    ".".join(prefix.split(".")[2:])
                    if prefix.startswith("layers.")
                    else prefix
                )
                if draft
                else ".".join(prefix.split(".")[4:])
            )
            if k % 64 or n % 4:
                continue
            groups[(name, n, k, bits)].append((str(file), prefix))
    if draft:
        for first, second, fused in [
            ("self_attn.k_proj", "self_attn.v_proj", "self_attn.kv"),
            ("mlp.gate_proj", "mlp.up_proj", "mlp.gate_up"),
        ]:
            for (name, n, k, bits), members in list(groups.items()):
                if name == first:
                    groups[(fused, n * 2, k, bits)] = [
                        (file, prefix + "|" + prefix.removesuffix(first) + second)
                        for file, prefix in members
                    ]
        kv = next(
            (
                (key, members)
                for key, members in groups.items()
                if key[0] == "self_attn.kv"
            ),
            None,
        )
        if kv and len({file for file, _ in kv[1]}) == 1:
            (_, n, k, bits), members = kv
            groups[("self_attn.context_kv", n * len(members), k, bits)] = [
                (members[0][0], "|".join(prefix for _, prefix in members))
            ]
    return groups


def projection_graph(mx, layers, inputs, serial=False):
    """One eval graph; optional dependencies retain inputs without arithmetic."""
    outputs = []
    for layer, x in zip(layers, inputs, strict=True):
        if serial and outputs:
            x = mx.depends(x, outputs[-1])
        outputs.append(layer(x))
    return outputs


def activate_mode(q, reference, native, install, mode):
    if reference is not None:
        q.lane_matmul = reference.lane_matmul if mode.startswith("base") else native

        def undo():
            q.lane_matmul = native

        return undo
    return install() if mode.startswith("cand") else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True)
    p.add_argument("--draft", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--direct8", action="store_true")
    p.add_argument("--compare-direct8", action="store_true")
    p.add_argument("--selective8", action="store_true")
    p.add_argument("--compare-half", action="store_true")
    p.add_argument("--compare-pair16", action="store_true")
    p.add_argument("--half-row-block", type=int, choices=(16, 32))
    p.add_argument("--parity", action="store_true")
    p.add_argument("--baseline-qmm", type=Path)
    p.add_argument("--gate-only", action="store_true")
    p.add_argument("--only-draft", action="store_true")
    p.add_argument("--only-name")
    p.add_argument("--whole-forward", action="store_true")
    p.add_argument("--serial", action="store_true")
    p.add_argument("--whole-reps", type=int, default=15)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--peak-gbps", type=float, default=614)
    a = p.parse_args()
    if a.baseline_qmm and not a.baseline_qmm.is_file():
        p.error("baseline qmm source does not exist")
    if a.gate_only and not a.parity:
        p.error("--gate-only requires --parity")
    if a.baseline_qmm and (
        a.compare_pair16 or a.compare_half or a.compare_direct8 or a.direct8
    ):
        p.error("baseline mode measures native candidate, without research installers")
    jobs = [
        (kind, key, members)
        for kind, root in [("target", a.target), ("draft", a.draft)]
        for key, members in inventory(root, kind == "draft").items()
    ]
    if a.only_name:
        jobs = [job for job in jobs if job[1][0] == a.only_name]
    if not jobs:
        p.error("no matching projection shapes")
    if a.only_draft:
        jobs = [job for job in jobs if job[0] == "draft"]
    if a.dry_run:
        if a.compare_pair16:
            from lane_pair16 import preflight

            print(
                json.dumps(
                    {"pair16_preflight": preflight(Path(__file__).resolve().parents[2])}
                )
            )
        print(json.dumps([dict(kind=c, shape=k, layers=len(v)) for c, k, v in jobs]))
        return
    import mlx.core as mx

    from yunshu_engine.kernels.lane_linear import LaneLinear
    from yunshu_engine.kernels.tensorfold import lane_qmm as q

    reference = None
    native = q.lane_matmul
    if a.baseline_qmm:
        spec = importlib.util.spec_from_file_location(
            "yunshu_engine.kernels.tensorfold._lanebw_reference", a.baseline_qmm
        )
        reference = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = reference
        spec.loader.exec_module(reference)

    def activate(mode):
        return activate_mode(q, reference, native, install, mode)

    serving_draft = any(
        kind == "draft" and key[0] == "self_attn.context_kv" for kind, key, _ in jobs
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as out:

        def emit(row):
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps(row), flush=True)

        emit(
            dict(
                phase="device",
                info=mx.device_info(),
                peak_gbps=a.peak_gbps,
                native_source=str(q.__file__),
                reference_source=str(a.baseline_qmm) if a.baseline_qmm else None,
            )
        )
        if a.compare_pair16:
            from lane_pair16 import install
        elif a.compare_half:
            from lane_half_tile import install as half_install

            install = partial(half_install, row_block=a.half_row_block)
        else:
            from lane_direct8 import install as byte_install

            install = partial(byte_install, selective=a.selective8)

        retained = []
        for kind, (name, n, k, bits), members in jobs[:1] if a.smoke else jobs:
            loaded = {}
            linears = []
            for file, prefix in members[:2] if a.smoke else members:
                if file not in loaded:
                    loaded[file] = mx.load(file)
                data = loaded[file]
                if kind == "draft":
                    parts = [
                        mx.quantize(data[pre + ".weight"], bits=8, group_size=64)
                        for pre in prefix.split("|")
                    ]
                    w, s, b = [
                        mx.concatenate([part[i] for part in parts], axis=0)
                        for i in range(3)
                    ]
                    del parts
                else:
                    w, s, b = [
                        data[prefix + suffix]
                        for suffix in (".weight", ".scales", ".biases")
                    ]
                layer = LaneLinear(w, s, b, bits, 64)
                if kind == "draft":
                    from yunshu_engine.dflash_fast import draft_split_k

                    layer.sk = draft_split_k(layer)
                mx.eval(layer.parameters())
                linears.append(layer)
            del loaded, data, w, s, b
            excluded = ("self_attn.kv", "mlp.gate_up", "self_attn.context_kv")
            if kind == "draft" and serving_draft:
                excluded = (
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.kv",
                    "mlp.gate_proj",
                    "mlp.up_proj",
                )
            if a.whole_forward and name not in excluded:
                retained.extend((kind, l) for l in linears)
            traffic = sum(l.weight.nbytes + l.sbt.nbytes for l in linears)
            if a.parity:
                xgate = mx.random.normal((32, k)).astype(mx.bfloat16)
                layer = linears[0]
                for count in range(1, 33):
                    undo_base = activate("base")
                    expected = layer(xgate[:count])
                    mx.eval(expected)
                    if undo_base:
                        undo_base()
                    undo = activate("cand")
                    actual = layer(xgate[:count])
                    mx.eval(actual)
                    if undo:
                        undo()
                    if not mx.array_equal(expected, actual).item():
                        raise RuntimeError((kind, name, bits, count, "parity"))
                emit(
                    dict(
                        phase="parity",
                        kind=kind,
                        name=name,
                        bits=bits,
                        rows="1..32",
                        equal=True,
                    )
                )
                del xgate, expected, actual
            if a.gate_only:
                del linears, layer
                q._xs_cache.clear()
                mx.clear_cache()
                continue
            for rows in (1, 8, 16, 17, 32):
                x = mx.random.normal((rows, k)).astype(mx.bfloat16)
                mx.eval(x)
                modes = (
                    ("base", "cand", "cand-r1", "base-r1")
                    if a.baseline_qmm
                    or (a.compare_direct8 and bits == 8)
                    or (a.compare_half and bits == 4)
                    or (a.compare_pair16 and bits in (4, 5))
                    else (("cand",) if a.direct8 else ("base",))
                )
                for mode in modes:
                    undo = activate(mode)

                    def graph(x=x, linears=linears):
                        return projection_graph(
                            mx, linears, [x] * len(linears), a.serial
                        )

                    for _ in range(3):
                        mx.eval(graph())
                    samples = []
                    for _ in range(a.reps):
                        # Construct graph before timing; force all outputs in ONE eval.
                        ys = graph()
                        mx.synchronize()
                        start = time.perf_counter()
                        mx.eval(ys)
                        mx.synchronize()
                        samples.append((time.perf_counter() - start) * 1000)
                    ms = statistics.median(samples)
                    gbps = traffic / ms / 1e6
                    emit(
                        dict(
                            phase="timing",
                            dependency_chain=a.serial,
                            mode=mode,
                            kind=kind,
                            name=name,
                            n=n,
                            k=k,
                            bits=bits,
                            rows=rows,
                            layers=len(linears),
                            bytes=traffic,
                            sk=getattr(linears[0], "sk", None) or q.split_k(n, k),
                            samples_ms=samples,
                            ms=ms,
                            gbps=gbps,
                            peak_fraction=gbps / a.peak_gbps,
                            row_blocks=(
                                1
                                if rows <= 16
                                or (
                                    not (
                                        reference is not None
                                        and mode.startswith("base")
                                    )
                                    and getattr(
                                        q, "_lane_schedule", lambda *args: "default"
                                    )(bits, n, k, rows, 64)
                                    == "pair16"
                                )
                                else 2
                            ),
                            cache_resident_possible=traffic < 128 * 1024**2,
                        )
                    )
                    if undo:
                        undo()
            del linears, layer, ys, x, graph
            q._xs_cache.clear()
            mx.clear_cache()
        for kind in ("target", "draft") if a.whole_forward else ():
            layers = [l for c, l in retained if c == kind]
            if not layers:
                continue
            traffic = sum(l.weight.nbytes + l.sbt.nbytes for l in layers)
            for rows in (1, 8, 16, 17, 32):
                inputs = [
                    mx.random.normal((rows, l.input_dims)).astype(mx.bfloat16)
                    for l in layers
                ]
                mx.eval(inputs)
                modes = (
                    ("base", "cand", "cand-r1", "base-r1")
                    if (
                        a.baseline_qmm
                        or (a.compare_direct8 and kind == "draft")
                        or (a.compare_half and kind == "target")
                        or (a.compare_pair16 and kind == "target")
                    )
                    else (("cand",) if a.direct8 else ("base",))
                )
                for mode in modes:
                    undo = activate(mode)
                    for _ in range(3):
                        mx.eval(projection_graph(mx, layers, inputs, a.serial))
                    samples = []
                    for _ in range(a.whole_reps):
                        ys = projection_graph(mx, layers, inputs, a.serial)
                        mx.synchronize()
                        begin = time.perf_counter()
                        mx.eval(ys)
                        mx.synchronize()
                        samples.append((time.perf_counter() - begin) * 1000)
                    ms = statistics.median(samples)
                    emit(
                        dict(
                            phase="whole_projection_graph",
                            dependency_chain=a.serial,
                            mode=mode,
                            excluded=["embedding", "lm_head", "nonquantized"],
                            projection_layout="serving-fused"
                            if kind == "draft" and serving_draft
                            else "raw",
                            kind=kind,
                            rows=rows,
                            projections=len(layers),
                            bytes=traffic,
                            ms=ms,
                            samples_ms=samples,
                            gbps=traffic / ms / 1e6,
                            peak_fraction=traffic / ms / 1e6 / a.peak_gbps,
                        )
                    )
                    if undo:
                        undo()
        emit(dict(phase="complete", complete=True))


if __name__ == "__main__":
    main()
