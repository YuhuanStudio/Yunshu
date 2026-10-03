"""GPUQ-only NAX roofline, capture, and lossless layout A/B probes.

--dry-run validates arguments without importing MLX or touching the GPU.
Microbench timings evaluate prebuilt GPU graphs with one host fence; model probes
evaluate both hidden outputs and cache (no pruned final MLP).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("micro", "model", "narrow"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", type=Path)
    p.add_argument("--tokens", type=int, nargs="+", default=[8192, 32768])
    p.add_argument("--chunks", type=int, nargs="+", default=[2048])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--capture", type=Path)
    p.add_argument("--classes", action="store_true")
    p.add_argument("--tiles", action="store_true")
    p.add_argument("--lane-loader", action="store_true")
    p.add_argument(
        "--arms",
        nargs="+",
        choices=(
            "base",
            "planes",
            "stock",
            "tile128",
            "lane64",
            "lane128",
            "narrow",
            "combo",
        ),
        default=["base", "planes", "stock"],
    )
    p.add_argument("--dry-run", action="store_true")
    return p


def write(path, row):
    with path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    print(json.dumps(row, default=str), flush=True)


def micro(a, mx):
    mx.random.seed(17)
    shapes = (
        [(512, 256, 1024)]
        if a.tiny
        else [
            (m, k, n)
            for m in (512, 2048, 4096, 8192)
            for k, n in ((5120, 17408), (17408, 5120), (5120, 5120))
        ]
    )
    for m, k, n in shapes:
        for dtype in (mx.float16, mx.bfloat16):
            x = (mx.random.normal((m, k)) * 0.02).astype(dtype)
            w = (mx.random.normal((n, k)) * 0.02).astype(dtype)
            mx.eval(x, w)

            def dense(x=x, w=w):
                return x @ w.T

            funcs = {"dense": dense}
            held = []
            for bits in (4, 5, 8):
                q, s, b = mx.quantize(w, group_size=64, bits=bits)
                d = mx.dequantize(q, s, b, group_size=64, bits=bits)
                mx.eval(q, s, b, d)
                held.append((q, s, b, d))
                funcs[f"q{bits}"] = lambda x=x, q=q, s=s, b=b, bits=bits: (
                    mx.quantized_matmul(
                        x, q, s, b, transpose=True, group_size=64, bits=bits
                    )
                )
                funcs[f"deq{bits}+gemm"] = lambda x=x, q=q, s=s, b=b, bits=bits: (
                    x @ mx.dequantize(q, s, b, group_size=64, bits=bits).T
                )
                funcs[f"resident-deq{bits}"] = lambda x=x, d=d: x @ d.T
                qref = funcs[f"q{bits}"]()
                dref = funcs[f"resident-deq{bits}"]()
                mx.eval(qref, dref)
                write(
                    a.out,
                    dict(
                        type="dequant_parity",
                        bits=bits,
                        shape=[m, k, n],
                        dtype=str(dtype),
                        bit_equal=bool(mx.array_equal(qref, dref).item()),
                        max_abs=float(
                            mx.max(
                                mx.abs(
                                    qref.astype(mx.float32) - dref.astype(mx.float32)
                                )
                            ).item()
                        ),
                    ),
                )
            if a.tiles and dtype == mx.bfloat16:
                from nax_qmm_tiles import make

                q, s, b, _ = held[0]
                ref = funcs["q4"]()
                mx.eval(ref)
                for bm, bn in ((64, 64), (128, 64), (64, 128), (128, 128)):
                    fn = make(mx, x, q, s, b, bits=4, bm=bm, bn=bn)
                    candidate = fn()
                    mx.eval(candidate)
                    equal = bool(mx.array_equal(ref, candidate).item())
                    write(
                        a.out,
                        dict(
                            type="tile_parity",
                            bm=bm,
                            bn=bn,
                            shape=[m, k, n],
                            bit_equal=equal,
                            max_abs=float(
                                mx.max(
                                    mx.abs(
                                        ref.astype(mx.float32)
                                        - candidate.astype(mx.float32)
                                    )
                                ).item()
                            ),
                        ),
                    )
                    funcs[f"tile{bm}x{bn}"] = fn
            if a.lane_loader and dtype == mx.bfloat16:
                from nax_qmm_tiles import make

                from yunshu_engine.kernels.tensorfold import lane_qmm

                for bits, (q, s, b, _) in zip((4, 5, 8), held, strict=True):
                    qt = lane_qmm.tile_weight(q, bits=bits)
                    sbt = lane_qmm.pack_scales(s, b)
                    mx.eval(qt, sbt)
                    ref = funcs[f"q{bits}"]()
                    mx.eval(ref)
                    for bm in (64, 128):
                        fn = make(mx, x, qt, s, b, bits=bits, bm=bm, bn=64, sbt=sbt)
                        candidate = fn()
                        mx.eval(candidate)
                        equal = bool(mx.array_equal(ref, candidate).item())
                        write(
                            a.out,
                            dict(
                                type="lane_loader_parity",
                                bits=bits,
                                bm=bm,
                                shape=[m, k, n],
                                bit_equal=equal,
                            ),
                        )
                        if not equal:
                            raise RuntimeError(
                                "lane loader changed native QMM arithmetic"
                            )
                        funcs[f"lane-q{bits}-bm{bm}"] = fn
            if a.capture and (m, k, n) == shapes[0] and dtype == mx.bfloat16:
                mx.metal.start_capture(str(a.capture))
                for fn in funcs.values():
                    mx.eval(fn())
                mx.metal.stop_capture()
            for rep in range(a.reps):
                order = list(funcs) if rep % 2 == 0 else list(reversed(funcs))
                for name in order:
                    fn = funcs[name]
                    mx.eval(fn())
                    # Build graphs outside timing, then one eval/fence for ten GPU calls.
                    outputs = [fn() for _ in range(10)]
                    mx.synchronize()
                    started = time.perf_counter()
                    mx.eval(outputs)
                    dt = (time.perf_counter() - started) / 10
                    del outputs
                    write(
                        a.out,
                        dict(
                            type="micro",
                            shape=[m, k, n],
                            dtype=str(dtype),
                            arm=name,
                            rep=rep,
                            seconds=dt,
                            effective_tflops=2 * m * k * n / dt / 1e12,
                        ),
                    )
            del funcs, held, x, w
            mx.clear_cache()


def model(a, mx):
    from contextlib import nullcontext

    import nax_prefill_dispatch as dispatch

    from yunshu_engine.kernels import batch_invariant, lane_linear, ragged_kv
    from yunshu_engine.mrope import clear_rope_state
    from yunshu_engine.vlm_engine import VLMEngine

    dispatch.install("base")
    engine = VLMEngine(str(a.model))
    asyncio.run(engine.start())
    batch_invariant.set_active(True)
    ragged_kv.set_dense_lane(True)
    lm = engine._model.language_model
    inner = lm.model
    modules = [
        (name, mod)
        for name, mod in lm.named_modules()
        if isinstance(mod, lane_linear.LaneLinear)
        and mod.output_dims >= lane_linear.NARROW
    ]
    write(
        a.out,
        dict(
            type="engagement",
            kernel_id=engine._prefill_kernel_id(),
            draft_kind=engine._batch_runner.draft_kind,
            projections=len(modules),
            model=str(a.model),
            layers=len(inner.layers),
        ),
    )
    tok = engine._tokenizer
    ids = tok.encode(
        "The quick brown fox jumps over the lazy dog. " * (max(a.tokens) // 8 + 100)
    )[: max(a.tokens)]
    if len(ids) < max(a.tokens):
        raise RuntimeError("not enough input tokens")
    originals = {id(mod): mod.stock for _, mod in modules}
    retained = {}
    references = {}
    cross_chunk_references = {}

    def arm(name):
        import nax_prefill_dispatch as dispatch

        dispatch.install(name if name not in ("planes", "stock") else "base")
        retained.clear()
        for _, mod in modules:
            original = originals[id(mod)]
            mod.stock = original
            if name in ("planes", "stock"):
                w, s, b = original()
                if name == "planes":
                    mx.eval(s, b)
                    retained[id(mod)] = (s, b)
                    mod.stock = lambda original=original, s=s, b=b: (
                        original()[0],
                        s,
                        b,
                    )
                elif name == "stock":
                    mx.eval(w, s, b)
                    retained[id(mod)] = (w, s, b)
                    mod.stock = lambda w=w, s=s, b=b: (w, s, b)
        mx.synchronize()
        mx.clear_cache()

    def forward(ntok, chunk, *, check=False):
        clear_rope_state(engine._model)
        cache = lm.make_cache()
        x = mx.array(ids[:ntok])[None]
        mx.eval(x)
        mx.synchronize()
        start = time.perf_counter()
        chunks = []
        with nullcontext():
            for pos in range(0, ntok, chunk):
                t = time.perf_counter()
                y = inner(x[:, pos : pos + chunk], cache=cache)
                mx.eval(y, [c.state for c in cache])
                chunks.append(time.perf_counter() - t)
        elapsed = time.perf_counter() - start
        import numpy as np

        digest = hashlib.sha256(np.array(y.astype(mx.float32)).tobytes()).hexdigest()
        last_digest = hashlib.sha256(
            np.array(y[:, -1].astype(mx.float32)).tobytes()
        ).hexdigest()
        equal = None
        cross_equal = None
        if check:
            arrays = [value for entry in cache for value in entry.state]
            key = (ntok, chunk)
            if key not in references:
                references[key] = (y, arrays)
            else:
                old_y, old_arrays = references[key]
                if len(old_arrays) != len(arrays):
                    raise RuntimeError("cache state structure changed")
                equal = bool(mx.array_equal(y, old_y).item())
                for old, new in zip(old_arrays, arrays, strict=True):
                    if old is None or new is None:
                        equal = equal and old is new
                    else:
                        equal = equal and bool(mx.array_equal(old, new).item())
                if not equal:
                    raise RuntimeError("layout arm changed hidden output or cache bits")
            if ntok not in cross_chunk_references:
                cross_chunk_references[ntok] = (y[:, -1], arrays)
            else:
                old_last, old_arrays = cross_chunk_references[ntok]
                cross_equal = bool(mx.array_equal(y[:, -1], old_last).item())
                for old, new in zip(old_arrays, arrays, strict=True):
                    if old is None or new is None:
                        cross_equal = cross_equal and old is new
                    else:
                        cross_equal = cross_equal and bool(
                            mx.array_equal(old, new).item()
                        )
        return dict(
            forward_s=elapsed,
            chunk_s=chunks,
            hidden_digest=digest,
            last_hidden_digest=last_digest,
            cache_bit_equal=equal,
            cross_chunk_bit_equal=cross_equal,
            active_bytes=mx.get_active_memory(),
            peak_bytes=mx.get_peak_memory(),
        )

    if a.capture:
        # Capture real layer calls, not the whole model's large buffer set.
        saved = []
        for idx in (0, 3):
            layer = inner.layers[idx]
            cls = type(layer)
            saved.append((layer, cls))
            target = a.capture.with_name(f"{a.capture.stem}-layer{idx}.gputrace")

            def captured(self, *args, _base=cls.__call__, _target=target, **kwargs):
                mx.eval(args[0])
                mx.synchronize()
                mx.metal.start_capture(str(_target))
                try:
                    out = _base(self, *args, **kwargs)
                    mx.eval(out, kwargs["cache"].state)
                finally:
                    mx.metal.stop_capture()
                return out

            layer.__class__ = type(
                cls.__name__ + "Capture", (cls,), {"__call__": captured}
            )
        try:
            arm("base")
            forward(a.chunks[0], a.chunks[0])
        finally:
            for layer, cls in saved:
                layer.__class__ = cls
        write(
            a.out, dict(type="layer_capture", base_path=str(a.capture), layers=[0, 3])
        )

    if a.classes:
        from prefill_profile import Timers, instrument

        arm("base")
        forward(min(a.tokens[0], a.chunks[0]), a.chunks[0])
        timers = Timers()
        instrument(inner, timers)
        result = forward(a.tokens[0], a.chunks[0])
        result["classes"] = {
            k: dict(seconds=v, count=timers.n[k], tflops=timers.flops[k] / v / 1e12)
            for k, v in timers.exclusive().items()
        }
        write(a.out, dict(type="classes", **result))
        return
    for ntok in a.tokens:
        for rep in range(a.reps):
            for chunk in a.chunks if rep % 2 == 0 else list(reversed(a.chunks)):
                order = a.arms if rep % 2 == 0 else list(reversed(a.arms))
                for name in order:
                    arm(name)
                    forward(min(ntok, chunk), chunk)
                    import nax_prefill_dispatch as dispatch

                    before_calls = sum(dispatch.calls.values())
                    result = forward(ntok, chunk, check=True)
                    result["dispatch_calls"] = (
                        sum(dispatch.calls.values()) - before_calls
                    )
                    if (
                        modules
                        and name not in ("base", "planes", "stock")
                        and not result["dispatch_calls"]
                    ):
                        raise RuntimeError("candidate dispatch did not engage")
                    write(
                        a.out,
                        dict(
                            type="model",
                            arm=name,
                            tokens=ntok,
                            chunk=chunk,
                            rep=rep,
                            **result,
                        ),
                    )


def narrow(a, mx):
    from yunshu_engine.kernels import lane_linear
    from yunshu_engine.kernels.tensorfold import lane_qmm

    piece = lane_linear.PIECE
    mx.random.seed(113)
    for m in [512] if a.tiny else [512, 1024, 2048, 4096, 8192]:
        x = mx.random.normal((m, 5120)).astype(mx.bfloat16)
        w = mx.random.normal((48, 5120)).astype(mx.bfloat16)
        mx.eval(x, w)
        for bits in (4, 5, 8):
            q, s, b = mx.quantize(w, bits=bits, group_size=64)
            sbt = lane_qmm.pack_scales(s, b)
            mx.eval(q, sbt)

            def base():
                lane_qmm._xs_cache.clear()
                return mx.concatenate(
                    [
                        lane_qmm.lane_matmul(
                            x[i : i + piece],
                            q,
                            sbt,
                            group=64,
                            row_block=32,
                            row_limit=piece,
                        )
                        for i in range(0, m, piece)
                    ]
                )

            def full():
                lane_qmm._xs_cache.clear()
                return lane_qmm.lane_matmul(
                    x,
                    q,
                    sbt,
                    group=64,
                    row_block=32,
                    row_limit=piece,
                    prefill_narrow=m > 512,
                )

            ref, got = base(), full()
            mx.eval(ref, got)
            equal = bool(mx.array_equal(ref, got).item())
            write(a.out, dict(type="narrow_parity", m=m, bits=bits, bit_equal=equal))
            if not equal:
                raise RuntimeError("full narrow span changed row arithmetic")
            for rep in range(a.reps):
                arms = [("pieces", base), ("full", full)]
                for name, fn in arms if rep % 2 == 0 else list(reversed(arms)):
                    mx.eval(fn())
                    outs = [fn() for _ in range(10)]
                    mx.synchronize()
                    start = time.perf_counter()
                    mx.eval(outs)
                    write(
                        a.out,
                        dict(
                            type="narrow",
                            m=m,
                            bits=bits,
                            arm=name,
                            rep=rep,
                            seconds=(time.perf_counter() - start) / 10,
                        ),
                    )
                    del outs


def main():
    p = parser()
    a = p.parse_args()
    if a.reps < 1 or min(a.tokens + a.chunks) < 1:
        p.error("positive reps/tokens/chunks required")
    if a.mode == "model" and (not a.model or not (a.model / "config.json").is_file()):
        p.error("model mode requires a local --model/config.json")
    if a.dry_run:
        print(json.dumps(vars(a), default=str))
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import mlx.core as mx

    mx.set_default_device(mx.gpu)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.out.exists():
        raise FileExistsError(a.out)
    write(
        a.out,
        dict(
            type="start",
            args=vars(a),
            mlx=importlib.metadata.version("mlx"),
            device=mx.metal.device_info(),
            sha=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            script_hashes={
                name: hashlib.sha256(
                    Path(__file__).with_name(name).read_bytes()
                ).hexdigest()
                for name in (
                    "nax_prefill_probe.py",
                    "nax_qmm_tiles.py",
                    "prefill_profile.py",
                    "nax_prefill_dispatch.py",
                )
            },
        ),
    )
    if a.mode == "micro":
        micro(a, mx)
    elif a.mode == "narrow":
        narrow(a, mx)
    else:
        model(a, mx)
    write(a.out, dict(type="complete", success=True))


if __name__ == "__main__":
    main()
