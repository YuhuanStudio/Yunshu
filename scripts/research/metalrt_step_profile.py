"""Per-step host/GPU split of a decode or verify step, with the command-buffer interposer (metalrt_trace.mm).

Run through gpuq with the dylib inserted:
    env DYLD_INSERT_LIBRARIES=$B/libmetalrt_trace.dylib PYTHONPATH=python python scripts/research/metalrt_step_profile.py \
        dense --model DIR --rows 1 2 16 --ctx 1024 --out F.jsonl
kinds: chain (N dependent adds, self-test), dense (Qwen3.5-family target via mlx_vlm + Yunshu lane stack),
       flash (Qwen3.8-Flash-Next via VLMEngine, PLE on SSD).
Per row: serial split (graph build / async_eval encode+commit / wait), pipelined ms per step, and the interposer's
window over the pipelined run: command buffers, dispatches, GPU span, idle between buffers, commit() CPU time.
Fails closed: nonzero exit if the interposer is not loaded or reports zero command buffers.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import os

# must be the image DYLD_INSERT_LIBRARIES loaded: a second copy would have its own (unswizzled) state
DYLIB = Path(
    os.environ.get("DYLD_INSERT_LIBRARIES")
    or "/Volumes/P5Plus/yunshu-build/metalrt/libmetalrt_trace.dylib"
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "kind", choices=["chain", "dense", "flash", "flashverify", "dflash"]
    )
    ap.add_argument("--model", type=Path)
    ap.add_argument("--drafter", type=Path, help="dflash: drafter dir")
    ap.add_argument(
        "--replay",
        action="store_true",
        help="capture one step and replay it serial/concurrent/ICB + per-kernel",
    )
    ap.add_argument("--rows", type=int, nargs="+", default=[1])
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--n", type=int, default=3000, help="chain: kernels per step")
    ap.add_argument("--dylib", type=Path, default=DYLIB)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.kind != "chain" and a.model is None:
        ap.error("--model is required")
    if a.kind == "dflash" and a.drafter is None:
        ap.error("--drafter is required")
    return a


class Tracer:
    def __init__(self, path):
        self.lib = ctypes.CDLL(str(path))
        self.lib.metalrt_begin.restype = None
        self.lib.metalrt_replay.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self.lib.metalrt_replay.restype = ctypes.c_int
        self.lib.metalrt_profile.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self.lib.metalrt_profile.restype = ctypes.c_int
        self.lib.metalrt_cap_end.restype = ctypes.c_int
        self.lib.metalrt_end.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.metalrt_end.restype = ctypes.c_int

    def cap_begin(self):
        self.lib.metalrt_cap_begin()

    def cap_end(self):
        return self.lib.metalrt_cap_end()

    def replay(self, mode, reps=7):
        buf = ctypes.create_string_buffer(4096)
        n = self.lib.metalrt_replay(mode, reps, 1, buf, 4096)
        return json.loads(buf.raw[:n].decode())

    def profile(self, topn=30):
        buf = ctypes.create_string_buffer(1 << 16)
        n = self.lib.metalrt_profile(topn, buf, 1 << 16)
        return json.loads(buf.raw[:n].decode())

    def begin(self):
        self.lib.metalrt_begin()

    def end(self):
        buf = ctypes.create_string_buffer(2048)
        n = self.lib.metalrt_end(buf, 2048)
        return json.loads(buf.raw[:n].decode())


def measure(mx, tracer, make_step, steps, label, extra=None):
    """make_step(prev) -> lazy output of one step (prev = previous evaluated output, for a data dependency)."""
    y = make_step(None)
    mx.eval(y)
    for _ in range(3):
        y = make_step(y)
        mx.eval(y)
    build, enc, wait = [], [], []
    for _ in range(steps):
        t0 = time.perf_counter()
        y = make_step(y)
        t1 = time.perf_counter()
        mx.async_eval(y)
        t2 = time.perf_counter()
        mx.eval(y)
        t3 = time.perf_counter()
        build.append((t1 - t0) * 1e3)
        enc.append((t2 - t1) * 1e3)
        wait.append((t3 - t2) * 1e3)
    mx.synchronize()
    tracer.begin()
    t0 = time.perf_counter()
    nxt = make_step(y)
    mx.async_eval(nxt)
    for _ in range(steps):
        cur = nxt
        nxt = make_step(cur)
        mx.async_eval(nxt)
        mx.eval(cur)
    mx.eval(nxt)
    mx.synchronize()
    wall = (time.perf_counter() - t0) * 1e3
    tr = tracer.end()
    n = steps + 1
    if tr["cbs"] == 0:
        sys.exit("interposer saw zero command buffers (dylib not inserted?)")
    row = {
        "label": label,
        "serial_build_ms": round(statistics.median(build), 3),
        "serial_async_eval_ms": round(statistics.median(enc), 3),
        "serial_wait_ms": round(statistics.median(wait), 3),
        "pipelined_ms": round(wall / n, 3),
        "cbs_per_step": round(tr["cbs"] / n, 2),
        "dispatches_per_step": round(tr["dispatches"] / n, 1),
        "gpu_span_ms_per_step": round(tr["gpu_span_ms"] / n, 3),
        "gpu_busy_union_ms_per_step": round(tr["busy_union_ms"] / n, 3),
        "first_to_last_ms_per_step": round(tr["first_to_last_ms"] / n, 3),
        "idle_between_cb_ms_per_step": round(tr["idle_between_cb_ms"] / n, 3),
        "idle_gaps_over_50us_ms_per_step": round(tr["idle_gaps_over_50us_ms"] / n, 3),
        "commit_cpu_ms_per_step": round(tr["commit_cpu_ms"] / n, 3),
        "gpu_span_us_per_dispatch": round(
            tr["gpu_span_ms"] * 1e3 / max(tr["dispatches"], 1), 3
        ),
    }
    row.update(extra or {})
    return row


def replay_study(a, mx, tracer, make_step, label):
    """Capture one step's dispatch stream and replay it serial / concurrent / ICB; then per-kernel durations."""
    y = make_step(None)
    mx.eval(y)
    mx.synchronize()
    tracer.cap_begin()
    y = make_step(y)
    mx.eval(y)
    mx.synchronize()
    n = tracer.cap_end()
    if n == 0:
        sys.exit("capture recorded zero dispatches")
    names = {
        0: "serial",
        1: "concurrent+captured barriers",
        2: "icb+captured barriers",
        3: "icb all barriers",
    }
    for mode in (0, 1, 2, 3, 0):
        r = tracer.replay(mode)
        if "error" in r:
            sys.exit(f"replay mode {mode} failed: {r}")
        r["label"] = f"{label} replay {names[mode]}"
        emit(a, r)
    pr = tracer.profile(40)
    if "error" in pr:
        sys.exit(f"profile failed: {pr}")
    pr["label"] = f"{label} per-kernel"
    emit(a, pr)


def emit(a, row):
    with a.out.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def run_chain(a, mx, tracer):
    x0 = mx.zeros((5120,), dtype=mx.float32)

    def step(prev):
        x = x0 if prev is None else prev
        for _ in range(a.n):
            x = x + 1.0
        return x

    emit(a, measure(mx, tracer, step, a.steps, f"chain{a.n}"))
    if a.replay:
        replay_study(a, mx, tracer, step, f"chain{a.n}")


def run_dense(a, mx, tracer):
    from mlx_vlm import load

    from yunshu_engine.kernels import omlx, ragged_kv
    from yunshu_engine.kernels.batch_invariant import install as install_invariant
    from yunshu_engine.kernels.batch_invariant import set_active

    model, processor = load(str(a.model))
    lm = model.language_model
    omlx.apply(row_exact=False)
    install_invariant(lm, model=model, packed=omlx.is_nax_available())
    set_active(True)
    ragged_kv.install()
    ragged_kv.set_dense_lane(True)
    ids = [(i * 7919) % 100000 + 1000 for i in range(a.ctx + max(a.rows) + 64)]
    cache = lm.make_cache()
    for s in range(0, a.ctx, 2048):
        lm(mx.array(ids[s : min(s + 2048, a.ctx)])[None], cache=cache)
        mx.eval([c.state for c in cache])
    for T in a.rows:
        toks = mx.array(ids[a.ctx : a.ctx + T])[None]

        def step(prev, toks=toks):
            return mx.argmax(lm(toks, cache=cache).logits[:, -1], axis=-1)

        emit(
            a,
            measure(mx, tracer, step, a.steps, f"dense T={T}", {"T": T, "ctx": a.ctx}),
        )
        if a.replay:
            replay_study(a, mx, tracer, step, f"dense T={T}")


def run_flash(a, mx, tracer):
    from yunshu_engine import settings
    from yunshu_engine.types import EngineConfig
    from yunshu_engine.vlm_engine import VLMEngine

    async def go():
        settings.set_override("YUNSHU_VLM_APC_MEMORY_GB", 1.0)
        engine = VLMEngine(
            str(a.model), EngineConfig(prefill_step_size=512, completion_batch_size=1)
        )
        await engine.start()
        lm = engine._model.language_model
        ids = [(i * 7919) % 200000 + 1000 for i in range(a.ctx + 64)]
        cache = lm.make_cache()
        for i in range(0, a.ctx, 512):
            o = lm(mx.array([ids[i : i + 512]]), cache=cache)
            mx.eval(o.logits)
            mx.eval([c.state for c in cache if hasattr(c, "state")])
        for T in a.rows:
            toks = mx.array([ids[a.ctx : a.ctx + T]])

            def step(prev, toks=toks):
                return mx.argmax(lm(toks, cache=cache).logits[:, -1, :], axis=-1)

            emit(
                a,
                measure(
                    mx, tracer, step, a.steps, f"flash T={T}", {"T": T, "ctx": a.ctx}
                ),
            )
            if a.replay:
                replay_study(a, mx, tracer, step, f"flash T={T}")
        # per-layer async_eval disabled: host tracing vs encode+GPU
        orig = mx.async_eval
        mx.async_eval = lambda *x, **k: None
        try:
            for T in a.rows:
                toks = mx.array([ids[a.ctx : a.ctx + T]])
                b, e = [], []
                for i in range(a.steps + 3):
                    t0 = time.perf_counter()
                    y = mx.argmax(lm(toks, cache=cache).logits[:, -1, :], axis=-1)
                    t1 = time.perf_counter()
                    mx.eval(y)
                    t2 = time.perf_counter()
                    if i >= 3:
                        b.append((t1 - t0) * 1e3)
                        e.append((t2 - t1) * 1e3)
                emit(
                    a,
                    {
                        "label": f"flash T={T} no-layer-async",
                        "trace_ms": round(statistics.median(b), 3),
                        "eval_ms": round(statistics.median(e), 3),
                    },
                )
        finally:
            mx.async_eval = orig

    asyncio.run(go())


def run_flashverify(a, mx, tracer):
    """Flash-Next speculative verify of T rows (the engine's MTP/draft verify forward) on a prefilled cache."""
    from yunshu_engine import settings
    from yunshu_engine.types import EngineConfig
    from yunshu_engine.vlm_engine import VLMEngine

    async def go():
        settings.set_override("YUNSHU_VLM_DRAFT", "off")
        # the oQ3e pack's 6-bit head makes configure_lane refuse the load; this probe times the verify forward only
        from yunshu_engine import qwen4_mtp

        qwen4_mtp.configure_lane = lambda lm, drafter, block: ({}, block)
        import threading

        alive = threading.Event()

        def heartbeat():
            while not alive.wait(20):
                print(
                    "loading...", flush=True
                )  # the 95 GB load is silent for minutes (gpuq stall rule)

        threading.Thread(target=heartbeat, daemon=True).start()
        settings.set_override("YUNSHU_VLM_APC_MEMORY_GB", 1.0)
        engine = VLMEngine(
            str(a.model), EngineConfig(prefill_step_size=512, completion_batch_size=1)
        )
        await engine.start()
        alive.set()
        lm = engine._model.language_model
        ids = [(i * 7919) % 200000 + 1000 for i in range(a.ctx + 64)]
        cache = lm.make_cache()
        for i in range(0, a.ctx, 512):
            o = lm(mx.array([ids[i : i + 512]]), cache=cache)
            mx.eval(o.logits)
            mx.eval([c.state for c in cache if hasattr(c, "state")])
        for T in a.rows:
            toks = mx.array([ids[a.ctx : a.ctx + T]])

            def step(prev, toks=toks):
                hidden, _extra, tx = lm.speculative_verify_hidden(toks, cache)
                tx.abort()
                return hidden

            emit(
                a,
                measure(
                    mx,
                    tracer,
                    step,
                    a.steps,
                    f"flashverify T={T}",
                    {"T": T, "ctx": a.ctx},
                ),
            )
            if a.replay:
                replay_study(a, mx, tracer, step, f"flashverify T={T}")

    asyncio.run(go())


def run_dflash(a, mx, tracer):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mlx_vlm.speculative.drafters import (
        load_drafter,
        validate_drafter_compatibility,
    )
    from probe_wide_verify import setup

    from yunshu_engine import dflash_context, dflash_tree

    model, lm, _ids = setup(str(a.model))
    drafter, kind = load_drafter(str(a.drafter))
    validate_drafter_compatibility(model, drafter, kind)
    dflash_tree.quantize_drafter(drafter, 8)
    dflash_context.install(lm)
    cache = drafter.reset(model)
    cfg = drafter.config
    d = len(cfg.target_layer_ids) * cfg.hidden_size
    mx.random.seed(0)
    hid = mx.random.normal((1, 4, d)).astype(mx.bfloat16)
    mx.eval(hid)

    def step(prev):
        lat = dflash_tree.compute_lattice_gpu(drafter, 1000, hid, cache, 7)
        t, p = dflash_tree.search_tree(lat, 15)
        return mx.concatenate(
            [t.reshape(-1).astype(mx.float32), p.reshape(-1).astype(mx.float32)]
        )

    emit(
        a,
        measure(mx, tracer, step, a.steps, "dflash lattice+search", {"kind": "dflash"}),
    )
    if a.replay:
        replay_study(a, mx, tracer, step, "dflash")


def main(argv=None):
    a = parse_args(argv)
    import mlx.core as mx

    a.out.parent.mkdir(parents=True, exist_ok=True)
    tracer = Tracer(a.dylib)
    {
        "chain": run_chain,
        "dense": run_dense,
        "flash": run_flash,
        "flashverify": run_flashverify,
        "dflash": run_dflash,
    }[a.kind](a, mx, tracer)


if __name__ == "__main__":
    main()
