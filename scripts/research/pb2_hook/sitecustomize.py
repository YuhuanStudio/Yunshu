"""Server-side per-layer prefill timer (research; enabled by PB2_PROF_OUT=<json path>).

Put this directory first on PYTHONPATH. After mlx_vlm's qwen3_5 language module
imports, every decoder layer call with >= 2 rows is evaluated and timed;
totals per (kind, rows) are rewritten to the json every 32 records. Same hook in
both arms: eval-per-layer perturbs both equally.
"""

import contextlib
import importlib.abc
import json
import os
import sys
import time

OUT = os.environ.get("PB2_PROF_OUT")
TARGET = "mlx_vlm.models.qwen3_5.language"


def summarize(acc):
    return {
        k: {"calls": v[0], "seconds": round(v[1], 4)} for k, v in sorted(acc.items())
    }


LSUM = os.environ.get("PB2_LAYERSUM_OUT")


def patch_layersum(mod):
    """Log a checksum of every decoder layer's input and output (rows >= 2): the first layer call
    whose output differs between two trees localizes an arithmetic difference."""
    import mlx.core as mx

    cls = mod.Qwen3_5DecoderLayer
    orig = cls.__call__
    count = [0]

    def summed(self, x, *args, **kw):
        out = orig(self, x, *args, **kw)
        if x.ndim >= 2 and x.shape[-2] >= 2:
            o = out[0] if isinstance(out, (tuple, list)) else out
            fi, fo = x.astype(mx.float32), o.astype(mx.float32)
            vals = mx.stack(
                [mx.sum(fi), mx.max(mx.abs(fi)), mx.sum(fo), mx.max(mx.abs(fo))]
            ).tolist()
            with open(LSUM, "a") as f:
                f.write(
                    f"{count[0]} rows={x.shape[-2]} in={vals[0]:.9g} inmax={vals[1]:.9g} "
                    f"out={vals[2]:.9g} outmax={vals[3]:.9g}\n"
                )
            count[0] += 1
        return out

    cls.__call__ = summed


def patch(mod):
    import mlx.core as mx

    if LSUM:
        return patch_layersum(mod)
    cls = mod.Qwen3_5DecoderLayer
    orig = cls.__call__
    acc: dict = {}
    n = [0]

    def timed(self, x, *args, **kw):
        if x.ndim < 2 or x.shape[-2] < 2:
            return orig(self, x, *args, **kw)
        mx.eval(x)
        t0 = time.perf_counter()
        out = orig(self, x, *args, **kw)
        mx.eval(out)
        key = f"{'gdn' if self.is_linear else 'attn'}@{x.shape[-2]}"
        c = acc.setdefault(key, [0, 0.0])
        c[0] += 1
        c[1] += time.perf_counter() - t0
        n[0] += 1
        if n[0] % 8 == 0:
            with open(OUT, "w") as f:
                json.dump(summarize(acc), f)
        return out

    cls.__call__ = timed


DRV_OUT = os.environ.get("PB2_DRV_OUT")
DRV = "yunshu_engine.round_driver.driver"


def patch_driver(mod):
    """Log wall time of RoundDriver.add / _restore / _prefill_step (research timer)."""
    cls = mod.RoundDriver

    def wrap(name):
        orig = getattr(cls, name)

        def timed(self, *args, **kw):
            t0 = time.perf_counter()
            try:
                return orig(self, *args, **kw)
            finally:
                with open(DRV_OUT, "a") as f:
                    f.write(f"{name} {time.perf_counter() - t0:.4f} at {t0:.4f}\n")

        setattr(cls, name, timed)

    for name in ("add", "_prefill_step", "_restore", "_draw", "_record_tail"):
        wrap(name)

    def wrap_fn(holder, name, label):
        orig = getattr(holder, name)

        def timed(*args, **kw):
            t0 = time.perf_counter()
            try:
                return orig(*args, **kw)
            finally:
                with open(DRV_OUT, "a") as f:
                    f.write(f"{label} {time.perf_counter() - t0:.4f}\n")

        setattr(holder, name, timed)

    class _MxProxy:
        """The driver module's ``mx`` with a timed ``eval`` (host wait for the GPU)."""

        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

        def eval(self, *args):
            t0 = time.perf_counter()
            try:
                return self._real.eval(*args)
            finally:
                with open(DRV_OUT, "a") as f:
                    f.write(f"mx_eval {time.perf_counter() - t0:.4f}\n")

    mod.mx = _MxProxy(mod.mx)
    wrap("_seed_window")
    orig_new = cls._new_coordinator

    def new_coordinator(self, *args, **kw):
        co = orig_new(self, *args, **kw)
        for meth in ("lookup", "prepare_prefill", "checkpoint_lengths"):
            fn = getattr(co, meth)

            def timed(*a, _fn=fn, _m=meth, **k):
                t0 = time.perf_counter()
                try:
                    return _fn(*a, **k)
                finally:
                    with open(DRV_OUT, "a") as f:
                        f.write(f"apc_{_m} {time.perf_counter() - t0:.4f}\n")

            setattr(co, meth, timed)
        return co

    cls._new_coordinator = new_coordinator
    wrap_fn(mod, "reserve_prompt_kv", "reserve_kv")
    wrap_fn(mod, "lane_forward", "lane_forward")
    wrap_fn(mod.replay, "restore", "replay_restore")


class _DrvFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != DRV:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            patch_driver(module)

        spec.loader.exec_module = exec_module
        return spec


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            patch(module)

        loader.exec_module = exec_module
        return spec


if DRV_OUT:
    import importlib.util

    sys.meta_path.insert(0, _DrvFinder())

if OUT or LSUM:
    import importlib.util

    sys.meta_path.insert(0, _Finder())


TRACE_OUT = os.environ.get("PB2_TRACE_OUT")
TL_OUT = os.environ.get("PB2_TL_OUT")
REPLAY = "yunshu_engine.prompt_end_replay"


def wrap_tl(holder, name, label):
    """Time-line wrapper (start, duration) written to PB2_TL_OUT; skipped when absent.

    Keeps staticmethod / classmethod descriptors (the runner's ``_next_preemptible`` is a
    staticmethod: wrapping it as a plain function would pass ``self`` as its argument)."""
    import inspect

    raw = inspect.getattr_static(holder, name, None)
    if raw is None:
        return
    kind = type(raw) if isinstance(raw, (staticmethod, classmethod)) else None
    orig = raw.__func__ if kind is not None else raw

    def timed(*args, **kw):
        t0 = time.perf_counter()
        try:
            return orig(*args, **kw)
        finally:
            with open(TL_OUT, "a") as f:
                f.write(f"{label} {time.perf_counter() - t0:.4f} at {t0:.4f}\n")

    setattr(holder, name, kind(timed) if kind is not None else timed)


class _ReplayFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != REPLAY:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            wrap_tl(module, "open_session", "replay.open_session")
            wrap_tl(module.Session, "begin", "replay.begin")
            wrap_tl(module.Session, "_replay", "replay._replay")
            wrap_tl(module.Session, "store", "replay.store")

        spec.loader.exec_module = exec_module
        return spec


POOL = "yunshu_engine.kernels.buffer_cache"


def wrap_pool(mod):
    """Log restore-pool traffic: offers (with the pooled byte count) and takes (hit or miss)."""

    def log(line):
        with open(TL_OUT, "a") as f:
            f.write(f"{line} at {time.perf_counter():.4f}\n")

    take = mod.take_restore_buffer

    def timed_take(a, capacity):
        out = take(a, capacity)
        log(
            f"pool.take {'hit' if out is not None else 'miss'} shape={tuple(a.shape)} cap={capacity} 0.0000"
        )
        return out

    mod.take_restore_buffer = timed_take
    offer_arrays = getattr(mod, "offer_arrays", None)
    if offer_arrays is not None:

        def timed_offer(arrays, *, aged=False):
            ok = offer_arrays(arrays, aged=aged)
            size = sum(a.nbytes for a in arrays if a is not None)
            log(
                f"pool.offer {'taken' if ok else 'refused'} bytes={size} limit={mod._STATE['limit']} 0.0000"
            )
            return ok

        mod.offer_arrays = timed_offer
    offer = mod.offer_restore_buffers

    def timed_offer_caches(caches):
        before = mod.restore_buffer_bytes()
        offer(caches)
        log(f"pool.offer_caches bytes {before}->{mod.restore_buffer_bytes()} 0.0000")

    mod.offer_restore_buffers = timed_offer_caches


class _PoolFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != POOL:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            wrap_pool(module)

        spec.loader.exec_module = exec_module
        return spec


if TL_OUT:
    import importlib.util

    sys.meta_path.insert(0, _ReplayFinder())
    sys.meta_path.insert(0, _PoolFinder())
RUNNER = "yunshu_engine.vlm_batch_runner"
_TRACED = (
    "t_submit",
    "t_admit",
    "t_prefill_start",
    "t_first",
    "t_last",
    "t_prefill_end",
)


ATTN_OUT = os.environ.get("PB2_ATTN_OUT")


def patch_attn():
    """Log checksums of the decode-side attention calls (query, keys, output of window token 0):
    the DecodeBatch path (ragged_decode_attention) and the tree path (tree_verify.tree_attention)
    see the same token over the same keys, so the first line whose output differs between two
    runs names the first differing call."""
    import mlx.core as mx

    from yunshu_engine import tree_verify as tv
    from yunshu_engine.kernels import ragged_attention as ra

    count = [0]

    from mlx_vlm.models.qwen3_5 import language as _lang

    mlp_orig = _lang.Qwen3_5MLP.__call__
    mlp_count = [0]

    def mlp_call(self, x, *a, **kw):
        out = mlp_orig(self, x, *a, **kw)
        try:
            xi = x.reshape(-1, x.shape[-1])[0].astype(mx.float32)
            oo = out.reshape(-1, out.shape[-1])[0].astype(mx.float32)
            vals = mx.stack([mx.sum(xi), mx.sum(oo), mx.max(mx.abs(oo))]).tolist()
            rows = int(x.reshape(-1, x.shape[-1]).shape[0])
            with open(ATTN_OUT, "a") as f:
                f.write(
                    f"M{mlp_count[0]} mlp rows={rows} in={vals[0]:.9g} out={vals[1]:.9g} omax={vals[2]:.9g}\n"
                )
            mlp_count[0] += 1
        except Exception as e:
            with open(ATTN_OUT, "a") as f:
                f.write(f"mlp-probe-error {e!r}\n")
        return out

    _lang.Qwen3_5MLP.__call__ = mlp_call

    vcls = type(_lang._EXACT_SPECULATIVE_VERIFIER)
    ff_orig = vcls._feed_forward

    def ff_call(self, mlp, x, *a, **kw):
        out = ff_orig(self, mlp, x, *a, **kw)
        try:
            xi = x.reshape(-1, x.shape[-1])[0].astype(mx.float32)
            oo = out.reshape(-1, out.shape[-1])[0].astype(mx.float32)
            vals = mx.stack([mx.sum(xi), mx.sum(oo), mx.max(mx.abs(oo))]).tolist()
            rows = int(x.reshape(-1, x.shape[-1]).shape[0])
            with open(ATTN_OUT, "a") as f:
                f.write(
                    f"M{mlp_count[0]} mlp rows={rows} in={vals[0]:.9g} out={vals[1]:.9g} omax={vals[2]:.9g}\n"
                )
            mlp_count[0] += 1
        except Exception as e:
            with open(ATTN_OUT, "a") as f:
                f.write(f"ff-probe-error {e!r}\n")
        return out

    vcls._feed_forward = ff_call

    def line(kind, q, k, o, n):
        vals = mx.stack(
            [
                mx.sum(q.astype(mx.float32)),
                mx.sum(k.astype(mx.float32)),
                mx.sum(o.astype(mx.float32)),
                mx.max(mx.abs(o.astype(mx.float32))),
            ]
        ).tolist()
        with open(ATTN_OUT, "a") as f:
            f.write(
                f"{count[0]} {kind} n={n} q={vals[0]:.9g} k={vals[1]:.9g} "
                f"o={vals[2]:.9g} omax={vals[3]:.9g}\n"
            )
        count[0] += 1

    orig_ragged = ra.ragged_decode_attention

    def ragged(queries, keys, values, lengths, scale, *a, **kw):
        out = orig_ragged(queries, keys, values, lengths, scale, *a, **kw)
        try:
            slots = kw.get("slots")
            slot = int(slots[0]) if slots is not None else 0
            n = int(lengths[0])
            line(
                "ragged",
                queries[0, :, 0],
                keys[slot, :, :n],
                out[0, :, 0],
                n,
            )
        except Exception as e:  # a probe only
            with open(ATTN_OUT, "a") as f:
                f.write(f"ragged-probe-error {e!r}\n")
        return out

    ra.ragged_decode_attention = ragged

    orig_tree = tv.tree_attention

    def tree(queries, cache, scale, shape, n0, *a, **kw):
        out = orig_tree(queries, cache, scale, shape, n0, *a, **kw)
        try:
            n = int(n0) + 1
            line(
                "tree",
                queries[0, :, 0],
                cache.keys[0, :, :n],
                out[0, :, 0],
                n,
            )
        except Exception as e:
            with open(ATTN_OUT, "a") as f:
                f.write(f"tree-probe-error {e!r}\n")
        return out

    tv.tree_attention = tree


STEP_OUT = os.environ.get("PB2_STEP_OUT")


def patch_steps():
    """Per decode-step wall time of the driver (rows in the batch at the step, seconds, seconds spent in
    DecodeBatch._gdn when PB2_GDN_EVAL=1: that mode evaluates around the layer, so its step times are
    inflated and only the _gdn share is read from it)."""
    import mlx.core as mx

    from yunshu_engine.round_driver import batch as bt
    from yunshu_engine.round_driver import driver as drv

    gdn = [0.0]
    if os.environ.get("PB2_GDN_EVAL"):
        orig_gdn = bt.DecodeBatch._gdn

        def timed_gdn(self, *args, **kw):
            xn = next(a for a in args if hasattr(a, "ndim") and a.ndim == 3)
            mx.eval(xn)
            t0 = time.perf_counter()
            out = orig_gdn(self, *args, **kw)
            mx.eval(out)
            gdn[0] += time.perf_counter() - t0
            return out

        bt.DecodeBatch._gdn = timed_gdn

    orig = drv.RoundDriver._decode_step

    def step(self):
        gdn[0] = 0.0
        rows = len(self.batch.rows)
        t0 = time.perf_counter()
        out = orig(self)
        with open(STEP_OUT, "a") as f:
            f.write(f"{rows} {time.perf_counter() - t0:.6f} {gdn[0]:.6f}\n")
        return out

    drv.RoundDriver._decode_step = step


def patch_runner(mod):
    """Log each request's stats timestamps as they are set (same runner file in both arms)."""
    if ATTN_OUT:
        patch_attn()
    if STEP_OUT:
        patch_steps()
    cls = next(
        c
        for c in vars(mod).values()
        if isinstance(c, type) and "t_submit" in getattr(c, "__dataclass_fields__", {})
    )
    orig = cls.__setattr__
    seen: dict = {}

    def traced(self, name, value):
        orig(self, name, value)
        if name in _TRACED and value:
            if name == "t_last":
                seen[id(self)] = seen.get(id(self), 0) + 1
                if seen[id(self)] > 3:
                    return  # only the first few token stamps
            with open(TRACE_OUT, "a") as f:
                f.write(
                    f"{time.perf_counter():.4f} {name} req={id(self) % 100000} "
                    f"val={value:.4f} pt={getattr(self, 'prefill_total', 0)} "
                    f"cached={getattr(self, 'cached_tokens', 0)}\n"
                )

    cls.__setattr__ = traced
    if TL_OUT:
        for name in (
            "_admit",
            "_step_driver",
            "_drive_slice_body",
            "_next_preemptible",
            "_emit_ready_first",
            "_finish",
            "_admit_driver",
        ):
            wrap_tl(mod.VLMBatchRunner, name, f"runner.{name}")
    if MEM_OUT:
        wrap_mem(mod.VLMBatchRunner)


MEM_OUT = os.environ.get("PB2_MEM_OUT")
GIB = float(1 << 30)



def _dtype_gib(arrays) -> dict:
    """GiB per dtype over distinct arrays (decode KV: int8 codes + fp16 scales vs bf16)."""
    seen, out = set(), {}
    for a in arrays:
        if a is None or id(a) in seen:
            continue
        seen.add(id(a))
        out[str(a.dtype)] = out.get(str(a.dtype), 0) + int(a.nbytes)
    return {k: round(v / GIB, 4) for k, v in out.items()}


def _row_cache_arrays(drv) -> list:
    """K / V (and int8 scales) of the caches the driver rows still hold (the prefill caches)."""
    out = []
    for r in drv.rows:
        for c in r.cache or ():
            for name in ("keys", "values", "k_scales", "v_scales"):
                a = getattr(c, name, None)
                if a is not None:
                    out.append(a)
    return out


def holders(runner) -> dict:
    """GiB held by the engine's known retainers (each guarded: a missing module is skipped)."""
    out: dict = {}

    def put(name, fn):
        with contextlib.suppress(Exception):
            out[name] = round(fn() / GIB, 4)

    put("apc_resident", lambda: runner.apc_manager.resident_bytes())
    from yunshu_engine.kernels import buffer_cache

    put("restore_pool", buffer_cache.restore_buffer_bytes)
    from yunshu_engine import dflash_apc_window, prompt_end_replay

    put("replay_main", prompt_end_replay.resident_bytes)
    put("dflash_windows", dflash_apc_window.resident_bytes)
    try:
        from yunshu_engine.round_driver import replay as driver_replay

        put("replay_driver", driver_replay.resident_bytes)
    except Exception:
        pass
    for name in ("driver", "_aux_driver"):
        drv = getattr(runner, name, None)
        if drv is not None:
            put(
                f"{name}_batch",
                lambda drv=drv: sum(a.nbytes for a in drv.batch.arrays()),
            )
            with contextlib.suppress(Exception):
                out[f"{name}_batch_dtypes"] = _dtype_gib(drv.batch.arrays())
            with contextlib.suppress(Exception):
                out[f"{name}_prefill_rows_dtypes"] = _dtype_gib(_row_cache_arrays(drv))
            put(
                f"{name}_rows",
                lambda drv=drv: sum(
                    a.nbytes
                    for r in drv.rows
                    if r.cache
                    for c in r.cache
                    for a in (getattr(c, "keys", None), getattr(c, "values", None))
                    if a is not None
                ),
            )
    return out


def wrap_mem(cls):
    """Log mlx active / cache / peak memory and the known retainers around requests and every 4 s."""
    import threading

    import mlx.core as mx

    state = {"runner": None, "thread": None}

    def snap(tag):
        r = state["runner"]
        rec = {
            "t": round(time.perf_counter(), 3),
            "tag": tag,
            "active": round(mx.get_active_memory() / GIB, 3),
            "cache": round(mx.get_cache_memory() / GIB, 3),
            "peak": round(mx.get_peak_memory() / GIB, 3),
        }
        rec.update(holders(r) if r is not None else {})
        with open(MEM_OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def tick():
        while True:
            time.sleep(4.0)
            with contextlib.suppress(Exception):
                snap("tick")

    for name in ("_admit", "_drained"):
        orig = getattr(cls, name)

        def wrapped(self, *args, _orig=orig, _name=name, **kw):
            state["runner"] = self
            if state["thread"] is None:
                state["thread"] = threading.Thread(target=tick, daemon=True)
                state["thread"].start()
            if _name == "_admit":
                snap("admit")
                mx.reset_peak_memory()
            try:
                return _orig(self, *args, **kw)
            finally:
                if _name == "_drained":
                    snap("drained")

        setattr(cls, name, wrapped)


class _RunnerFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != RUNNER:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            patch_runner(module)

        spec.loader.exec_module = exec_module
        return spec


if TRACE_OUT:
    import importlib.util

    sys.meta_path.insert(0, _RunnerFinder())


APC_OUT = os.environ.get("PB2_APC_OUT")
APC = "yunshu_engine.apc_manager"


def patch_apc(mod):
    """Time APC coordinator calls (same file in both arms); calls over 20 ms also get a
    cProfile top-12 (cumulative) so the Python-side share is visible."""
    import cProfile
    import io
    import pstats

    depth = [0]

    def wrap(holder, name):
        orig = getattr(holder, name)

        def timed(*args, **kw):
            # cProfile cannot nest (Python 3.12+): only the outermost APC call is profiled
            outer = not depth[0] and not os.environ.get("PB2_APC_NOPROF")
            prof = cProfile.Profile() if outer else None
            t0 = time.perf_counter()
            depth[0] += 1
            if prof is not None:
                prof.enable()
            result = "raised"
            try:
                result = orig(*args, **kw)
                return result
            finally:
                depth[0] -= 1
                if prof is not None:
                    prof.disable()
                dur = time.perf_counter() - t0
                shown = (
                    result
                    if isinstance(result, (bool, type(None)))
                    else type(result).__name__
                )
                n_ids = (
                    len(args[1]) if len(args) > 1 and hasattr(args[1], "__len__") else 0
                )
                with open(APC_OUT, "a") as f:
                    f.write(f"apc.{name} {dur:.4f} at {t0:.4f} ret={shown} n={n_ids}\n")
                    if prof is not None and dur > 0.02:
                        buf = io.StringIO()
                        stats = pstats.Stats(prof, stream=buf)
                        stats.sort_stats("cumulative").print_stats(12)
                        for ln in buf.getvalue().splitlines():
                            if ln.strip():
                                f.write("   | " + ln + "\n")

        setattr(holder, name, timed)

    for name in ("lookup", "store_checkpoint", "checkpoint_lengths", "prepare_prefill"):
        wrap(mod._Coordinator, name)


class _ApcFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != APC:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            patch_apc(module)

        spec.loader.exec_module = exec_module
        return spec


if APC_OUT:
    import importlib.util

    sys.meta_path.insert(0, _ApcFinder())
