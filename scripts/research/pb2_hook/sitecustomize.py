"""Server-side per-layer prefill timer (research; enabled by PB2_PROF_OUT=<json path>).

Put this directory first on PYTHONPATH. After mlx_vlm's qwen3_5 language module
imports, every decoder layer call with >= 2 rows is evaluated and timed;
totals per (kind, rows) are rewritten to the json every 32 records. Same hook in
both arms: eval-per-layer perturbs both equally.
"""

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


def patch(mod):
    import mlx.core as mx

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

if OUT:
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


if TL_OUT:
    import importlib.util

    sys.meta_path.insert(0, _ReplayFinder())
RUNNER = "yunshu_engine.vlm_batch_runner"
_TRACED = (
    "t_submit",
    "t_admit",
    "t_prefill_start",
    "t_first",
    "t_last",
    "t_prefill_end",
)


def patch_runner(mod):
    """Log each request's stats timestamps as they are set (same runner file in both arms)."""
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
