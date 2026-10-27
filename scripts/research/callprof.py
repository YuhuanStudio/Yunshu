"""Serve the real CLI with wall-time wrappers on engine-side functions (no arithmetic change).

Every call slower than 2 ms inside the wrapped modules is logged as
`CALLPROF <depth> <ms> <qualname>` so a warm request's host timeline can be read from the
server log. Nested calls are logged too (inclusive times). No sync is added: GPU time shows up
in whichever call blocks on it."""

import functools
import importlib
import inspect
import logging
import sys
import threading
import time

MODULES = [
    "yunshu_engine.vlm_batch_runner",
    "yunshu_engine.vlm_engine",
    "yunshu_engine.apc_manager",
    "yunshu_engine.cache_prefill",
    "yunshu_engine.dflash_fast",
    "yunshu_engine.dflash_apc_window",
    "yunshu_engine.dflash_context",
    "yunshu_engine.dflash_copy",
    "yunshu_engine.dflash_ring",
    "yunshu_engine.mtp_lane",
    "yunshu_engine.tree_verify",
    "yunshu_engine.next_turn",
    "yunshu_engine.idle_memory",
    "mlx_vlm.generate.ar",
    "mlx_vlm.apc",
    "mlx_vlm.speculative.utils",
]
log = logging.getLogger("callprof")
local = threading.local()
SKIP = {
    "__init_subclass__",
    "__class_getitem__",
    "__repr__",
    "__getattr__",
    "__setattr__",
}


def wrap(fn, name):
    if getattr(fn, "_callprof", False):
        return fn

    @functools.wraps(fn)
    def inner(*a, **k):
        depth = getattr(local, "depth", 0)
        local.depth = depth + 1
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            local.depth = depth
            ms = (time.perf_counter() - t0) * 1000
            if ms >= 2.0:
                log.warning("CALLPROF %d %.1f %s", depth, ms, name)

    inner._callprof = True
    return inner


def install():
    for modname in MODULES:
        try:
            mod = importlib.import_module(modname)
        except Exception as e:  # noqa: BLE001
            print("callprof skip", modname, e, file=sys.stderr)
            continue
        for attr, obj in list(vars(mod).items()):
            if inspect.isfunction(obj) and obj.__module__ == modname:
                if inspect.isgeneratorfunction(obj):
                    continue
                setattr(mod, attr, wrap(obj, f"{modname}.{attr}"))
            elif inspect.isclass(obj) and obj.__module__ == modname:
                for m, f in list(vars(obj).items()):
                    if (
                        m in SKIP
                        or not inspect.isfunction(f)
                        or inspect.isgeneratorfunction(f)
                    ):
                        continue
                    try:
                        setattr(obj, m, wrap(f, f"{modname}.{attr}.{m}"))
                    except (TypeError, AttributeError):
                        pass


if __name__ == "__main__":
    install()
    from yunshu_cli import main as cli

    sys.argv = ["yunshu", *sys.argv[1:]]
    cli()
