"""oMLX cold-vs-reuse micro-bench (head-to-head). Emits @@RESULTOMLX@@ <json>.

oMLX (reference/omlx) is pinned to older mlx-lm/mlx-vlm/openai-harmony; we stub
the version-mismatched OPTIONAL imports it pulls in at module load (none are used
by an LLM-only benchmark). Its prefix cache only activates when
SchedulerConfig.paged_ssd_cache_dir is set. Same cold/reuse protocol as ours.

Run:  PYTHONPATH=.:./reference/omlx OMLX_MODEL=... OMLX_SSD=... uv run python scripts/_bench_omlx.py
"""
import asyncio
import importlib.abc
import importlib.util
import json
import logging
import os
import sys
import time
import types

logging.basicConfig(level=logging.ERROR)

# --- stub optional deps oMLX imports but our LLM bench never exercises ---
_STUB_PREFIXES = ("openai_harmony",)


def _mk(name):
    m = types.ModuleType(name)
    m.__path__ = []
    m.__getattr__ = lambda n: (type(n, (), {}) if n[:1].isupper() else (lambda *a, **k: None))
    return m


class _Loader(importlib.abc.Loader):
    def create_module(self, spec):
        return _mk(spec.name)

    def exec_module(self, module):
        pass


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in _STUB_PREFIXES:
            return importlib.util.spec_from_loader(name, _Loader(), is_package=True)
        return None


sys.meta_path.append(_Finder())


def _ensure(modname, attrs):
    m = sys.modules.get(modname) or types.ModuleType(modname)
    sys.modules[modname] = m
    for a, v in attrs.items():
        if not hasattr(m, a):
            setattr(m, a, v)
    return m


_s = lambda *a, **k: None  # noqa: E731
_ensure("mlx_vlm.speculative", {"load_drafter": _s, "_mtp_rounds": _s, "_mtp_rounds_batch": _s})
_ensure("mlx_vlm.speculative.utils", {"_mtp_rounds": _s, "_mtp_rounds_batch": _s})

MODEL = os.environ["OMLX_MODEL"]
QUERY = "List the first 6 even numbers, comma separated."


def _doc(t):
    return f"Reference document {t}. " + (
        "Photosynthesis converts sunlight into chemical energy stored in glucose. " * 110)


async def main():
    name = os.path.basename(MODEL.rstrip("/"))
    try:
        from omlx.engine.batched import BatchedEngine
        from omlx.scheduler import SchedulerConfig
        sc = SchedulerConfig()
        sc.paged_ssd_cache_dir = os.environ.get("OMLX_SSD", "/tmp/omlx_pc")
        sc.paged_cache_block_size = 128
        e = BatchedEngine(model_name=MODEL, scheduler_config=sc)
        await e.start()
    except Exception as ex:
        print("@@RESULTOMLX@@ " + json.dumps(
            {"model": name, "status": f"LOAD FAILED: {type(ex).__name__}: {str(ex)[:80]}"}))
        return

    async def chat(s, u, mt):
        t = time.perf_counter()
        o = await e.chat(messages=[{"role": "system", "content": s},
                                   {"role": "user", "content": u}],
                         max_tokens=mt, temperature=0.0)
        return time.perf_counter() - t, o

    try:
        await chat(_doc("W"), "hi", 2)
        colds = []
        for i in range(3):
            d, _ = await chat(_doc(f"C{i}"), QUERY, 1)
            colds.append(d)
        cold = min(colds)
        _, rref = await chat(_doc("R"), QUERY, 16)
        ref = rref.text.strip()
        await chat(P := _doc("PRIMARY"), "Summarize briefly.", 8)
        rs = []
        for _ in range(3):
            d, _o = await chat(P, QUERY, 1)
            rs.append(d)
        reuse = min(rs)
        _, ot = await chat(P, QUERY, 16)
        res = {
            "model": name,
            "cold_ms": round(cold * 1000, 1),
            "reuse_ms": round(reuse * 1000, 1),
            "speedup": round(cold / reuse, 2) if reuse else 0,
            "lossless": ot.text.strip() == ref,
            "status": "ok",
        }
    except Exception as ex:
        res = {"model": name, "status": f"RUN FAILED: {type(ex).__name__}: {str(ex)[:80]}"}
    print("@@RESULTOMLX@@ " + json.dumps(res))
    import contextlib
    with contextlib.suppress(Exception):
        await e.stop()


if __name__ == "__main__":
    asyncio.run(main())
