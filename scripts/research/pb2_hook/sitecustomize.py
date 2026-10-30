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
    return {k: {"calls": v[0], "seconds": round(v[1], 4)} for k, v in sorted(acc.items())}


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


if OUT:
    import importlib.util

    sys.meta_path.insert(0, _Finder())
