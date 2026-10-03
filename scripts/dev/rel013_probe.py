"""Hosted Metal diagnostics; no model downloads."""
import mlx.core as mx
import numpy as np

print("DEVICE", mx.metal.device_info(), flush=True)
for device in (mx.gpu, mx.cpu):
    with mx.stream(device):
        for n in (1, 8, 200, 600):
            row = np.full((1, 248320), -np.inf, dtype=np.float32)
            row[0, :5] = np.arange(5)
            x = mx.broadcast_to(mx.array(row), (n, 248320))
            out = np.array(mx.argmax(x, axis=-1))
            print("ARGMAX", device, n, np.unique(out).tolist(), flush=True)
        from yunshu_engine.keyed_sampling import gumbel
        g = np.array(gumbel(0, mx.arange(8), 248320))
        print("GUMBEL", device, np.isfinite(g).all(), g.min(), g.max(), flush=True)

# Separate model-prefix arithmetic from APC snapshot/copy/storage.
import runpy

model = runpy.run_path("tests/unit/test_apc_manager_e2e.py")["model"]

for device in (mx.gpu, mx.cpu):
    mx.set_default_device(device)
    for dtype in (mx.bfloat16, mx.float16, mx.float32):
        lm = model.__wrapped__().language_model
        lm.set_dtype(dtype)
        ids = mx.array([[i % 797 + 3 for i in range(300)]])
        full = lm(ids, cache=lm.make_cache()).logits[:, -1]
        cache = lm.make_cache()
        lm(ids[:, :259], cache=cache)
        split = lm(ids[:, 259:], cache=cache).logits[:, -1]
        mx.eval(full, split)
        print("PREFIX", device, dtype, "maxerr", mx.max(mx.abs(full - split)).item(),
              "tokens", mx.argmax(full, -1).item(), mx.argmax(split, -1).item(), flush=True)
        del lm, cache, full, split
    mx.clear_cache()

import tempfile
from pathlib import Path
from yunshu_engine.vlm_batch_runner import RowParams
from yunshu_engine.keyed_sampling import KeyedSampler

mx.set_default_device(mx.gpu)
for n in (8, 200, 600):
    row = np.full((1, 248320), -np.inf, dtype=np.float32)
    row[0, :5] = 0
    lp = mx.broadcast_to(mx.array(row), (n, 248320))
    draw = KeyedSampler(RowParams(temperature=1), 0).sample_positions(lp, range(n))
    print("DRAW", n, np.unique(np.array(draw)).tolist(), flush=True)
    mx.clear_cache()

namespace = runpy.run_path("tests/unit/test_apc_manager_e2e.py")
original = namespace["_run"]
def traced(runner, ids):
    out, stats = original(runner, ids)
    print("APC", len(ids), stats.cache_tier, stats.cached_tokens, out, flush=True)
    return out, stats
namespace["test_ssd_reload_gives_the_same_tokens_as_ram_and_cold"].__globals__["_run"] = traced
for dtype in (mx.bfloat16, mx.float16, mx.float32):
    m = model.__wrapped__()
    m.language_model.set_dtype(dtype)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            namespace["test_ssd_reload_gives_the_same_tokens_as_ram_and_cold"](m, Path(tmp))
            print("APC_RESULT", dtype, "pass", flush=True)
        except AssertionError:
            print("APC_RESULT", dtype, "fail", flush=True)
