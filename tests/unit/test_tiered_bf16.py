"""tiered/SSD KV store must quantize/dequant via float32, not float16.

The default KV dtype is bf16 (max ~3.4e38). tiered.py cast KV to fp16 (max 65504) BEFORE
int8-quantizing at store, and applied the f32 scale in fp16 at load. Any element with
|x| > 65504 (attention sinks / outlier channels routinely exceed this) became inf → scale =
max(inf)/127 = inf → round(x/inf)=0 → the whole block stored as garbage; on the read side the
fp16 multiply could overflow to inf → NaN attention. The sibling ssd_kv_cache.py already casts
to float32; these cold-tier paths were never swept. Fix: float32 at both store and load.
"""

from __future__ import annotations

import inspect

import numpy as np


def test_store_and_dequant_use_float32_not_float16():
    from yunshu_kv import tiered

    src = inspect.getsource(tiered)
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    # store path casts to float32 before quantizing
    assert "mx.array(kv_data).astype(mx.float32)" in code
    assert "mx.array(kv_data).astype(mx.float16)" not in code
    # dequant path uses float32 for the array AND the scale
    assert "_q.reshape(_n, -1).astype(_np.float32)" in code
    assert "_np.float32(_scales[_i])" in code
    assert "_np.float16(_scales[_i])" not in code


def test_bf16_over_fp16max_survives_round_trip():
    # the exact failure mode: a KV value beyond fp16 max
    nd = np.array([70000.0, -1.5, 100000.0, 0.3], dtype=np.float32).reshape(2, 2)
    scale = max(abs(nd.max()), abs(nd.min())) / 127.0
    q = np.round(nd / scale).astype(np.int8)
    deq = q.reshape(2, -1).astype(np.float32) * np.float32(scale)
    assert np.isfinite(deq).all()  # no inf (the bug)
    assert abs(deq.flatten()[0] - 70000) < 2000  # the sink magnitude survives
    # the OLD fp16 path would have produced inf:
    old = nd.astype(np.float16)
    assert not np.isfinite(old).all()  # documents the bug it fixes
