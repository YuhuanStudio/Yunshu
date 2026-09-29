"""Decode-proxy step time under the current environment (e.g. MLX_MAX_OPS_PER_BUFFER).

    MLX_MAX_OPS_PER_BUFFER=100 python scripts/research/hw/decode_proxy_step.py
"""

import os

import mlx.core as mx
from _common import Out
from _proxy import DecodeProxy, stats


def main():
    out = Out("decode_proxy_step")
    d = DecodeProxy()
    d.run(2)
    s = stats(d.run(8))
    out(kind="decode_proxy", max_ops=os.environ.get("MLX_MAX_OPS_PER_BUFFER", "default"),
        max_mb=os.environ.get("MLX_MAX_MB_PER_BUFFER", "default"),
        weight_GB=round(d.nbytes / 1e9, 2),
        GBps=round(d.nbytes / (s["ms_median"] / 1e3) / 1e9, 1), **s)
    _ = mx


if __name__ == "__main__":
    main()
