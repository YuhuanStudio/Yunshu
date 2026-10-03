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
