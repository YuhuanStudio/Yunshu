"""Check the bounded keyed draw on the hosted paravirtual GPU."""
import mlx.core as mx
import numpy as np

from yunshu_engine.keyed_sampling import KeyedSampler
from yunshu_engine.vlm_batch_runner import RowParams
from yunshu_engine.utils.hardware import is_paravirtual_metal

print("DEVICE", mx.device_info(mx.gpu), "VIRTUAL", is_paravirtual_metal(), flush=True)
row = np.full((1, 248320), -np.inf, dtype=np.float32)
row[0, :5] = 0
lp = mx.broadcast_to(mx.array(row), (600, 248320))
for seed in (0, 1, 2):
    sampler = KeyedSampler(RowParams(temperature=1, top_p=1, top_k=0, min_p=0), seed)
    tokens = np.array(sampler.sample_positions(lp, range(600)))
    assert tokens.max() < 5
    print("DRAW", seed, np.unique(tokens).tolist(), flush=True)
print("complete", flush=True)
