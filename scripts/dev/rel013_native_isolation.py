"""Isolate upstream APC arithmetic without Yunshu span/restore optimizations."""
import random
import runpy
from types import SimpleNamespace
import mlx.core as mx
from mlx_vlm.apc import APCManager
from yunshu_engine import cache_prefill
from yunshu_engine.utils import hardware
from yunshu_engine.vlm_batch_runner import VLMBatchRunner

print("DEVICE", mx.device_info(mx.gpu), flush=True)
fixture = runpy.run_path("tests/unit/test_apc_manager_e2e.py")
# Only bypass the test fixture's VM skip; preserve the real BF16 weights.
hardware.is_paravirtual_metal = lambda: False
cache_prefill.install = lambda: None
rnd = random.Random(5)
a = [rnd.randrange(3,800) for _ in range(260)]
_ = [rnd.randrange(3,800) for _ in range(260)]
a2 = a + [rnd.randrange(3,800) for _ in range(40)]
for dtype in (mx.bfloat16, mx.float16, mx.float32):
    model = fixture["model"].__wrapped__(SimpleNamespace(param=dtype))
    manager = APCManager(num_blocks=64, block_size=16, overrides={"memory_max_gb":1,"checkpoint_interval_tokens":0})
    warm = VLMBatchRunner(model, processor=fixture["_processor"](), apc_manager=manager, apc_semantic_hash=0)
    fixture["_run"](warm, a)
    got, stats = fixture["_run"](warm, a2)
    cold, _ = fixture["_run"](VLMBatchRunner(model, processor=fixture["_processor"]()), a2)
    print("NATIVE_APC", dtype, "cached", stats.cached_tokens, "warm", got, "cold", cold, "equal", got==cold, flush=True)
print("complete", flush=True)
