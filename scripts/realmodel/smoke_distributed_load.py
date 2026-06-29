"""real-model smoke: distributed model load via mlx-lm native
sharded_load (TENSOR_PARALLEL). On one Mac the group is world_size=1 → trivial
size-1 shard = full model, so this validates the integration seam (the engine
loading a SHARDED model and serving from it). A real 2-Mac cluster shards across
both nodes via the same path.

Run: YUNSHU_TENSOR_PARALLEL=1 PYTHONPATH=. uv run python scripts/realmodel/smoke_distributed_load.py
"""
import asyncio
import os

os.environ.setdefault("YUNSHU_TENSOR_PARALLEL", "1")
MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


async def main():
    from yunshu_engine.batched_engine import BatchedEngine
    eng = BatchedEngine(MODEL)
    await eng.start()
    r = await eng.chat(
        [{"role": "user", "content": "What is the capital of France? One word."}],
        max_tokens=10, temperature=0.0,
    )
    print(f"[TP world_size=1] output={r.text[:60]!r} finish={r.finish_reason!r}")
    assert r.text.strip(), "distributed-loaded model produced empty output"
    assert "paris" in r.text.lower(), f"unexpected output: {r.text!r}"
    await eng.stop()
    print("\nW739 distributed-load (TP, world_size=1) smoke: PASS")


if __name__ == "__main__":
    asyncio.run(main())
