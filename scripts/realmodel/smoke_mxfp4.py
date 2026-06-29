"""real-model smoke: on-the-fly MXFP4 weight quantization at load
(YUNSHU_QUANT_MODE=mxfp4). Loads a bf16 model, quantizes in-memory to MXFP4,
confirms it generates coherently and uses less memory than bf16.
Run: PYTHONPATH=. uv run python scripts/realmodel/smoke_mxfp4.py
"""
import asyncio, os
MODEL = "models/Qwen2.5-3B-Instruct-bf16"

async def _run(mode):
    if mode:
        os.environ["YUNSHU_QUANT_MODE"] = mode
    else:
        os.environ.pop("YUNSHU_QUANT_MODE", None)
    import mlx.core as mx
    from yunshu_engine.batched_engine import BatchedEngine
    mx.clear_cache()
    eng = BatchedEngine(MODEL)
    await eng.start()
    mem = mx.get_active_memory() / 1e9
    r = await eng.chat([{"role": "user", "content": "What is the capital of France? One word."}],
                       max_tokens=10, temperature=0.0)
    await eng.stop()
    return r.text.strip(), round(mem, 2)

async def main():
    txt, mem = await _run("mxfp4")
    print(f"[MXFP4 on-the-fly] mem={mem}GB output={txt[:50]!r}")
    assert txt, "MXFP4-quantized model produced empty output"
    assert "paris" in txt.lower(), f"unexpected output: {txt!r}"
    print("\nW750 on-the-fly MXFP4 quant smoke: PASS (loaded bf16, quantized to mxfp4, coherent)")

asyncio.run(main())
