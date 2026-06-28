"""Wave 746 — quantify the W740 length-gated KV-quant decode speedup at long
context (Qwen2.5-0.5B-4bit). Compares decode tok/s with the auto-gate ON vs OFF
for a long prompt that exceeds the threshold.

Run: PYTHONPATH=. uv run python scripts/bench/bench_kv_quant_longctx.py
"""
import asyncio
import os
import time

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
# Long prompt to exceed the 8192-token auto-quant threshold so KV bytes dominate.
LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 900  # ~9k tokens
GEN_TOKENS = 128


async def _run(auto_quant: str) -> dict:
    os.environ["YUNSHU_KV_QUANT_AUTO"] = auto_quant
    # W747: gate is KV-BYTES based; force ON by min_bytes=0 (quantize regardless of size).
    os.environ["YUNSHU_KV_QUANT_AUTO_MIN_BYTES"] = "0" if auto_quant == "1" else str(10**18)
    from yunshu_engine.batched_engine import BatchedEngine
    eng = BatchedEngine(MODEL)
    await eng.start()
    ntok = len(eng._tokenizer.encode(LONG_PROMPT))
    eff = eng._effective_kv_quant_bits(ntok + GEN_TOKENS)
    # warmup
    await eng.generate(LONG_PROMPT, max_tokens=8, temperature=0.0)
    t0 = time.perf_counter()
    r = await eng.generate(LONG_PROMPT, max_tokens=GEN_TOKENS, temperature=0.0)
    dt = time.perf_counter() - t0
    await eng.stop()
    tps = r.completion_tokens / dt if dt > 0 else 0.0
    return {"prompt_tokens": ntok, "kv_bits": eff, "gen": r.completion_tokens,
            "secs": round(dt, 2), "tok_s": round(tps, 1)}


async def main():
    print(f"model={MODEL} gen_tokens={GEN_TOKENS}")
    on = await _run("1")
    print(f"[auto-quant ON ] {on}")
    off = await _run("0")
    print(f"[auto-quant OFF] {off}")
    if off["tok_s"] > 0:
        ratio = on["tok_s"] / off["tok_s"]
        print(f"\ndecode speedup (ON/OFF) = {ratio:.2f}x  "
              f"(ON kv_bits={on['kv_bits']}, OFF kv_bits={off['kv_bits']})")


if __name__ == "__main__":
    asyncio.run(main())
