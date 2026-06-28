"""Wave 658: verify the LLM KV prefix cache is FAITHFUL on the production config
for this 30-core M3 Max — i.e. QUANTIZED models (the real lever: 4-bit weights
decode ~3x faster, see memory hardware_ceiling_m3max30).

Two contracts our hard-won 4-tier KV prefix cache was only ever verified on bf16:
  A. 4-bit WEIGHT model + standard KV.
  B. 4-bit weights + KV-cache quantization (YUNSHU_KV_QUANT_BITS=8).

CONTRACT — "faithful reuse" (the right bar for quantized models, NOT byte-lossless):
reusing a cached prefix must produce the SAME probability distribution as computing
it fresh, up to numerical noise. bf16 has enough precision margin to be byte-lossless;
4-bit's closely-spaced logits can flip an occasional EXACT TIE because any prefix
reuse changes the attention reduction order by sub-ULP (vLLM/SGLang document the same
— greedy is not guaranteed bit-identical under prefix caching, especially quantized).
So we pass when: warm == cold (byte-lossless), OR the first divergence is a genuine
near-tie in the COLD distribution (top-2 gap < TIE_GAP) — proving the reused KV is
numerically faithful and not corrupted (a corrupted KV would shift logits a LOT, not
tip a tie). Verified live (Wave 658): bf16 byte-lossless; 4-bit flips one exact tie
(gen-pos 35: cold top-2 both -0.798) into equally-coherent text.

Greedy (temp 0, seed 0). CTX-sharing prompts so the warm run reuses a long prefix.

Usage: PYTHONPATH=python uv run python scripts/verify_quant_kv_prefix.py \
           models/Qwen2.5-3B-Instruct-4bit
"""
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.abspath("python"))

CTX = "You are a helpful assistant.\n\n" + "".join(
    f"Fact {i}: city-{i:03d} has population {i * 1234 % 99999}.\n" for i in range(40))
P1 = CTX + "\n\nWrite a detailed multi-paragraph summary of the facts above."
P2 = CTX + "\n\nList the three largest cities by population, in detail."


TIE_GAP = 0.3  # cold top-2 logprob gap below this = a genuine near-tie (benign flip)


async def run(eng, prompt, mt=120):
    out = await eng.generate(prompt=prompt, max_tokens=mt, temperature=0.0, seed=0,
                             logprobs=True, top_logprobs=3)
    return out


def _faithful(cold, warm):
    """True if warm reuse is byte-lossless OR diverges only at a cold near-tie."""
    if cold.text == warm.text:
        return True, "byte-lossless"
    cl, wl = cold.logprobs or [], warm.logprobs or []
    for i in range(min(len(cl), len(wl))):
        if cl[i].get("token") != wl[i].get("token"):
            top = cl[i].get("top_logprobs") or []
            gap = (top[0]["logprob"] - top[1]["logprob"]) if len(top) >= 2 else 99.0
            if gap < TIE_GAP:
                return True, f"faithful: 1st flip at pos {i} is a near-tie (cold gap {gap:.3f})"
            return False, f"CORRUPT: 1st flip at pos {i} cold gap {gap:.3f} >= {TIE_GAP} (not a tie)"
    return True, "byte-lossless (token-aligned)"


async def fresh(model, kv_quant=None):
    # set the KV-quant env BEFORE constructing the engine (read in start())
    if kv_quant:
        os.environ["YUNSHU_KV_QUANT_BITS"] = str(kv_quant)
        os.environ["YUNSHU_KV_QUANT_START"] = "0"
    else:
        os.environ.pop("YUNSHU_KV_QUANT_BITS", None)
    from yunshu_engine.batched_engine import BatchedEngine
    eng = BatchedEngine(model)
    await eng.start()
    return eng


async def check(model, kv_quant=None):
    tag = f"KV-quant int{kv_quant}" if kv_quant else "standard KV"
    # cold: a fresh engine, P2 with NOTHING pre-cached
    eng = await fresh(model, kv_quant)
    c2 = await run(eng, P2)
    await eng.stop()

    # warm: P1 first (caches CTX), then P2 reuses the shared CTX prefix
    eng = await fresh(model, kv_quant)
    w1 = await run(eng, P1)
    # decode tps on the warm engine (4-bit weights → expect the ~3x lever)
    t0 = time.perf_counter()
    w2 = await run(eng, P2, mt=160)
    wall = time.perf_counter() - t0
    hits = eng._kv_prefix_cache._total_hits if eng._kv_prefix_cache else 0
    await eng.stop()

    dec = (w2.completion_tokens) / max(wall - w2.ttft_ms / 1000, 1e-6)
    faithful, why = _faithful(c2, w2)
    print(f"  [{tag}] faithful={faithful} ({why}) | warm_cached_tokens={w2.cached_tokens} "
          f"hits={hits} | decode={dec:.1f} tok/s")
    return faithful and w2.cached_tokens > 0


async def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen2.5-3B-Instruct-4bit"
    print(f"model: {model}")
    a = await check(model, kv_quant=None)   # A: 4-bit weights, standard KV
    b = await check(model, kv_quant=8)      # B: 4-bit weights + int8 KV-cache quant
    ok = a and b
    print(f"\n{'PASS' if ok else 'FAIL'}: quantized-model KV prefix cache "
          f"(weights={'OK' if a else 'FAIL'}, kv-quant={'OK' if b else 'FAIL'})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
