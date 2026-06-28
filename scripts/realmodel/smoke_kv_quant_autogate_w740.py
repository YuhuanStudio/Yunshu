import asyncio, os
os.environ["YUNSHU_KV_QUANT_AUTO_THRESHOLD"] = "8"  # force quant on a short prompt
MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
async def main():
    from yunshu_engine.batched_engine import BatchedEngine
    eng = BatchedEngine(MODEL)
    await eng.start()
    eff = eng._effective_kv_quant_bits(100)
    print("effective bits @100 tokens, threshold 8:", eff)
    assert eff == 8, "length-gated default should kick in"
    assert eng._effective_kv_quant_bits(4) is None, "below threshold → None"
    # Generate with quant active — output must stay coherent.
    r = await eng.chat([{"role":"user","content":"Count from 1 to 5."}], max_tokens=40, temperature=0.0)
    print("[KV-quant active] output:", repr(r.text[:80]))
    assert r.text.strip() and any(d in r.text for d in "12345"), f"incoherent: {r.text!r}"
    # opt-out
    os.environ["YUNSHU_KV_QUANT_AUTO"] = "0"
    assert eng._effective_kv_quant_bits(100000) is None, "opt-out must disable"
    await eng.stop()
    print("\nW740 length-gated KV-quant smoke: PASS")
asyncio.run(main())
