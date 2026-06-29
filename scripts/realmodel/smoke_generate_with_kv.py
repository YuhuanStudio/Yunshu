import asyncio
MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
async def main():
    import mlx.core as mx
    from yunshu_engine.batched_engine import BatchedEngine
    from yunshu_engine.external_prefill import ExternalPrefiller
    eng = BatchedEngine(MODEL)
    await eng.start()
    tok = eng._tokenizer
    prompt = "The quick brown fox"
    ids = tok.encode(prompt)
    # Prefill via ExternalPrefiller, then decode reusing the KV.
    pf = ExternalPrefiller(eng._model, tok)
    res = pf.prefill(ids)
    assert res.kv_cache is not None and res.last_logits is not None, "prefill must produce cache+logits"
    out = await eng.generate_with_kv(res.kv_cache, ids, res.last_logits,
                                     max_tokens=20, temperature=0.0)
    # Reference: normal greedy generate from the same raw prompt (no chat template).
    ref = await eng.generate(prompt, max_tokens=20, temperature=0.0)
    print("[generate_with_kv] :", repr(out.text[:80]), "ct=", out.completion_tokens)
    print("[reference generate]:", repr(ref.text[:80]), "ct=", ref.completion_tokens)
    # Greedy + correct KV reuse → identical text.
    assert out.text == ref.text, f"KV-reuse decode != reference:\n  {out.text!r}\n  {ref.text!r}"
    await eng.stop()
    print("\nW741 generate_with_kv (KV reuse == reference greedy) smoke: PASS")
asyncio.run(main())
