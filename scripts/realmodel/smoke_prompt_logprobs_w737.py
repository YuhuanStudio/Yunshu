"""Wave 737 real-model smoke: prompt_logprobs (eval/perplexity) on Qwen2.5-0.5B.

Verifies coherent text scores a higher (less negative) avg logprob than
gibberish, element 0 is None, and top_logprobs is populated.
Run: PYTHONPATH=. uv run python scripts/realmodel/smoke_prompt_logprobs_w737.py
"""
import asyncio
MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
async def main():
    from yunshu_engine.batched_engine import BatchedEngine
    eng = BatchedEngine(MODEL)
    await eng.start()
    # A coherent sentence should have higher (less negative) avg logprob than gibberish.
    r1 = await eng.generate("The capital of France is Paris.", max_tokens=1,
                            temperature=0.0, prompt_logprobs=3)
    r2 = await eng.generate("Colorless green ideas sleep furiously zxqw.", max_tokens=1,
                            temperature=0.0, prompt_logprobs=0)
    pl1 = r1.prompt_logprobs
    assert pl1 is not None and pl1[0] is None, "element 0 must be None"
    vals1 = [e["logprob"] for e in pl1 if e]
    assert all(v <= 0.0 for v in vals1), "logprobs must be <= 0"
    assert pl1[1] and "top_logprobs" in pl1[1] and len(pl1[1]["top_logprobs"]) == 3
    # realized token must be in its own top_logprobs or have a valid logprob
    print(f"[coherent]  n={len(pl1)} avg_logprob={sum(vals1)/len(vals1):.3f} top0={pl1[1]['top_logprobs'][0]}")
    pl2 = r2.prompt_logprobs
    vals2 = [e["logprob"] for e in pl2 if e]
    print(f"[gibberish] n={len(pl2)} avg_logprob={sum(vals2)/len(vals2):.3f}")
    # Coherent text should be more predictable (higher avg logprob) than gibberish.
    assert sum(vals1)/len(vals1) > sum(vals2)/len(vals2), "coherent should be more predictable"
    await eng.stop()
    print("\nW737 prompt_logprobs real-model smoke: PASS")
asyncio.run(main())
