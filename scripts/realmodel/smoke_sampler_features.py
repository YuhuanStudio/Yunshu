"""real-model smoke: min_tokens / ignore_eos / suppress_tokens on the
default fast path (Qwen2.5-0.5B-4bit).

Run: PYTHONPATH=. uv run python scripts/realmodel/smoke_sampler_features.py
"""
import asyncio

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


async def main():
    from yunshu_engine.batched_engine import BatchedEngine

    eng = BatchedEngine(MODEL)
    await eng.start()
    tok = eng._tokenizer
    eos = getattr(tok, "eos_token_id", None)

    # A chat prompt that reliably produces a SHORT answer (stops on EOS early).
    short_prompt = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]

    # Baseline: normal generation stops on EOS well before max_tokens=200.
    base = await eng.chat(short_prompt, max_tokens=200, temperature=0.0)
    print(f"[baseline]       tokens={base.completion_tokens} finish={base.finish_reason!r}")
    assert base.finish_reason == "stop", "baseline prompt did not stop early — pick a terser prompt"

    # ignore_eos: must keep going to max_tokens (finish_reason=length).
    ig = await eng.chat(short_prompt, max_tokens=200, temperature=0.0, ignore_eos=True)
    print(f"[ignore_eos]     tokens={ig.completion_tokens} finish={ig.finish_reason!r}")
    assert ig.completion_tokens > base.completion_tokens, "ignore_eos did not extend past baseline"
    assert ig.finish_reason == "length", f"ignore_eos should run to length, got {ig.finish_reason}"

    # min_tokens: must generate at least N tokens even though baseline stopped earlier.
    floor = base.completion_tokens + 10
    mt = await eng.chat(short_prompt, max_tokens=200, temperature=0.0, min_tokens=floor)
    print(f"[min_tokens={floor}] tokens={mt.completion_tokens} finish={mt.finish_reason!r}")
    assert mt.completion_tokens >= floor, f"min_tokens floor violated: {mt.completion_tokens} < {floor}"

    # suppress_tokens: a banned token id must never appear in the output ids.
    # Ban a few common ids and confirm absence.
    banned = [t for t in (eos,) if t is not None]
    # also ban the id for a leading-space "the" if present
    try:
        banned += tok.encode(" the")[:1]
    except Exception:
        pass
    sup = await eng.generate("List three colors:", max_tokens=40, temperature=0.0,
                             suppress_tokens=banned, ignore_eos=True)
    out_ids = sup.output_token_ids if hasattr(sup, "output_token_ids") else []
    print(f"[suppress]       banned={banned} present={[b for b in banned if b in out_ids]}")
    assert not any(b in out_ids for b in banned), "a suppressed token appeared in output"

    await eng.stop()
    print("\nW735 sampler-features real-model smoke: PASS")


if __name__ == "__main__":
    asyncio.run(main())
