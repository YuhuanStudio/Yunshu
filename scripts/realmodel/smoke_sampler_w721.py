"""Wave 721/726 real-model smoke: confirm the default fast-path sampler with
temperature>0 + top_p produces valid, coherent output on a real model (both
streaming and non-streaming), after the temp-position fix.

Run isolated: PYTHONPATH=. uv run python scripts/realmodel/smoke_sampler_w721.py
"""
import asyncio

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


async def main():
    from yunshu_engine.batched_engine import BatchedEngine

    eng = BatchedEngine(MODEL)
    await eng.start()
    msgs = [{"role": "user", "content": "Name three primary colors."}]

    # Non-streaming, temp>0 + top_p (exercises _build_noncached_sampler_text).
    out = await eng.chat(msgs, max_tokens=40, temperature=0.8, top_p=0.9, seed=42)
    ns_text = out.text
    print("[non-stream temp=0.8 top_p=0.9]:", repr(ns_text[:120]))
    assert ns_text.strip(), "non-streaming produced empty output"

    # Streaming, same params (exercises make_sampler streaming path).
    chunks = []
    async for o in eng.stream_chat(msgs, max_tokens=40, temperature=0.8, top_p=0.9, seed=42):
        chunks.append(o.new_text or "")
    st_text = "".join(chunks)
    print("[stream     temp=0.8 top_p=0.9]:", repr(st_text[:120]))
    assert st_text.strip(), "streaming produced empty output"

    # Combined top_k + top_p + temp (the precise W721/W726 trigger).
    out2 = await eng.chat(msgs, max_tokens=40, temperature=1.3, top_p=0.8, top_k=40, seed=7)
    print("[non-stream temp=1.3 top_p=0.8 top_k=40]:", repr(out2.text[:120]))
    assert out2.text.strip(), "combined-filter sampling produced empty output"

    await eng.stop()
    print("\nW721/W726 real-model sampler smoke: PASS")


if __name__ == "__main__":
    asyncio.run(main())
