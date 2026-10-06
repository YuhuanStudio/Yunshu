"""Diagnostic: how Yunshu's engine.embed() path differs from the official recipe (run via gpuq)."""

import sys

sys.path.insert(0, "python")
sys.path.insert(0, "scripts/research")
from embed_reference import cosine, reference_embed

M = sys.argv[1]
T = [
    "The central bank raised interest rates and bond yields climbed.",
    "a football match report",
    "a cooking recipe",
    "financial news about interest rates and markets",
]
ref = reference_embed(M, T)
noeos = reference_embed(M, T, add_eos=False)
mean = reference_embed(M, T, pooling="mean")
for i, t in enumerate(T):
    print(
        i,
        "ref-vs-noeos",
        round(cosine(ref[i], noeos[i]), 4),
        "ref-vs-mean",
        round(cosine(ref[i], mean[i]), 4),
    )
print("ref sims input vs labels", [round(cosine(ref[0], ref[j]), 3) for j in (1, 2, 3)])
print("mean sims", [round(cosine(mean[0], mean[j]), 3) for j in (1, 2, 3)])
# what the engine would do
import asyncio
from yunshu_engine.batched_engine import BatchedEngine


async def go():
    e = BatchedEngine(model_name=M)
    await e.start()
    print(
        "pooling",
        e._resolve_embedding_pooling(),
        "backbone",
        type(e._get_backbone()).__name__,
    )
    got = e.embed(T)
    for i in range(len(T)):
        print(
            i,
            "engine-vs-ref",
            round(cosine(got[i], ref[i]), 4),
            "engine-vs-mean",
            round(cosine(got[i], mean[i]), 4),
        )
    print("engine sims", [round(cosine(got[0], got[j]), 3) for j in (1, 2, 3)])
    print("engine ids", e._tokenizer.encode(T[1])[-3:])


asyncio.run(go())
