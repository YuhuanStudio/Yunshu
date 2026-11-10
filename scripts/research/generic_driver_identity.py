"""In-process: RoundDriver generic layer on a real model, row A alone vs beside row B (tokens, first divergence)."""
import sys

import mlx.core as mx
from mlx_vlm import load

from yunshu_engine.round_driver import driver as drv

model, proc = load(sys.argv[1])
tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
lm = model.language_model
text_a = "The history of foxes in the forest. " * 800
text_b = "A completely different story about boats and rivers. " * 600
ids = {k: tok.encode(t) for k, t in (("a", text_a), ("b", text_b))}
print({k: len(v) for k, v in ids.items()})


def req(k, n=64):
    ii = ids[k]
    pk = model.get_input_embeddings(mx.array(ii)[None])
    pk = pk.to_dict() if hasattr(pk, "to_dict") else dict(pk)
    return drv.Request(ii, n, handle=k, prompt_kwargs=pk, draft=False, logprobs=True)


def run(keys, join=None):
    d = drv.RoundDriver(lm, generic_layer=True, stop_tokens=set())
    out = {}
    pend = {k: j for k, j in zip(keys, join or [0] * len(keys), strict=True)}
    steps = 0
    while d.busy() or pend:
        for k in list(pend):
            if pend[k] <= steps:
                d.add(req(k)); del pend[k]
        for e in d.step():
            out.setdefault(e.handle, []).append((e.token, getattr(e, "logprob", None)))
        steps += 1
    return out


solo = {k: run([k])[k] for k in ids}
for name, got in (("burst", run(["a", "b"])), ("stagger", run(["a", "b"], [0, 6]))):
    for k in ids:
        s, c = solo[k], got[k]
        div = next((i for i, (x, y) in enumerate(zip(s, c, strict=False)) if x != y), None)
        print(name, k, "identical" if div is None and len(s) == len(c) else f"DIVERGES at {div}")
