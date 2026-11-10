"""Probe: first op whose row-A output differs between batch-1 decode and batch-2 (left padded) decode."""

import sys

import mlx.core as mx
import mlx.nn as nn

model_dir = sys.argv[1]
from mlx_vlm import load
from mlx_vlm.generate import ar
from mlx_vlm.models import cache as vc

model, proc = load(model_dir)
lm = model.language_model
if len(sys.argv) > 2 and sys.argv[2] == 'inv':
    from yunshu_engine.round_driver import generic_invariant
    print(generic_invariant.install(lm))
tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
ida = tok.encode(
    "The quick brown fox jumps over the lazy dog. Tell me about foxes and their habits in the wild forest."
)
mult = int(sys.argv[3]) if len(sys.argv) > 3 else 1
ida = ida * mult
idb = tok.encode("Hello there, how are you today my friend?")
print("lens", len(ida), len(idb))

REC = []
SKIP = (nn.Module,)


def patch():
    classes = {}
    for name, m in lm.named_modules():
        m._pname = name
        classes[type(m)] = 1
    for cls in classes:
        if "__call__" not in cls.__dict__ or cls.__name__ in (
            "Gemma4TextModel",
            "Model",
            "LanguageModel",
        ):
            continue
        orig = cls.__dict__["__call__"]

        def mk(orig):
            def call(self, *a, **k):
                out = orig(self, *a, **k)
                if (
                    REC is not None
                    and getattr(self, "_pname", None) is not None
                    and isinstance(out, mx.array)
                ):
                    REC.append((self._pname, type(self).__name__, out))
                return out

            return call

        cls.__call__ = mk(orig)


patch()
import mlx_vlm.models.gemma4.language as g4l

_orig_sdpa = g4l.scaled_dot_product_attention
_SD = [0]


def _sdpa(q, k, v, cache=None, scale=None, mask=None, **kw):
    out = _orig_sdpa(q, k, v, cache=cache, scale=scale, mask=mask, **kw)
    REC.append((f'sdpa k={tuple(k.shape)} mask={None if mask is None else (getattr(mask, "shape", mask), str(getattr(mask, "dtype", "")))} {type(cache).__name__}', "SDPA", out.transpose(0, 2, 1, 3)))
    _SD[0] += 1
    return out


g4l.scaled_dot_product_attention = _sdpa


def prefill(ids):
    c = vc.make_prompt_cache(lm)
    out = lm(mx.array(ids)[None], cache=c)
    lg = out.logits if hasattr(out, "logits") else out
    mx.eval(lg, *[x for cc in c for x in cc.state if isinstance(x, mx.array)])
    return c, int(mx.argmax(lg[0, -1]).item())


REC_ON = [False]
ca, ta = prefill(ida)
cb, tb = prefill(idb)


# fresh copies for the two runs
def clone(c):
    ca2 = vc.make_prompt_cache(lm)
    for d, s in zip(ca2, c):
        d.state = s.state
        d.meta_state = s.meta_state
    return ca2


REC.clear(); _SD[0] = 0
solo_c = clone(ca)
out1 = lm(mx.array([[ta]]), cache=solo_c)
rec1 = list(REC)
REC.clear()
bc = None
bc = ar._extend_cache(bc, clone(ca))
bc = ar._extend_cache(bc, clone(cb))
out2 = lm(mx.array([[ta], [tb]]), cache=bc)
rec2 = list(REC)
REC.clear()
mx.eval([r[2] for r in rec1], [r[2] for r in rec2])
print("calls", len(rec1), len(rec2))
n = 0
first = None
seen = set()
for (n1, t1, o1), (n2, t2, o2) in zip(rec1, rec2):
    if t1 == 'SDPA':
        print('sdpa solo', n1, '| batch', n2) if not mx.array_equal(o1[0], o2[0]).item() else None
        n1 = n2 = 'sdpa'
    assert n1 == n2, (n1, n2)
    a = o1[0]
    b = o2[0]
    if a.shape != b.shape:
        continue
    if not mx.array_equal(a, b).item():
        n += 1
        if first is None:
            first = (n1, t1)
        key = t1
        if key not in seen:
            seen.add(key)
            print(
                "differs:",
                n1,
                t1,
                "maxabs",
                mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).max().item(),
            )
print("FIRST differing op:", first, "total differing calls", n)
l1 = (out1.logits if hasattr(out1, "logits") else out1)[0, -1]
l2 = (out2.logits if hasattr(out2, "logits") else out2)[0, -1]
print("logits equal", mx.array_equal(l1, l2).item())
