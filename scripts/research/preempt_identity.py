"""Does evaluating every few decoder layers change the numbers? (kernels/preempt.py)

Runs the same chunked prefill twice on one checkpoint: plain, and inside ``preempt.scope`` that
evaluates every ``--every`` layers (never cancelling). The logits of every chunk and every cache
state must be bit-equal. Writes one JSON record with ``complete: true`` and exits nonzero on any
difference or when the preempt hook did not engage (no evaluations counted).

    preempt_identity.py --model PATH --tokens 4096 --chunk 2048 --out result.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def digest(arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(bytes(memoryview(a)))
    return h.hexdigest()[:16]


def verdict(plain: list[str], scoped: list[str], evals: int) -> dict:
    """Pass only when every chunk digest matches and the scope really evaluated."""
    problems = []
    if not plain or len(plain) != len(scoped):
        problems.append("missing or unequal chunk results")
    elif plain != scoped:
        problems.append(
            "chunks differ: "
            + ",".join(str(i) for i, (a, b) in enumerate(zip(plain, scoped)) if a != b)
        )
    if evals <= 0:
        problems.append("the preempt scope evaluated nothing (hook not engaged)")
    return {"pass": not problems, "problems": problems}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--src", default=None, help="python/ dir of the tree under test")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.src:
        sys.path.insert(0, a.src)

    import mlx.core as mx
    from mlx_vlm.utils import load

    from yunshu_engine.kernels import preempt

    if not preempt.install():
        raise SystemExit("preempt.install() failed: no Qwen3.5 decoder layer")
    model, _ = load(a.model)
    lm = model.language_model
    mx.random.seed(0)
    ids = mx.random.randint(1000, 20000, (1, a.tokens))

    def run(scoped: bool):
        cache = lm.make_cache()
        out = []
        import threading

        ctx = preempt.scope(threading.Event(), a.every) if scoped else None
        state = ctx.__enter__() if ctx is not None else None
        try:
            for start in range(0, a.tokens, a.chunk):
                res = lm(ids[:, start : start + a.chunk], cache=cache)
                logits = res.logits if hasattr(res, "logits") else res
                states = [c.state for c in cache]
                flat = [logits] + [
                    x
                    for s in states
                    for x in (s if isinstance(s, (list, tuple)) else [s])
                    if x is not None
                ]
                mx.eval(flat)
                out.append(digest([mx.contiguous(x).astype(mx.float32) for x in flat]))
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)
        return out, (state.evals if state is not None else 0)

    plain, _ = run(False)
    scoped, evals = run(True)
    result = verdict(plain, scoped, evals)
    result.update(
        model=a.model,
        tokens=a.tokens,
        chunk=a.chunk,
        every=a.every,
        evals=evals,
        plain=plain,
        scoped=scoped,
        complete=True,
        sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12],
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("pass", "problems", "evals")}))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
