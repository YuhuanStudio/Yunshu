"""Is the Qwen4 MTP head right on a real pack?  Teacher-forced next-next-token accuracy.

Production acceptance on oQ4e-MTP was 0.4% (random).  This probe feeds natural text through the target,
takes the pre-mixer hyper-connection hidden h_t, and asks the retained head for token t+2 given
embed(token_{t+1}) and h_t, for each embedding shift s in {0,1,2} (embed token_{t+s}), and compares with the
target's own next-token accuracy.  A correct head scores far above chance for s=1.

Run through gpuq only:  flashnext80_mtp_align.py --model PACK --out F.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

TEXT = (
    "The lighthouse stood at the edge of the cliff, its white tower weathered by a hundred winters. "
    "Every evening the keeper climbed the spiral stairs, trimmed the wick, and watched the ships pass "
    "far below. In the morning he wrote the weather in a worn ledger, noting the wind, the tide, and "
    "the color of the sky. When storms came he stayed awake all night, listening to the waves, "
    "and he never once let the light go out. "
) * 4


async def run(args):
    import mlx.core as mx

    from yunshu_engine import qwen4_mtp, settings
    from yunshu_engine.types import EngineConfig
    from yunshu_engine.vlm_engine import VLMEngine

    settings.set_override("YUNSHU_VLM_APC_MEMORY_GB", 1.0)
    settings.set_override("YUNSHU_VLM_DRAFT", "mtp")
    captured = {}
    original = qwen4_mtp.load_native_head

    def capture(config, weights):
        captured["drafter"] = original(config, weights)
        return captured["drafter"]

    qwen4_mtp.load_native_head = capture
    engine = VLMEngine(
        str(args.model), EngineConfig(prefill_step_size=512, completion_batch_size=1)
    )
    await engine.start()
    drafter = captured.get("drafter")
    out = args.out.open("w")

    def emit(row):
        out.write(json.dumps(row) + "\n")
        out.flush()
        print(json.dumps(row), flush=True)

    if drafter is None:
        emit({"complete": False, "reason": "no drafter was loaded"})
        return 1
    model = engine._model
    chunks = []
    async for o in engine.generate_stream(
        messages=[
            {
                "role": "user",
                "content": "What is the capital of France? Answer in one word.",
            }
        ],
        max_tokens=24,
        temperature=0,
        enable_thinking=False,
    ):
        chunks.append(o.new_text)
    emit({"kind": "generate", "text": "".join(chunks)})
    emit(
        {
            "kind": "mode",
            "training": bool(model.training),
            "lm_training": bool(model.language_model.training),
        }
    )
    lm = model.language_model
    tok = (
        engine._tokenizer
        if hasattr(engine, "_tokenizer")
        else engine._processor.tokenizer
    )
    ids = tok.encode(TEXT)[: args.tokens]
    model.eval()
    x = mx.array([ids])
    cache = lm.make_cache()
    res = lm(x, cache=cache, return_hidden=True)
    hidden = res.hidden_states[0]
    logits = res.logits
    mx.eval(hidden, logits)
    n = len(ids)
    target_next = mx.argmax(logits[0, :-1], axis=-1).tolist()
    tgt_acc = sum(int(target_next[i] == ids[i + 1]) for i in range(n - 1)) / (n - 1)
    plain = lm(x, cache=lm.make_cache())
    pl = plain.logits
    mx.eval(pl)
    plain_next = mx.argmax(pl[0, :-1], axis=-1).tolist()
    emit(
        {
            "kind": "target_plain",
            "acc": round(
                sum(int(plain_next[i] == ids[i + 1]) for i in range(n - 1)) / (n - 1), 4
            ),
            "logits_shape": list(pl.shape),
            "finite": bool(mx.all(mx.isfinite(pl)).item()),
            "same_as_hidden_call": bool(mx.array_equal(pl, logits).item()),
            "first_pred": plain_next[:8],
            "first_gold": ids[1:9],
            "decoded": tok.decode(plain_next[:24]),
        }
    )
    emit(
        {
            "kind": "target",
            "tokens": n,
            "next_token_acc": round(tgt_acc, 4),
            "hidden_shape": list(hidden.shape),
        }
    )
    drafter.reset(model)
    for shift in (0, 1, 2):
        # head input at position t: embed(token_{t+shift}), h_t ; label: token_{t+shift+1}
        m = n - 1 - shift
        if m <= 0:
            continue
        drafter.reset(model)
        toks = mx.array([ids[shift : shift + m]])
        lh, _ = drafter._forward_tokens(toks, hidden[:, :m, ...], mx.int32)
        pred = mx.argmax(drafter._lm_head_fn(lh), axis=-1)[0].tolist()
        gold = ids[shift + 1 : shift + 1 + m]
        acc = sum(int(a == b) for a, b in zip(pred, gold, strict=True)) / m
        emit(
            {
                "kind": "head",
                "embed_shift": shift,
                "label_offset": shift + 1,
                "acc": round(acc, 4),
                "n": m,
            }
        )
    emit({"complete": True})
    out.close()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    return asyncio.run(run(ap.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
