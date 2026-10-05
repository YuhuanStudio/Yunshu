"""Stock long-context reference: plain mlx-vlm load + stream_generate (no Yunshu code, no patches),
greedy, on the same prompts the yv `long` suite uses.

  long_stock.py --model M --ctx 32768 [--ctx 131072] --parts needle,decode --out F.jsonl [--apc]

Records (one JSON per line): needle items (`part=needle`, same fields as tfbench) and 2048-token
decode cells (`part=decode`, cold request only: ttft, decode tok/s, peak memory, 4-gram repeat
share, text). The process ends with a `part_done` record; a missing one means the job failed.
Every request is checked fail-closed: a decode cell must end finish=length with N tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import tfbench as t  # noqa: E402  (prompt files, needle items, repetition metric; no Yunshu imports)


def messages_for(text: str) -> list:
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def run_one(gen, model, proc, prompt, max_tokens, **kw):
    """One greedy request through `gen` (mlx_vlm.stream_generate). Returns a record dict."""
    t0 = time.perf_counter()
    first = None
    chunks, last = [], None
    for item in gen(model, proc, prompt, max_tokens=max_tokens, **kw):
        last = item
        if item.text and first is None:
            first = time.perf_counter() - t0
        chunks.append(item.text)
    if last is None:
        raise RuntimeError("no output")
    return dict(
        text="".join(chunks),
        ttft_s=round(first if first is not None else time.perf_counter() - t0, 3),
        total_s=round(time.perf_counter() - t0, 3),
        ct=int(last.generation_tokens),
        pt=int(last.prompt_tokens),
        finish=last.finish_reason,
        dec_tps=round(float(last.generation_tps), 2),
    )


def score_needle(items, answers):
    return [
        {
            "item": i,
            "name": nm,
            "expect": code,
            "answer": ans[:80],
            "correct": code in ans,
        }
        for i, ((nm, code), ans) in enumerate(zip(items, answers, strict=True))
    ]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=t.M)
    ap.add_argument("--ctx", type=int, action="append", required=True)
    ap.add_argument("--parts", default="needle,decode")
    ap.add_argument("--decode-tokens", type=int, default=2048)
    ap.add_argument("--kinds", default="prose,code")
    ap.add_argument(
        "--apc", action="store_true", help="mlx-vlm's own prefix cache for needle"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-items", type=int, default=0)
    a = ap.parse_args(argv)
    parts = a.parts.split(",")

    import mlx.core as mx
    import mlx_vlm
    from mlx_lm.sample_utils import make_sampler
    from mlx_vlm import load, stream_generate

    model, proc = load(a.model)
    sampler = make_sampler(temp=0.0)
    out = open(a.out, "a")  # noqa: SIM115

    def emit(**kw):
        out.write(json.dumps(kw) + "\n")
        out.flush()

    emit(part="session", engine="stock", mlx_vlm=getattr(mlx_vlm, "__version__", "?"))

    def chat(text):
        return proc.apply_chat_template(
            messages_for(text),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    for ctx in a.ctx:
        if "needle" in parts:
            full = t.load_prompt(f"prose-{ctx}")
            base = full[: full.rfind("\n\n---\n")]
            items = t.needle_items(ctx)
            hay = t.needle_haystack(base, ctx, items)
            kw = {}
            if a.apc:
                from mlx_vlm.apc import APCManager

                kw["apc_manager"] = APCManager(
                    num_blocks=ctx // 256 + 64, block_size=256
                )
            for i, (nm, code) in enumerate(items[: a.max_items or None]):
                mx.reset_peak_memory()
                r = run_one(
                    stream_generate,
                    model,
                    proc,
                    chat(hay + t.needle_question(nm)),
                    16,
                    sampler=sampler,
                    **kw,
                )
                emit(
                    part="needle",
                    ctx=ctx,
                    item=i,
                    name=nm,
                    expect=code,
                    answer=r["text"][:80],
                    correct=code in r["text"],
                    ttft_s=r["ttft_s"],
                    pt=r["pt"],
                )
        if "decode" in parts:
            for kind in a.kinds.split(","):
                text = t.load_prompt(f"{kind}-{ctx}") + (
                    t.LONG_ASK if a.long_ask else ""
                )
                mx.reset_peak_memory()
                r = run_one(
                    stream_generate,
                    model,
                    proc,
                    chat(text),
                    a.decode_tokens,
                    sampler=sampler,
                )
                t.check_decode_len(r, a.decode_tokens, f"stock decode {kind}-{ctx}")
                r["peak_gib"] = round(mx.get_peak_memory() / 2**30, 2)
                r["rep4"] = t.ngram_repeat(r["text"])
                emit(part="decode", ctx=ctx, kind=kind, phase="cold", **r)
    emit(part="part_done", engine="stock", complete=True)
    out.close()


if __name__ == "__main__":
    main()
