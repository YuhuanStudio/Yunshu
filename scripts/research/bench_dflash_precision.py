"""Draft-only precision A/B on the served invariant DFlash chain, via gpuq.

The target and its shared embedding/readout remain untouched. Quantization
happens before bind; every arm reloads its private drafter. Output IDs, full
decode time and commits/round decide the tradeoff, not component latency alone.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from spec_bench_snapshot import freeze, refuse_contended, was_contended


def chat_ids(tok, text):
    """Prompt token ids as a plain list (newer transformers return a BatchEncoding)."""
    out = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    out = out["input_ids"] if hasattr(out, "keys") else out
    return list(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("drafter")
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 4])
    ap.add_argument("--fusion", type=int, choices=[0, 1], nargs="+", default=[0, 1])
    ap.add_argument("--contexts", type=int, nargs="+", default=[1024, 8192, 32768])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if refuse_contended(a.output):
        return 0
    a.output.parent.mkdir(parents=True, exist_ok=True)
    source, fingerprint = freeze(a.output)
    sys.path.insert(0, str(source))
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.generate.ar import BatchGenerator
    from mlx_vlm.speculative.drafters import load_drafter

    from yunshu_engine.dflash_context import install as context_install
    from yunshu_engine.dflash_draft import (
        install_quantized_context,
        install_selector_readout,
    )
    from yunshu_engine.dflash_tree import quantize_drafter
    from yunshu_engine.kernels import gdn_prefill, lane_linear, omlx, ragged_kv
    from yunshu_engine.kernels.batch_invariant import install, set_active
    from yunshu_engine.mrope import clear_rope_state
    from yunshu_engine.spec_schedule import install_chain_budget

    omlx.apply(row_exact=False)
    model, processor = load(a.model)
    lm, tok = model.language_model, processor.tokenizer
    lane_linear.convert(lm)
    lane_linear.set_stock_rows(lane_linear.PIECE)
    install(lm, model=model, packed=False)
    gdn_prefill.install()
    set_active(True)
    ragged_kv.install()
    ragged_kv.set_dense_lane(True)
    context_install(lm)
    install_chain_budget()
    target_head = lm.lm_head
    target_weight = target_head.weight
    arms = {}
    for bits in a.bits:
        for fused in a.fusion:
            for selector in [False, True] if bits == 8 and fused else [False]:
                drafter, kind = load_drafter(a.drafter, kind="dflash")
                converted = quantize_drafter(drafter, bits)
                if fused and not install_quantized_context(drafter):
                    raise RuntimeError("quantized context fusion did not engage")
                if selector and not install_selector_readout(drafter):
                    raise RuntimeError("DFlash2 selector readout did not engage")
                assert lm.lm_head is target_head and target_head.weight is target_weight
                arms[bits, bool(fused), selector] = drafter
                print(
                    f"Speculative decoding: {kind}; private draft q{bits}, {converted} layers, context_fused={bool(fused)}, selector={selector}",
                    flush=True,
                )

    filler = tok.encode(
        "".join(f"Sensor {i}: pressure {i * 37 % 1000}.\n" for i in range(40000)),
        add_special_tokens=False,
    )
    tasks = {
        "code": "Write a Python LRU cache class with get, put, delete and resize. Output code only.",
        "prose": "Explain in detail how a refrigerator works, including compressor and evaporator.",
        "zh": "請用繁體中文詳細解釋冰箱的工作原理，包括壓縮機、冷凝器與蒸發器。",
    }

    def run(ids, draft, tokens, seed=None):
        sample_kw = {}
        if seed is not None:
            from yunshu_engine.keyed_sampling import KeyedSampler
            from yunshu_engine.vlm_batch_runner import RowParams

            sample_kw["sampler"] = KeyedSampler(
                RowParams(temperature=0.7, top_p=0.9, top_k=40, min_p=0.0), seed
            )
        gen = BatchGenerator(
            lm,
            processor,
            max_tokens=tokens,
            draft_model=draft,
            draft_kind="dflash" if draft is not None else None,
            draft_block_size=int(draft.config.block_size)
            if draft is not None
            else None,
            greedy_sampling=seed is None,
            compute_logprobs=False,
            **sample_kw,
        )
        clear_rope_state(model)
        kwargs = model.get_input_embeddings(
            mx.array(ids)[None], None, mask=None
        ).to_dict()
        start = time.perf_counter()
        uid = gen.insert([ids], max_tokens=tokens, prompt_kwargs=[kwargs])[0]
        emitted, first, stamps = [], None, []
        try:
            done = False
            while not done:
                _, responses = gen.next()
                for response in responses:
                    if response.uid != uid:
                        continue
                    now = time.perf_counter()
                    first = now if first is None else first
                    stamps.append(now)
                    emitted.append(int(response.token))
                    done |= response.finish_reason is not None
        finally:
            gen.close()
        mx.synchronize()
        end = time.perf_counter()
        rounds = len(draft.accept_lens) if draft is not None else len(emitted) - 1
        return {
            "tokens": len(emitted),
            "ids": emitted,
            "sha": hashlib.sha256(json.dumps(emitted).encode()).hexdigest(),
            "ttft_s": first - start,
            "decode_s": end - first,
            "tps": (len(emitted) - 1) / (end - first),
            "rounds": rounds,
            "commits_per_round": (len(emitted) - 1) / rounds,
            "round_ms": (end - first) * 1000 / rounds,
        }

    reference = {}
    parity = True
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as out:
        out.write(json.dumps(dict(part="snapshot", **fingerprint)) + "\n")
        for draft in arms.values():
            run(tok.encode("Hello."), draft, 24)
        for rep in range(a.reps):
            for ctx in a.contexts:
                for task, ask in tasks.items():
                    text = tok.decode(filler[:ctx]) + "\n\n" + ask
                    ids = chat_ids(tok, text)
                    key = (ctx, task)
                    if rep == 0:
                        plain = run(ids, None, a.tokens)
                        reference[key] = plain.pop("ids")
                        plain.update(
                            bits=0, rep=rep, context=ctx, task=task, mode="plain"
                        )
                        out.write(json.dumps(plain) + "\n")
                        out.flush()
                    order = list(arms) if rep % 2 == 0 else list(arms)[::-1]
                    for bits, fused, selector in order:
                        row = run(ids, arms[bits, fused, selector], a.tokens)
                        expected = reference[key]
                        row["parity"] = row.pop("ids") == expected
                        parity &= row["parity"]
                        row.update(
                            bits=bits,
                            rep=rep,
                            context=ctx,
                            task=task,
                            mode="dflash",
                            context_fused=fused,
                            selector=selector,
                        )
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                        print(json.dumps(row), flush=True)
                        if was_contended():
                            out.write(
                                json.dumps(
                                    {
                                        "complete": True,
                                        "success": False,
                                        "contended": True,
                                    }
                                )
                                + "\n"
                            )
                            return 0
                        mx.clear_cache()
        # Same seeded target stream with speculation off, q8 and q4. This is
        # correctness evidence; the short checks do not claim sampled speed.
        for seed in (17, 37, 1234):
            for task, ask in tasks.items():
                ids = chat_ids(tok, tok.decode(filler[:1024]) + "\n\n" + ask)
                expected = None
                for (bits, fused, selector), draft in [
                    ((0, False, False), None),
                    *arms.items(),
                ]:
                    row = run(ids, draft, 96, seed=seed)
                    expected = row["ids"] if expected is None else expected
                    row["parity"] = row.pop("ids") == expected
                    parity &= row["parity"]
                    row.update(
                        bits=bits,
                        seed=seed,
                        context=1024,
                        task=task,
                        mode="sampled-check",
                        context_fused=fused,
                        selector=selector,
                    )
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    print(json.dumps(row), flush=True)
        out.write(json.dumps({"complete": True, "parity": parity}) + "\n")
    return int(not parity)


if __name__ == "__main__":
    raise SystemExit(main())
