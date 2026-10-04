"""200 paired questions through the actual default auto serving round hook."""

import argparse, hashlib, json, sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--dry-run", action="store_true")
p.add_argument("--items", type=int, default=200)
p.add_argument("--start", type=int, default=0)
p.add_argument("--context", type=int, default=0, help="0 = CONTEXT_LIMIT-10")
p.add_argument("--arms", default="off,auto")
p.add_argument(
    "--track", type=int, default=0, help="emit memory/array counts every N items"
)
p.add_argument(
    "--contexts", default="", help="comma list cycled by item (overrides --context)"
)
p.add_argument("--max-tokens", type=int, default=4096)
a = p.parse_args()
if a.dry_run:
    print(
        json.dumps(
            dict(
                dry_run=True,
                paired_items=a.items,
                context="padded 1K-class",
                arms=["off", "auto"],
            )
        )
    )
    sys.exit()
ROOT = Path("/Volumes/P5Plus/yunshu-build/codex/worktrees/wide-lead")
sys.path.insert(0, str(ROOT / "scripts/research"))
from spec_bench_snapshot import freeze

source, fingerprint = freeze(a.output)
sys.path.insert(0, str(source))
import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.generate.ar import BatchGenerator
from mlx_vlm.speculative.drafters import load_drafter
from yunshu_engine import dflash_fast, dflash_copy, mtp_lane, settings
from yunshu_engine.dflash_tree import quantize_drafter
from yunshu_engine.dflash_context import install as context_install
from yunshu_engine.kernels import omlx, lane_linear, gdn_prefill, ragged_kv
from yunshu_engine.kernels.batch_invariant import install, set_active
from yunshu_engine.kernels import nax_prefill
from yunshu_engine.mrope import clear_rope_state

M = "/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
D = "/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2"
omlx.apply(row_exact=False)
model, processor = load(M)
lm = model.language_model
tok = processor.tokenizer
lane_linear.convert(lm)
lane_linear.set_stock_rows(512)
install(lm, model=model, packed=False)
set_active(True)
gdn_prefill.install()
ragged_kv.install()
ragged_kv.set_dense_lane(True)
context_install(lm)
if nax_prefill.enable():
    nax_prefill.warmup(
        {m.bits for _, m in lm.named_modules() if isinstance(m, lane_linear.LaneLinear)}
    )
draft, kind = load_drafter(D, kind="dflash")
quantize_drafter(draft, 8)
assert dflash_fast.supported(lm, draft)
dflash_fast.prepare(draft)
rows, enabled = dflash_copy.configure(lm, 16, invariant=True, lane_projections=True)
assert enabled and rows == 16
assert settings.get("YUNSHU_SPEC_TREE") == "auto"
lane_linear.configure_sum_policy(True)
actual = [0]
handoffs = [0]
points = []
raw_resume = dflash_copy.resume_chain


def resumed(*args, **kw):
    handoffs[0] += 1
    context = len(mtp_lane._STATE["context"])
    generated = a.max_tokens - kw["max_tokens"] + 1
    points.append(dict(context=context, generated=generated, total=context + generated))
    yield from raw_resume(*args, **kw)


dflash_copy.resume_chain = resumed
raw = dflash_fast.rounds


def traced(*args, **kw):
    actual[0] += 1
    yield from raw(*args, **kw)


dflash_fast.rounds = traced
padding = tok.encode(
    "".join(f"Sensor {i}: pressure {i * 37 % 1000}.\n" for i in range(2000)),
    add_special_tokens=False,
)
filler = "Background only; ignore these notes when answering.\n" + "".join(
    f"Sensor {i}: pressure {i * 37 % 1000}.\n" for i in range(80)
)
arms = a.arms.split(",")
correct = {"off": 0, "auto": 0}
parity = True
with a.output.open("x") as out:

    def emit(r):
        out.write(json.dumps(r) + "\n")
        out.flush()
        print(json.dumps(r), flush=True)

    emit(
        dict(
            part="snapshot",
            mode=kind,
            default_tree=settings.get("YUNSHU_SPEC_TREE"),
            **fingerprint,
        )
    )
    for item in range(a.start, a.start + a.items):
        expected = (
            f"def bump_{item}(value):\n    return value + {item}"
            if item % 2
            else f"The cedar tree beside station {item} shelters twelve quiet birds during the evening rain."
        )
        ask = (
            "Repeat the following exactly. Return only this text, no fences or explanations:\n"
            + expected
        )
        ids = tok.apply_chat_template(
            [{"role": "user", "content": ask}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        cycle = [int(c) for c in a.contexts.split(",") if c]
        desired = (
            cycle[item % len(cycle)]
            if cycle
            else a.context or dflash_fast.CONTEXT_LIMIT - 10
        )
        ids = ids[:3] + padding[: desired - len(ids)] + ids[3:]
        assert len(ids) == desired, len(ids)
        refs = []
        for arm in arms if item % 2 == 0 else arms[::-1]:
            mtp_lane.set_context(ids)
            before = actual[0]
            gen = BatchGenerator(
                lm,
                processor,
                max_tokens=a.max_tokens,
                draft_model=draft if arm == "auto" else None,
                draft_kind="dflash" if arm == "auto" else None,
                draft_block_size=8,
                greedy_sampling=True,
                compute_logprobs=False,
            )
            clear_rope_state(model)
            kwargs = model.get_input_embeddings(
                mx.array(ids)[None], None, mask=None
            ).to_dict()
            uid = gen.insert([ids], max_tokens=a.max_tokens, prompt_kwargs=[kwargs])[0]
            tokens = []
            done = False
            try:
                while not done:
                    _, responses = gen.next()
                    for response in responses:
                        if response.uid == uid:
                            tokens.append(int(response.token))
                            done |= response.finish_reason is not None
            finally:
                gen.close()
                mtp_lane.set_context(None)
            if arm == "auto":
                assert actual[0] > before, "actual fast path did not engage"
            sha = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
            refs.append(sha)
            scored = (
                tok.decode(tokens, skip_special_tokens=True).strip() == expected.strip()
            )
            correct[arm] += int(scored)
            emit(
                dict(
                    part="quality",
                    item=item,
                    arm=arm,
                    sha=sha,
                    correct=scored,
                    actual_fast=actual[0] - before,
                    prompt_tokens=len(ids),
                )
            )
        parity &= len(set(refs)) == 1
        if a.track and (item - a.start) % a.track == 0:
            import gc
            import resource

            emit(
                dict(
                    part="track",
                    item=item,
                    active_gb=mx.get_active_memory() / 1e9,
                    cache_gb=mx.get_cache_memory() / 1e9,
                    arrays=sum(1 for o in gc.get_objects() if isinstance(o, mx.array)),
                    rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
                )
            )
        mx.clear_cache()
    switch = not a.context or a.context + a.max_tokens > dflash_fast.CONTEXT_LIMIT
    delta = abs(correct["off"] - correct["auto"]) if len(arms) == 2 else 0
    success = (
        parity
        and delta <= 1
        and (
            not switch
            or (
                handoffs[0] > 0
                and all(point["total"] == dflash_fast.CONTEXT_LIMIT for point in points)
            )
        )
    )
    emit(
        dict(
            complete=True,
            success=success,
            paired_items=a.items,
            correct=correct,
            net_correct_difference=delta,
            all_digest_equal=parity,
            actual_fast_requests=actual[0],
            actual_chain_handoffs=handoffs[0],
            handoff_points=points,
            context_limit=dflash_fast.CONTEXT_LIMIT,
        )
    )
if not success:
    raise SystemExit(1)
