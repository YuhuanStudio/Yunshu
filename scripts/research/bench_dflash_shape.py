"""Full-shape DFlash2 proposals and round telemetry; run GPU arms only via gpuq."""

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("drafter")
    p.add_argument(
        "--arms",
        nargs="+",
        default=["main", "chain8", "chain12", "chain16", "selector16"],
    )
    p.add_argument("--contexts", nargs="+", type=int, default=[1024, 8192, 32768])
    p.add_argument(
        "--tasks", nargs="+", choices=["code", "prose"], default=["code", "prose"]
    )
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--barrier", action="store_true")
    p.add_argument("--quality-items", type=int, default=0)
    p.add_argument("--capture-proposals", action="store_true")
    p.add_argument(
        "--prototype-dir",
        type=Path,
        default=Path("/Volumes/P5Plus/yunshu-build/codex/wide4"),
    )
    p.add_argument("--corpus", choices=["tfbench", "sensor"], default="tfbench")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    for arm in a.arms:
        if arm != "main" and arm not in [
            "chain8",
            "chain12",
            "chain16",
            "selector8",
            "selector12",
            "selector16",
            "selectorgpu16",
            "tree7",
            "tree15",
            "tree15gpu",
            "tree15gpuplan",
            "tree15gpucompact",
            "copy8",
            "copy16",
            "shape16",
            "nativecopy8",
        ]:
            p.error(f"unknown arm: {arm}")
    if a.quality_items < 0:
        p.error("quality-items must be nonnegative")
    if a.tokens < 2 or a.reps < 1:
        p.error("tokens >= 2 and reps >= 1 required")
    return a


def trim_trace_budget(records, tokens, *, speculative):
    """Only published tokens count; a first-token EOS has no decode round."""
    room = tokens - 1
    for record in records:
        record["committed_uncut"] = record["committed"]
        record["committed"] = min(record["committed"], room)
        room -= record["committed"]
    if speculative and (room != 0 or (tokens > 1 and not records)):
        raise RuntimeError(f"untraced speculative output: {room} tokens missing")


def main():
    a = arguments()
    if a.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "arms": a.arms,
                    "cells": a.quality_items or len(a.contexts) * len(a.tasks) * a.reps,
                }
            )
        )
        return
    from spec_bench_snapshot import freeze, refuse_contended

    if refuse_contended(a.output):
        return
    source, fingerprint = freeze(a.output)
    frozen_prototypes = a.output.with_name(a.output.stem + "-prototypes")
    frozen_prototypes.mkdir()
    for name in (
        "selector_gpu.py",
        "tree_gpu.py",
        "shape_budget.py",
        "tree_plan_gpu.py",
        "tree_tail_compact.py",
    ):
        if (a.prototype_dir / name).is_file():
            shutil.copy2(a.prototype_dir / name, frozen_prototypes / name)
    a.prototype_dir = frozen_prototypes
    sys.path.insert(0, str(source))
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.generate.ar import BatchGenerator
    from mlx_vlm.speculative import dflash
    from mlx_vlm.speculative import utils as spec_utils
    from mlx_vlm.speculative.drafters import load_drafter

    from yunshu_engine import dflash_tree
    from yunshu_engine.copy_drafter import CopyDrafter
    from yunshu_engine.dflash_context import install as install_context
    from yunshu_engine.dflash_copy import CopyDraft
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
    install_context(lm)
    install_chain_budget()
    choose_main = dflash._dflash_next_block_size
    draft, kind = load_drafter(a.drafter, kind="dflash")
    quantize_drafter(draft, 8)
    raw_greedy, selector = draft.draft_block_greedy, draft.draft_block
    print(
        f"Speculative decoding: {kind}; target={a.model}; q8; trained_block={draft.config.block_size}",
        flush=True,
    )
    rounds, current = [], {}
    copy = None
    next_copy = []
    pending_hidden = None
    active_ids = []
    shape_budget = None
    raw_rounds = spec_utils._dflash_rounds
    raw_search = dflash_tree.search_tree
    raw_lattice = dflash_tree.compute_lattice_gpu
    raw_tree_walk = dflash_tree.walk
    raw_shape = dflash_tree.tv.DynamicShape
    raw_attention = dflash_tree.tv.tree_attention
    raw_tree_forward = dflash_tree.tv.tree_forward
    sys.path.insert(0, str(a.prototype_dir))
    gpu_proposal = None
    if "selectorgpu16" in a.arms:
        from selector_gpu import install as install_gpu_selector

        gpu_proposal = install_gpu_selector(draft)

    def lattice(*args, **kw):
        current.clear()
        current["started"] = time.perf_counter()
        t = current["started"]
        lat = raw_lattice(*args, **kw)
        if a.barrier:
            mx.eval(lat.cands, lat.unary, lat.hproj, lat.succ, lat.pred, lat.anchor)
        current["draft_ms" if a.barrier else "draft_build_ms"] = (
            time.perf_counter() - t
        ) * 1000
        current["proposal_block"] = args[-1] + 1
        return lat

    def tree_forward(lm, window, shape, cache, capture_ids=()):
        if shape.width == 1:
            current.clear()
            current.update(
                started=time.perf_counter(),
                proposal_block=1,
                **{"draft_ms" if a.barrier else "draft_build_ms": 0.0},
            )
        return raw_tree_forward(lm, window, shape, cache, capture_ids)

    dflash_tree.tv.tree_forward = tree_forward

    def tree_walk(window, parents, target):
        path = raw_tree_walk(window, parents, target)
        depths = [0]
        for par in parents[1:]:
            depths.append(depths[par] + 1)
        current.update(
            accepted=len(path) - 1,
            committed=len(path),
            verify_rows=len(window),
            max_depth=max(depths),
            proposal_kind="tree",
            new_tokens=[window[r] for r in path[1:]] + [target[path[-1]]],
            round_to_walk_ms=(time.perf_counter() - current["started"]) * 1000,
        )
        if a.capture_proposals:
            current.update(window=window, parents=parents, target_tokens=target)
        rounds.append(dict(current))
        return path

    def scoped_rounds(*args, **kw):
        nonlocal copy, pending_hidden, shape_budget
        shape_budget = None
        if active_arm == "shape16":
            from shape_budget import ShapeBudget

            shape_budget = ShapeBudget()
        copy = None
        pending_hidden = None
        if active_arm.startswith("copy"):
            copy = CopyDrafter(max_draft=15)
            copy.extend(active_ids)
            copy.extend([kw["first_bonus"]])
        if active_arm == "nativecopy8":
            proxy = CopyDraft(draft, active_ids, int(kw["first_bonus"]), 16)
            iterator = raw_rounds(args[0], proxy, *args[2:], **kw)
            try:
                for token, state in iterator:
                    proxy.emitted(int(token))
                    yield token, state
            finally:
                iterator.close()
                proxy.close()
        elif active_arm.startswith("tree"):
            dflash_tree.POSITIONS = 15 if active_arm != "tree7" else 7
            dflash_tree.compute_lattice_gpu = lattice
            dflash_tree.walk = tree_walk
            dflash_tree.tv.DynamicShape = raw_shape
            dflash_tree.tv.tree_attention = raw_attention
            if active_arm in ("tree15gpuplan", "tree15gpucompact"):
                from tree_plan_gpu import DeviceShape

                dflash_tree.tv.DynamicShape = DeviceShape
            if active_arm == "tree15gpucompact":
                from tree_tail_compact import build

                dflash_tree.tv.tree_attention = build(dflash_tree.tv)
            if active_arm.startswith("tree15gpu"):
                from tree_gpu import search

                dflash_tree.search_tree = search
            else:
                dflash_tree.search_tree = raw_search
            yield from dflash_tree.dflash_tree_rounds(*args, _original=raw_rounds, **kw)
        else:
            yield from raw_rounds(*args, **kw)

    spec_utils._dflash_rounds = scoped_rounds

    def choose(dm, ceiling, remaining, initial=None):
        nonlocal next_copy
        current.clear()
        current["started"] = time.perf_counter()
        bs = (
            dm.choose(ceiling, remaining)
            if isinstance(dm, CopyDraft)
            else choose_main(dm, ceiling, remaining, initial)
            if active_arm == "main"
            else min(ceiling, remaining)
        )
        if shape_budget is not None:
            bs = shape_budget.choose(remaining)
        next_copy = copy.draft(min(15, remaining - 1)) if copy else []
        if next_copy:
            bs = len(next_copy) + 1
        current["proposal_block"] = bs
        current["proposal_kind"] = (
            "copy"
            if next_copy or (isinstance(dm, CopyDraft) and dm._next_copy)
            else "model"
        )
        if current["proposal_kind"] == "copy":
            current["draft_ms" if a.barrier else "draft_build_ms"] = 0.0
        return bs

    def propose(*args, **kwargs):
        nonlocal pending_hidden
        started = time.perf_counter()
        fn = selector if active_arm.startswith("selector") else raw_greedy
        if active_arm == "selectorgpu16":
            fn = gpu_proposal
        if next_copy:
            pending_hidden = (
                args[1]
                if pending_hidden is None
                else mx.concatenate([pending_hidden, args[1]], axis=1)
            )
            out = mx.array([next_copy], dtype=mx.int32)
        else:
            if pending_hidden is not None:
                args = (
                    args[0],
                    mx.concatenate([pending_hidden, args[1]], axis=1),
                    *args[2:],
                )
                pending_hidden = None
            out = fn(*args, **kwargs)
        if a.barrier:
            mx.eval(out)
        current["draft_ms" if a.barrier else "draft_build_ms"] = (
            time.perf_counter() - started
        ) * 1000
        return out

    original_walk = dflash._speculative_walk

    def walk(drafts, targets, room):
        accepted, tokens = original_walk(drafts, targets, room)
        current.update(
            accepted=accepted,
            verify_rows=int(drafts.shape[1]) + 1,
            committed=len(tokens),
            round_to_walk_ms=(time.perf_counter() - current["started"]) * 1000,
        )
        current["new_tokens"] = list(tokens)
        if a.capture_proposals:
            current["draft_tokens"] = drafts.reshape(-1).tolist()
            current["target_tokens"] = targets.reshape(-1).tolist()
        rounds.append(dict(current))
        if shape_budget is not None and len(rounds) > 1:
            shape_budget.observe(
                current["proposal_block"],
                len(tokens),
                current["round_to_walk_ms"],
                censored=room < len(tokens) + 1,
            )
        if copy:
            if next_copy:
                copy.observe_copy(len(next_copy), accepted)
            else:
                copy.observe_model(len(tokens))
            copy.extend(tokens)
        return accepted, tokens

    dflash._dflash_next_block_size = choose
    dflash._speculative_walk = walk
    object.__setattr__(draft, "draft_block_greedy", propose)
    filler = (
        tok.encode(
            "".join(f"Sensor {i}: pressure {i * 37 % 1000}.\n" for i in range(40000)),
            add_special_tokens=False,
        )
        if a.corpus == "sensor" and not a.quality_items
        else None
    )
    tasks = {
        "code": "Write a Python LRU cache class with get, put, delete and resize. Output code only.",
        "prose": "Explain in detail how a refrigerator works, including compressor and evaporator.",
    }
    active_arm = "main"

    def run(ids, arm, limit):
        nonlocal active_arm, active_ids
        active_arm = arm
        active_ids = ids
        rounds.clear()
        block = (
            8
            if arm in ("main", "off")
            else int(
                arm.removeprefix("nativecopy")
                .removeprefix("chain")
                .removeprefix("selectorgpu")
                .removeprefix("selector")
                .removeprefix("copy")
            )
            if not arm.startswith("tree") and arm != "shape16"
            else 16
        )
        gen = BatchGenerator(
            lm,
            processor,
            max_tokens=limit,
            draft_model=draft if arm != "off" else None,
            draft_kind="dflash" if arm != "off" else None,
            draft_block_size=block,
            greedy_sampling=True,
            compute_logprobs=False,
        )
        clear_rope_state(model)
        kwargs = model.get_input_embeddings(
            mx.array(ids)[None], None, mask=None
        ).to_dict()
        started = time.perf_counter()
        uid = gen.insert([ids], max_tokens=limit, prompt_kwargs=[kwargs])[0]
        emitted, first, done = [], None, False
        try:
            while not done:
                _, responses = gen.next()
                for response in responses:
                    if response.uid == uid:
                        first = time.perf_counter() if first is None else first
                        emitted.append(int(response.token))
                        done |= response.finish_reason is not None
        finally:
            gen.close()
        mx.synchronize()
        elapsed = time.perf_counter() - first
        trim_trace_budget(rounds, len(emitted), speculative=arm != "off")
        return dict(
            sha=hashlib.sha256(json.dumps(emitted).encode()).hexdigest(),
            text=tok.decode(emitted, skip_special_tokens=True),
            ids=emitted,
            tokens=len(emitted),
            ttft_s=first - started,
            decode_s=elapsed,
            tps=(len(emitted) - 1) / elapsed,
            rounds=list(rounds),
            acceptance_histogram=dict(Counter(r["accepted"] for r in rounds)),
            blocks=dict(Counter(r["proposal_block"] for r in rounds)),
            commits_per_round=(len(emitted) - 1) / len(rounds) if rounds else None,
            round_ms=elapsed * 1000 / len(rounds) if rounds else None,
        )

    a.output.parent.mkdir(parents=True, exist_ok=True)
    parity = True
    with a.output.open("x") as out:

        def emit(row):
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(
                json.dumps(
                    {k: v for k, v in row.items() if k not in ("rounds", "ids", "text")}
                ),
                flush=True,
            )

        emit(
            dict(
                part="snapshot",
                mode=kind,
                trained_block=int(draft.config.block_size),
                model_metadata={
                    name: {
                        "config_sha256": hashlib.sha256(
                            (Path(directory) / "config.json").read_bytes()
                        ).hexdigest(),
                        "weights": [
                            {
                                "name": f.name,
                                "bytes": f.stat().st_size,
                                "mtime_ns": f.stat().st_mtime_ns,
                            }
                            for f in sorted(Path(directory).glob("*.safetensors"))
                        ],
                    }
                    for name, directory in [("target", a.model), ("drafter", a.drafter)]
                },
                barrier=a.barrier,
                capture_proposals=a.capture_proposals,
                arms=a.arms,
                contexts=a.contexts,
                tasks=a.tasks,
                reps=a.reps,
                corpus=a.corpus,
                prototype_sha256={
                    f: hashlib.sha256((a.prototype_dir / f).read_bytes()).hexdigest()
                    for f in (
                        "selector_gpu.py",
                        "tree_gpu.py",
                        "shape_budget.py",
                        "tree_plan_gpu.py",
                        "tree_tail_compact.py",
                    )
                    if (a.prototype_dir / f).is_file()
                },
                **fingerprint,
            )
        )
        failed_arms = set()
        for arm in a.arms:
            try:
                run(tok.encode("Hello."), arm, 16)
            except Exception as exc:
                failed_arms.add(arm)
                parity = False
                emit(
                    dict(part="failure", phase="warmup", arm=arm, rc=1, error=repr(exc))
                )
        refs = {}
        prompt_refs = {}
        if a.quality_items:
            correct = {arm: 0 for arm in ["off", *a.arms]}
            for item in range(a.quality_items):
                if item % 2:
                    expected = f"def bump_{item}(value):\n    return value + {item}"
                    ask = (
                        "Repeat the following code exactly. Return code only, without fences or explanations:\n"
                        + expected
                    )
                else:
                    expected = f"The cedar tree beside station {item} shelters twelve quiet birds during the evening rain."
                    ask = (
                        "Repeat the following sentence exactly, without quotation marks or any additional words:\n"
                        + expected
                    )
                ids = tok.apply_chat_template(
                    [{"role": "user", "content": ask}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
                baseline = None
                for arm in ["off", *a.arms] if item % 2 == 0 else [*a.arms, "off"]:
                    row = run(ids, arm, 96)
                    scored = row["text"].strip() == expected.strip()
                    correct[arm] += int(scored)
                    if baseline is None:
                        baseline = row["sha"]
                    equal = baseline == row["sha"]
                    parity &= equal
                    emit(
                        dict(
                            part="quality",
                            item=item,
                            arm=arm,
                            correct=scored,
                            parity=equal,
                            **row,
                        )
                    )
                mx.clear_cache()
            delta = max(correct.values()) - min(correct.values())
            emit(
                dict(
                    complete=True,
                    success=parity and delta <= 1,
                    mode=kind,
                    paired_items=a.quality_items,
                    correct=correct,
                    net_correct_difference=delta,
                )
            )
            if not parity or delta > 1:
                raise SystemExit(1)
            return
        for rep in range(a.reps):
            for ctx in a.contexts:
                for task in a.tasks:
                    text = (
                        (
                            Path("/Volumes/P5Plus/yunshu-build/tfnew/prompts")
                            / f"{task}-{ctx}.txt"
                        ).read_text()
                        if a.corpus == "tfbench"
                        else tok.decode(filler[:ctx]) + "\n\n" + tasks[task]
                    )
                    ids = tok.apply_chat_template(
                        [{"role": "user", "content": text}],
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
                    prompt_sha = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
                    if rep == 0:
                        prompt_refs[ctx, task] = prompt_sha
                        emit(
                            dict(
                                part="prompt",
                                context=ctx,
                                task=task,
                                ids=ids,
                                input_sha256=prompt_sha,
                                text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                prompt_tokens=len(ids),
                            )
                        )
                        row = run(ids, "off", a.tokens)
                        refs[ctx, task] = row["sha"]
                        emit(
                            dict(
                                part="reference",
                                context=ctx,
                                task=task,
                                arm="off",
                                rep=rep,
                                **row,
                            )
                        )
                    if prompt_refs[ctx, task] != prompt_sha:
                        raise RuntimeError("prompt changed during matrix")
                    for arm in a.arms if rep % 2 == 0 else a.arms[::-1]:
                        if arm in failed_arms:
                            continue
                        try:
                            row = run(ids, arm, a.tokens)
                        except Exception as exc:
                            failed_arms.add(arm)
                            parity = False
                            emit(
                                dict(
                                    part="failure",
                                    phase="matrix",
                                    context=ctx,
                                    task=task,
                                    rep=rep,
                                    arm=arm,
                                    rc=1,
                                    error=repr(exc),
                                )
                            )
                            continue
                        equal = row["sha"] == refs[ctx, task]
                        parity &= equal
                        emit(
                            dict(
                                part="result",
                                context=ctx,
                                task=task,
                                arm=arm,
                                rep=rep,
                                parity=equal,
                                **row,
                            )
                        )
                        mx.clear_cache()
        emit(dict(complete=True, success=parity, mode=kind))
    if not parity:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
