"""Separate one-token revisit overhead from APC clones and generator work."""

import argparse
import asyncio
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from mlx_vlm.apc_coordinator import APCCoordinator

    from yunshu_engine.apc_manager import _Coordinator
    from yunshu_engine.vlm_batch_runner import RunStats
    from yunshu_engine.vlm_engine import VLMEngine

    engine = VLMEngine("/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp")
    asyncio.run(engine.start())
    runner = engine._batch_runner
    flush = _Coordinator.flush_deferred_checkpoints
    merge = _Coordinator.merge_rows
    events, original = [], []
    for modname, symbol in [
        ("mlx_vlm.apc", "APCManager.lookup_exact_cache"),
        ("mlx_vlm.apc", "_clone_prompt_cache_for_apc"),
        ("mlx_vlm.generate.ar", "PromptProcessingBatch.prompt_step"),
        ("mlx_vlm.generate.ar", "PromptProcessingBatch.generate"),
        ("yunshu_engine.vlm_batch_runner", "VLMBatchRunner._admit"),
        ("yunshu_engine.vlm_batch_runner", "VLMBatchRunner._step_generator"),
    ]:
        obj = importlib.import_module(modname)
        *parents, name = symbol.split(".")
        for parent in parents:
            obj = getattr(obj, parent)
        fn = getattr(obj, name)

        def wrap(*args, _fn=fn, _name=symbol, **kwargs):
            start = time.perf_counter()
            try:
                return _fn(*args, **kwargs)
            finally:
                events.append(dict(name=_name, ms=1000 * (time.perf_counter() - start)))

        original.append((obj, name, fn))
        setattr(obj, name, wrap)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = (
            Path("/Volumes/P5Plus/yunshu-build/tfnew/prompts") / f"prose-{a.ctx}.txt"
        ).read_text()
        ids, _, salt = engine._executor.submit(
            engine._runner_input,
            [{"role": "user", "content": text}],
            [],
            [],
            False,
            {"enable_thinking": False},
        ).result()
        with a.out.open("w") as out:

            def run(tokens, label, mode, rep, n=16):
                events.clear()
                stats = RunStats()
                start = time.perf_counter()
                first, generated = None, []
                for tok in runner.iter_tokens(
                    tokens,
                    max_tokens=n,
                    temperature=0,
                    seed=1234,
                    apc_semantic_hash=salt,
                    stats=stats,
                ):
                    if first is None:
                        first = time.perf_counter() - start
                    generated.append(tok)
                if not generated or not stats.finish_reason:
                    raise RuntimeError("incomplete fixed-cost request")
                if not stats.used_draft:
                    raise RuntimeError("fixed-cost probe did not engage the drafter")
                # DONE can reach the consumer before its executor slice finishes.
                # Settle deferred publication / cleanup before changing APC or
                # attributing the next request's component events.
                engine._executor.submit(lambda: None).result()
                row = dict(
                    label=label,
                    mode=mode,
                    rep=rep,
                    ctx=a.ctx,
                    ttft_ms=first * 1000,
                    cached=stats.cached_tokens,
                    fresh=len(tokens) - stats.cached_tokens,
                    digest=hashlib.sha256(json.dumps(generated).encode()).hexdigest(),
                    used_draft=stats.used_draft,
                    spec_mode=stats.spec_mode,
                    events=list(events),
                )
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(json.dumps(row), flush=True)

            run(ids[:16], "compile", "deferred", -1)
            for rep in range(3):
                for mode in (
                    ("sync", "deferred", "reserved")
                    if rep != 1
                    else ("reserved", "deferred", "sync")
                ):
                    _Coordinator.flush_deferred_checkpoints = (
                        None if mode == "sync" else flush
                    )
                    _Coordinator.merge_rows = (
                        merge if mode == "reserved" else APCCoordinator.merge_rows
                    )
                    runner.apc_manager.clear()
                    run(ids, "prime", mode, rep)
                    for revisit in range(3):
                        run(ids, f"revisit-{revisit}", mode, rep)
                    # Same cached prefix, two checkpoint captures close to its end.
                    run(ids[:-1] + ids[-65:] + ids[-1:], "suffix-65", mode, rep)
            out.write(json.dumps(dict(phase="complete", success=True)) + "\n")
    finally:
        _Coordinator.flush_deferred_checkpoints = flush
        _Coordinator.merge_rows = merge
        for obj, name, fn in original:
            setattr(obj, name, fn)
        asyncio.run(engine.stop())


if __name__ == "__main__":
    main()
