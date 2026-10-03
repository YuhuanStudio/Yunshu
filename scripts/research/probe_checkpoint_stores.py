"""Measure first-token checkpoint deferral on the serving runner, not a synthetic copy.

Run sync/deferred in separate gpuq jobs with the same explicit drafter and seed.
Each record includes the complete token-id digest, terminal reason and capture
count. --parity additionally cold-prefills both allow_draft modes and fails on
any token difference. Captures use the same checkpoint positions in both modes.
"""

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def digest(tokens):
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode()
    ).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("sync", "deferred"), required=True)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--stock-min-rows", type=int)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from yunshu_engine.apc_manager import _Coordinator
    from yunshu_engine.vlm_batch_runner import RunStats
    from yunshu_engine.vlm_engine import VLMEngine

    if a.stock_min_rows is not None:
        from yunshu_engine.kernels import lane_linear

        lane_linear.STOCK_MIN_ROWS = a.stock_min_rows
    if a.mode == "sync":
        _Coordinator.flush_deferred_checkpoints = None
    engine = VLMEngine("/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp")
    asyncio.run(engine.start())
    runner = engine._batch_runner
    coordinators = []
    bind = runner.apc_manager.coordinator

    def coordinator(model):
        c = bind(model)
        coordinators.append(c)
        return c

    runner.apc_manager.coordinator = coordinator
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("a") as out:

        def put(row):
            row.update(
                mode=a.mode, ctx=a.ctx, rep=a.rep, stock_min_rows=a.stock_min_rows
            )
            line = json.dumps(row)
            out.write(line + "\n")
            out.flush()
            print(line, flush=True)

        def ask(messages, phase, kind, draft=True):
            ids, kwargs, salt = engine._executor.submit(
                engine._runner_input,
                messages,
                [],
                [],
                False,
                {"enable_thinking": False},
            ).result()
            stats = RunStats()
            before = sum(
                getattr(c, "deferred_checkpoint_count", 0) for c in coordinators
            )
            start = time.perf_counter()
            first = None
            tokens = []
            for token in runner.iter_tokens(
                ids,
                max_tokens=a.tokens,
                temperature=0.0,
                seed=1234,
                allow_draft=draft,
                prompt_kwargs=kwargs,
                apc_semantic_hash=salt,
                stats=stats,
            ):
                if first is None:
                    first = time.perf_counter() - start
                tokens.append(int(token))
            wall = time.perf_counter() - start
            engine._executor.submit(lambda: None).result()
            if not tokens or not stats.finish_reason:
                raise RuntimeError(f"incomplete {kind}/{phase}: {stats.finish_reason}")
            if draft and not stats.used_draft:
                raise RuntimeError("requested draft did not engage")
            captures = (
                sum(getattr(c, "deferred_checkpoint_count", 0) for c in coordinators)
                - before
            )
            put(
                dict(
                    phase=phase,
                    kind=kind,
                    prompt_tokens=len(ids),
                    cached=stats.cached_tokens,
                    ttft_s=first,
                    wall_s=wall,
                    tokens=len(tokens),
                    token_sha256=digest(tokens),
                    token_ids=tokens,
                    finish_reason=stats.finish_reason,
                    spec_mode=stats.spec_mode,
                    used_draft=stats.used_draft,
                    deferred_checkpoints=captures,
                )
            )
            return tokens

        try:
            ask([{"role": "user", "content": "Say hi."}], "compile", "short")
            for kind in ("prose", "code"):
                path = (
                    Path("/Volumes/P5Plus/yunshu-build/tfnew/prompts")
                    / f"{kind}-{a.ctx}.txt"
                )
                text = path.read_text()
                messages = [{"role": "user", "content": text}]
                runner.apc_manager.clear()
                tokens = ask(messages, "cold", kind)
                ask(messages, "warm", kind)
                extended = messages + [
                    {"role": "assistant", "content": engine._tokenizer.decode(tokens)},
                    {
                        "role": "user",
                        "content": "Continue with the next part, at the same length.",
                    },
                ]
                ask(extended, "turn2", kind)
                if a.parity:
                    runner.apc_manager.clear()
                    plain = ask(messages, "spec_off_cold", kind, draft=False)
                    runner.apc_manager.clear()
                    accelerated = ask(messages, "spec_on_cold", kind)
                    put(
                        dict(
                            phase="parity",
                            kind=kind,
                            equal=plain == accelerated,
                            first_diff=next(
                                (
                                    i
                                    for i, (x, y) in enumerate(
                                        zip(plain, accelerated, strict=False)
                                    )
                                    if x != y
                                ),
                                None,
                            ),
                        )
                    )
                    if plain != accelerated:
                        raise RuntimeError(f"{kind}: spec on/off token parity failed")
            put(dict(phase="complete", success=True))
        finally:
            asyncio.run(engine.stop())


if __name__ == "__main__":
    main()
