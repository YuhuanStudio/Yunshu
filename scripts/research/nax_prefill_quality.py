"""Paired 200-question cold arithmetic gate with raw IDs and all token LPs.

The actual prompt is exactly 2049 tokens, so its first 2048-row prefill must
exercise the candidate. Run via gpuq without --quiet (correctness job).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))


def question(index):
    left, right = 100 + (index * 7919) % 900, 10 + (index * 97) % 90
    op = "+" if index % 2 else "-"
    answer = left + right if op == "+" else left - right
    prompt = (
        "This is neutral background context; the final arithmetic question is the task. "
        * 300
    )
    return (
        prompt + f"\nCompute {left} {op} {right}. Reply with only the integer.",
        answer,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--arm",
        choices=("tile128", "lane64", "lane128", "narrow", "combo"),
        required=True,
    )
    p.add_argument("--items", type=int, default=200)
    p.add_argument(
        "--model", default="/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.items < 1:
        p.error("--items must be positive")
    if a.dry_run:
        print(json.dumps(vars(a), default=str))
        return
    import nax_prefill_dispatch as dispatch

    from yunshu_engine import settings
    from yunshu_engine.vlm_batch_runner import RunStats
    from yunshu_engine.vlm_engine import VLMEngine

    if "27B" in a.model and (
        settings.get_str("YUNSHU_VLM_DRAFT") != "mtp"
        or settings.get_bool("YUNSHU_VLM_APC_DISK")
    ):
        raise RuntimeError("27B gate requires explicit MTP and RAM-only APC")
    if a.out.exists():
        raise FileExistsError(a.out)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    engine = VLMEngine(a.model)
    asyncio.run(engine.start())
    runner = engine._batch_runner
    tok = getattr(runner.processor, "tokenizer", runner.processor)
    correct = {"base": 0, a.arm: 0}
    different = 0
    try:
        with a.out.open("w") as out:
            for index in range(a.items):
                prompt, answer = question(index)
                ids, _, salt = engine._executor.submit(
                    engine._runner_input,
                    [{"role": "user", "content": prompt}],
                    [],
                    [],
                    False,
                    {"enable_thinking": False},
                ).result()
                if len(ids) < 2049:
                    raise RuntimeError("insufficient neutral padding")
                # Remove only the middle of the neutral padding; preserve all
                # question/template tokens in the final 128-token tail.
                ids = list(ids[:1921]) + list(ids[-128:])
                results = {}
                for arm in ("base", a.arm) if index % 2 == 0 else (a.arm, "base"):
                    engine._executor.submit(dispatch.install, arm).result()
                    runner.apc_manager.clear()
                    stats = RunStats()
                    tokens, lps = [], []
                    for token in runner.iter_tokens(
                        ids,
                        max_tokens=16,
                        temperature=0,
                        seed=1234,
                        apc_semantic_hash=salt,
                        stats=stats,
                        logprobs=True,
                    ):
                        tokens.append(token)
                        lps.append(stats.last_logprob["logprob"])
                    engine._executor.submit(lambda: None).result()
                    if not tokens or not stats.finish_reason or stats.cached_tokens:
                        raise RuntimeError("incomplete / non-cold paired request")
                    count = sum(dispatch.calls.values())
                    if arm != "base" and "27B" in a.model and not count:
                        raise RuntimeError(
                            "candidate did not engage on the long prompt"
                        )
                    text = tok.decode(tokens, skip_special_tokens=True).strip()
                    match = re.fullmatch(r"\s*(-?\d+)\s*", text)
                    scored = bool(match and int(match[1]) == answer)
                    correct[arm] += scored
                    results[arm] = dict(
                        tokens=tokens,
                        lps=lps,
                        text=text,
                        correct=scored,
                        cached=stats.cached_tokens,
                        dispatch_calls=count,
                    )
                equal = (
                    results["base"]["tokens"] == results[a.arm]["tokens"]
                    and results["base"]["lps"] == results[a.arm]["lps"]
                )
                different += not equal
                row = dict(
                    index=index,
                    prompt_tokens=len(ids),
                    prompt_id_sha256=hashlib.sha256(
                        json.dumps(ids).encode()
                    ).hexdigest(),
                    answer=answer,
                    equal=equal,
                    arms=results,
                )
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(
                    json.dumps({k: v for k, v in row.items() if k != "arms"}),
                    flush=True,
                )
            success = different == 0 and abs(correct["base"] - correct[a.arm]) <= 1
            out.write(
                json.dumps(
                    dict(
                        phase="complete",
                        success=success,
                        items=a.items,
                        correct=correct,
                        different=different,
                        arm=a.arm,
                    )
                )
                + "\n"
            )
            if not success:
                raise RuntimeError("paired raw IDs / LPs / accuracy gate failed")
    finally:
        engine._executor.submit(dispatch.install, "base").result()
        asyncio.run(engine.stop())


if __name__ == "__main__":
    main()
