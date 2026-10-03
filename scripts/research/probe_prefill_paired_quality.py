"""Modest paired arithmetic gate for the research-only wide lane dispatch.

Score both arms on the same 200 questions. Equivalent token/logprob outputs
score identically; a correctness difference above one item fails closed.
"""

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
    operation = "+" if index % 2 else "-"
    answer = left + right if operation == "+" else left - right
    # Exercise different suffix sizes and final dispatch tails (129..512).
    padding = "This is background context; the final arithmetic question is the task. "
    prompt = padding * (10 + index % 25)
    prompt += f"\nCompute {left} {operation} {right}. Reply with only the integer."
    return prompt, answer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--variant", choices=("wide", "native"), default="wide")
    ap.add_argument(
        "--model", default="/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
    )
    a = ap.parse_args()
    from yunshu_engine.kernels import lane_linear
    from yunshu_engine.kernels.tensorfold import lane_qmm
    from yunshu_engine.vlm_batch_runner import RunStats
    from yunshu_engine.vlm_engine import VLMEngine

    if "27B" in a.model:
        from yunshu_engine import settings

        if settings.get_str("YUNSHU_VLM_DRAFT") != "mtp" or settings.get_bool(
            "YUNSHU_VLM_APC_DISK"
        ):
            raise RuntimeError("27B gate requires explicit MTP and RAM-only APC")
    engine = VLMEngine(a.model)
    asyncio.run(engine.start())
    runner = engine._batch_runner
    from mlx_vlm.models.qwen3_5.language import Qwen3_5Model
    from native_model_cache import wrap as native_wrap

    model_forward = Qwen3_5Model.__call__
    native_forward = native_wrap(model_forward)
    piece, maximum = lane_linear.PIECE, lane_qmm.MAX_ROWS
    tokenizer = getattr(runner.processor, "tokenizer", runner.processor)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    candidate = a.variant
    correct = {"baseline": 0, candidate: 0}
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
                results = {}
                for mode in (
                    ("baseline", candidate)
                    if index % 2 == 0
                    else (candidate, "baseline")
                ):
                    Qwen3_5Model.__call__ = (
                        native_forward if mode == "native" else model_forward
                    )
                    lane_linear.PIECE = 512 if mode == "wide" else piece
                    lane_qmm.MAX_ROWS = 512 if mode == "wide" else maximum
                    runner.apc_manager.clear()
                    native_before = native_forward.native_calls
                    stats, tokens, lps = RunStats(), [], []
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
                        raise RuntimeError("incomplete / non-cold paired item")
                    native_calls = native_forward.native_calls - native_before
                    if mode == "native" and not native_calls:
                        raise RuntimeError("native singleton path did not engage")
                    text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    match = re.fullmatch(r"\s*(-?\d+)\s*", text)
                    scored = bool(match and int(match[1]) == answer)
                    correct[mode] += scored
                    results[mode] = dict(
                        tokens=tokens, lps=lps, text=text, correct=scored
                    )
                equal = (
                    results["baseline"]["tokens"] == results[candidate]["tokens"]
                    and results["baseline"]["lps"] == results[candidate]["lps"]
                )
                different += not equal
                row = dict(
                    index=index,
                    prompt_tokens=len(ids),
                    answer=answer,
                    equal=equal,
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    arms=results,
                )
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(
                    json.dumps({k: v for k, v in row.items() if k != "arms"}),
                    flush=True,
                )
            success = (
                different == 0 and abs(correct["baseline"] - correct[candidate]) <= 1
            )
            out.write(
                json.dumps(
                    dict(
                        phase="complete",
                        success=success,
                        items=a.items,
                        variant=candidate,
                        correct=correct,
                        different=different,
                    )
                )
                + "\n"
            )
            if not success:
                raise RuntimeError("paired output / accuracy gate failed")
    finally:
        Qwen3_5Model.__call__ = model_forward
        lane_linear.PIECE, lane_qmm.MAX_ROWS = piece, maximum
        asyncio.run(engine.stop())


if __name__ == "__main__":
    main()
