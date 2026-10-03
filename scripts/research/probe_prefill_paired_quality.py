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
    ap.add_argument(
        "--variant",
        choices=(
            "wide",
            "native",
            "async",
            "barrier",
            "paired32",
            "cow",
            "cowasync",
            "spans",
        ),
        default="wide",
    )
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
    async_counts = {}
    uninstall_async = None
    span_counts = {}
    uninstall_spans = None
    if a.variant == "spans":
        from span_forward import install as install_spans

        span_counts, uninstall_spans = install_spans()
    cow_counts = {}
    uninstall_cow = None
    if a.variant in ("cow", "cowasync"):
        from cow_restore import install as install_cow

        cow_counts, uninstall_cow = install_cow()
    if a.variant in ("async", "cowasync"):
        from async_restore import install

        async_counts, uninstall_async = install()
    kernel_install = None
    kernel_uninstall = None
    if a.variant == "paired32":
        from lane_paired32 import install as kernel_install
    elif a.variant == "barrier":
        from lane_final_barrier import install as kernel_install
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
                prime_ids = ids
                if candidate in (
                    "async",
                    "barrier",
                    "paired32",
                    "cow",
                    "cowasync",
                    "spans",
                ):
                    if not runner.apc_manager.prefill_stride:
                        prime_ids = ids[:-65]
                    else:
                        prime_ids = ids
                        ids, _, salt = engine._executor.submit(
                            engine._runner_input,
                            [
                                {"role": "user", "content": prompt},
                                {
                                    "role": "assistant",
                                    "content": "The previous question is noted. " * 24,
                                },
                                {"role": "user", "content": prompt.rsplit("\n", 1)[-1]},
                            ],
                            [],
                            [],
                            False,
                            {"enable_thinking": False},
                        ).result()
                        if (
                            ids[: len(prime_ids) - 1] != prime_ids[:-1]
                            or len(ids) - len(prime_ids) < 64
                        ):
                            raise RuntimeError(
                                "quality chat does not share a canonical long-suffix boundary"
                            )
                results = {}
                for mode in (
                    ("baseline", candidate)
                    if index % 2 == 0
                    else (candidate, "baseline")
                ):
                    if span_counts:
                        span_counts["enabled"] = mode == "spans"
                    if cow_counts:
                        cow_counts["enabled"] = mode in ("cow", "cowasync")
                    if kernel_uninstall:
                        kernel_uninstall()
                        kernel_uninstall = None
                    if kernel_install and mode == candidate:
                        kernel_uninstall = kernel_install()
                    if async_counts:
                        async_counts["enabled"] = mode in ("async", "cowasync")
                    Qwen3_5Model.__call__ = (
                        native_forward if mode == "native" else model_forward
                    )
                    lane_linear.PIECE = 512 if mode == "wide" else piece
                    lane_qmm.MAX_ROWS = 512 if mode == "wide" else maximum
                    runner.apc_manager.clear()
                    if candidate in (
                        "native",
                        "async",
                        "barrier",
                        "paired32",
                        "cow",
                        "cowasync",
                        "spans",
                    ):
                        prime_stats = RunStats()
                        primed = list(
                            runner.iter_tokens(
                                prime_ids,
                                max_tokens=1,
                                temperature=0,
                                seed=1234,
                                apc_semantic_hash=salt,
                                stats=prime_stats,
                                logprobs=True,
                            )
                        )
                        engine._executor.submit(lambda: None).result()
                        if (
                            not primed
                            or not prime_stats.finish_reason
                            or prime_stats.cached_tokens
                        ):
                            raise RuntimeError(
                                "paired warm prime incomplete / not cold"
                            )
                    span_before = span_counts.get("forwards", 0)
                    cow_before = cow_counts.get("view_restores", 0)
                    async_before = async_counts.get("async_merges", 0)
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
                    expected_cached = (
                        len(prime_ids) - 1
                        if candidate
                        in ("async", "barrier", "paired32", "cow", "cowasync", "spans")
                        else len(ids) - 1
                        if candidate == "native"
                        else 0
                    )
                    if (
                        not tokens
                        or not stats.finish_reason
                        or stats.cached_tokens != expected_cached
                    ):
                        raise RuntimeError(
                            "incomplete / unexpected paired cache boundary"
                        )
                    if "27B" in a.model and (
                        not stats.used_draft or stats.spec_mode != "mtp"
                    ):
                        raise RuntimeError("paired quality did not engage MTP")
                    native_calls = native_forward.native_calls - native_before
                    if mode == "native" and not native_calls:
                        raise RuntimeError("native singleton path did not engage")
                    if mode == "spans" and span_counts["forwards"] <= span_before:
                        raise RuntimeError("joint span forward did not engage")
                    if (
                        mode in ("cow", "cowasync")
                        and cow_counts["view_restores"] <= cow_before
                    ):
                        raise RuntimeError("COW restore did not engage")
                    if (
                        mode in ("async", "cowasync")
                        and async_counts["async_merges"] <= async_before
                    ):
                        raise RuntimeError("async suffix restore did not engage")
                    text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    match = re.fullmatch(r"\s*(-?\d+)\s*", text)
                    scored = bool(match and int(match[1]) == answer)
                    correct[mode] += scored
                    results[mode] = dict(
                        tokens=tokens,
                        lps=lps,
                        text=text,
                        correct=scored,
                        cached=stats.cached_tokens,
                        native_calls=native_calls,
                        used_draft=stats.used_draft,
                        spec_mode=stats.spec_mode,
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
        if kernel_uninstall:
            kernel_uninstall()
        if uninstall_spans:
            uninstall_spans()
        if uninstall_cow:
            uninstall_cow()
        if uninstall_async:
            uninstall_async()
        asyncio.run(engine.stop())


if __name__ == "__main__":
    main()
