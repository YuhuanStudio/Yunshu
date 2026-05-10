#!/usr/bin/env python3
"""MMLU-Pro cross-framework comparison: mlx-lm vs vllm-mlx vs omlx.
42 questions (14 categories × 3), max_tokens=4096, temp=0.0.
"""
import asyncio
import gc
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/yuhuan/Documents/Yunshu")
REF_DIR = ROOT.parent / "reference"
MODEL_PATH = str(ROOT / "models" / "Qwen3.5-9B-MLX-bf16")
LOG = ROOT / "bench" / "results" / "framework_compare.log"

LETTERS = "ABCDEFGHIJ"
VALID = set(LETTERS)
ANSWER_RE = re.compile(r"answer is \(?([A-J])\)?", re.IGNORECASE)

# Add paths upfront
sys.path.insert(0, str(REF_DIR / "mlx-lm"))


def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
        f.flush()
    print(msg, flush=True)


def extract_answer(text):
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).upper()
    cleaned = re.sub(r"<think[^>]*>.*?</think[^>]*>", "", text, flags=re.DOTALL)
    m = ANSWER_RE.search(cleaned)
    if m:
        return m.group(1).upper()
    ls = re.findall(r"\b([A-J])\b", cleaned)
    return ls[-1] if ls else None


def build_mmlu_messages(fewshot_rows, test_q):
    msgs = [{"role": "system", "content":
        "You are an expert at answering multiple choice questions. "
        "Think step by step, then answer with 'answer is (X)' where X is the letter."}]
    for row in fewshot_rows:
        user = "Question:\n" + row["question"] + "\nOptions:\n"
        for j, opt in enumerate(row["options"]):
            if j >= 10:
                break
            user += f"{LETTERS[j]}. {opt.strip()}\n"
        cot = row["cot_content"].replace(
            "A: Let's think step by step.", "Answer: Let's think step by step."
        )
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": cot})
    test_user = "Question:\n" + test_q["question"] + "\nOptions:\n"
    for j, opt in enumerate(test_q["options"]):
        if j >= 10:
            break
        test_user += f"{LETTERS[j]}. {opt.strip()}\n"
    msgs.append({"role": "user", "content": test_user})
    return msgs


def load_data():
    from datasets import load_dataset
    val_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
    test_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    fewshot_by_cat = defaultdict(list)
    for row in val_ds:
        fewshot_by_cat[row["category"]].append(row)
    fewshot_by_cat = {c: rows[:5] for c, rows in fewshot_by_cat.items()}
    rng = random.Random(42)
    by_cat = defaultdict(list)
    for q in test_ds:
        by_cat[q["category"]].append(q)
    questions = []
    for cat, qs in by_cat.items():
        rng.shuffle(qs)
        questions.extend(qs[:3])
    rng.shuffle(questions)
    return questions, fewshot_by_cat


def cleanup():
    gc.collect()
    try:
        import mlx.core as mx
        mx.synchronize()
        mx.clear_cache()
    except Exception:
        pass


def apply_template(tokenizer, msgs):
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )


# ─── MLX-LM (baseline, sync) ───

def run_mlxlm(questions, fewshot_by_cat):
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    log("  Loading model...")
    model, tokenizer = load(MODEL_PATH)
    sampler = make_sampler(temp=0.0)
    log("  Model loaded")

    # Warmup
    warmup_prompt = apply_template(tokenizer, build_mmlu_messages(
        fewshot_by_cat.get(questions[0]["category"], [])[:5], questions[0]))
    for _ in stream_generate(model, tokenizer, warmup_prompt, max_tokens=32, sampler=sampler):
        pass
    log("  Warmup done")

    correct = 0
    total = 0
    times = []

    for i, q in enumerate(questions):
        cat = q["category"]
        msgs = build_mmlu_messages(fewshot_by_cat.get(cat, [])[:5], q)
        prompt = apply_template(tokenizer, msgs)
        t0 = time.perf_counter()
        text = ""
        n_tok = 0
        for resp in stream_generate(model, tokenizer, prompt, max_tokens=4096, sampler=sampler):
            text += resp.text
            n_tok += 1
            if ANSWER_RE.search(text):
                break
            if n_tok >= 4096:
                break
        dt = time.perf_counter() - t0
        times.append(dt)

        predicted = extract_answer(text)
        answer = q["answer"]
        ok = predicted == answer and predicted in VALID
        if ok:
            correct += 1
        total += 1
        log(f"  [{i+1}/{len(questions)}] {cat}: pred={predicted} ans={answer} "
            f"{'✓' if ok else '✗'} | {dt:.1f}s {n_tok}tok")

    acc = correct / total * 100 if total else 0
    se = math.sqrt(acc * (100 - acc) / total) if total else 0
    avg = sum(times) / len(times) if times else 0
    log(f"  RESULT: {correct}/{total} = {acc:.1f}% ±{se:.1f}% [avg {avg:.1f}s/q]")

    del model, tokenizer
    cleanup()
    return {"framework": "mlx-lm", "correct": correct, "total": total,
            "accuracy": acc, "stderr": se, "avg_time": avg}


# ─── VLLM-MLX ───

async def run_vllm_mlx(questions, fewshot_by_cat):
    sys.path.insert(0, str(REF_DIR / "vllm-mlx"))
    from vllm_mlx.engine.simple import SimpleEngine

    log("  Loading model...")
    engine = SimpleEngine(model_name=MODEL_PATH)
    # Qwen3.5- matches MLLM_PATTERNS in vllm-mlx, but Qwen3.5-9B is text-only
    engine._is_mllm = False
    await engine.start()
    tokenizer = engine.tokenizer
    log(f"  Model loaded. tokenizer={type(tokenizer).__name__}")

    # Warmup
    warmup_prompt = apply_template(tokenizer, build_mmlu_messages(
        fewshot_by_cat.get(questions[0]["category"], [])[:5], questions[0]))
    await engine.generate(warmup_prompt, max_tokens=32, temperature=0.0)
    log("  Warmup done")

    correct = 0
    total = 0
    times = []

    for i, q in enumerate(questions):
        cat = q["category"]
        msgs = build_mmlu_messages(fewshot_by_cat.get(cat, [])[:5], q)
        prompt = apply_template(tokenizer, msgs)
        t0 = time.perf_counter()
        result = await engine.generate(
            prompt, max_tokens=4096, temperature=0.0, stop=["Question:"])
        dt = time.perf_counter() - t0
        times.append(dt)

        text = result.text if hasattr(result, "text") else str(result)
        predicted = extract_answer(text)
        answer = q["answer"]
        ok = predicted == answer and predicted in VALID
        if ok:
            correct += 1
        total += 1
        n_tok = getattr(result, "completion_tokens", "?")
        log(f"  [{i+1}/{len(questions)}] {cat}: pred={predicted} ans={answer} "
            f"{'✓' if ok else '✗'} | {dt:.1f}s {n_tok}tok")

    acc = correct / total * 100 if total else 0
    se = math.sqrt(acc * (100 - acc) / total) if total else 0
    avg = sum(times) / len(times) if times else 0
    log(f"  RESULT: {correct}/{total} = {acc:.1f}% ±{se:.1f}% [avg {avg:.1f}s/q]")

    await engine.stop()
    cleanup()
    return {"framework": "vllm-mlx", "correct": correct, "total": total,
            "accuracy": acc, "stderr": se, "avg_time": avg}


# ─── OMLX ───

async def run_omlx(questions, fewshot_by_cat):
    sys.path.insert(0, str(REF_DIR / "omlx"))
    from omlx.engine import BatchedEngine

    log("  Loading model...")
    engine = BatchedEngine(model_name=MODEL_PATH)
    await engine.start()
    tokenizer = engine.tokenizer
    log(f"  Model loaded. tokenizer={type(tokenizer).__name__}")

    # Warmup
    warmup_prompt = apply_template(tokenizer, build_mmlu_messages(
        fewshot_by_cat.get(questions[0]["category"], [])[:5], questions[0]))
    await engine.generate(warmup_prompt, max_tokens=32, temperature=0.0)
    log("  Warmup done")

    correct = 0
    total = 0
    times = []

    for i, q in enumerate(questions):
        cat = q["category"]
        msgs = build_mmlu_messages(fewshot_by_cat.get(cat, [])[:5], q)
        prompt = apply_template(tokenizer, msgs)
        t0 = time.perf_counter()
        result = await engine.generate(
            prompt, max_tokens=4096, temperature=0.0, stop=["Question:"])
        dt = time.perf_counter() - t0
        times.append(dt)

        text = result.text if hasattr(result, "text") else str(result)
        predicted = extract_answer(text)
        answer = q["answer"]
        ok = predicted == answer and predicted in VALID
        if ok:
            correct += 1
        total += 1
        n_tok = getattr(result, "completion_tokens", "?")
        log(f"  [{i+1}/{len(questions)}] {cat}: pred={predicted} ans={answer} "
            f"{'✓' if ok else '✗'} | {dt:.1f}s {n_tok}tok")

    acc = correct / total * 100 if total else 0
    se = math.sqrt(acc * (100 - acc) / total) if total else 0
    avg = sum(times) / len(times) if times else 0
    log(f"  RESULT: {correct}/{total} = {acc:.1f}% ±{se:.1f}% [avg {avg:.1f}s/q]")

    await engine.stop()
    cleanup()
    return {"framework": "omlx", "correct": correct, "total": total,
            "accuracy": acc, "stderr": se, "avg_time": avg}


async def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("")

    log("=" * 70)
    log("  MMLU-Pro Cross-Framework Comparison")
    log("  Model: Qwen3.5-9B bf16 | max_tokens=4096 | temp=0.0")
    log("=" * 70)

    log("\nLoading datasets...")
    questions, fewshot_by_cat = load_data()
    log(f"  {len(questions)} questions (14 categories × 3)\n")

    results = []

    # ── Framework 1: mlx-lm (sync) ──
    log("─" * 70)
    log("  Framework: MLX-LM (stream_generate)")
    log("─" * 70)
    results.append(run_mlxlm(questions, fewshot_by_cat))
    log("")

    # ── Framework 2: vllm-mlx ──
    log("─" * 70)
    log("  Framework: vllm-mlx (SimpleEngine)")
    log("─" * 70)
    results.append(await run_vllm_mlx(questions, fewshot_by_cat))
    log("")

    # ── Framework 3: omlx ──
    log("─" * 70)
    log("  Framework: omlx (BatchedEngine)")
    log("─" * 70)
    results.append(await run_omlx(questions, fewshot_by_cat))
    log("")

    # ── Summary ──
    log("=" * 70)
    log("  SUMMARY")
    log("=" * 70)
    log(f"  {'Framework':<12} {'Accuracy':>10} {'Correct':>10} {'Avg s/q':>10} {'vs mlx-lm':>10}")
    log(f"  {'─' * 12} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 10}")
    base_acc = results[0]["accuracy"] if results else 0
    for r in results:
        diff = r["accuracy"] - base_acc if r["framework"] != "mlx-lm" else 0
        diff_str = "BASELINE" if r["framework"] == "mlx-lm" else f"{diff:+.1f}%"
        log(f"  {r['framework']:<12} {r['accuracy']:>9.1f}% {r['correct']:>5}/{r['total']:<4} "
            f"{r['avg_time']:>9.1f}s {diff_str:>10}")
    log("\nDONE")


if __name__ == "__main__":
    asyncio.run(main())
