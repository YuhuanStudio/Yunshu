#!/usr/bin/env python3
"""Quick plain text baseline: 14 questions (1 per category)."""
import math
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# Wave 687: was hardcoded /Users/yuhuan/Documents/Yunshu (missing YuhuanStudio/),
# so the model path didn't exist and mlx_lm tried to fetch it as an HF repo id →
# HFValidationError → the MMLU gate could never pass. Derive from the script path.
ROOT = Path(__file__).resolve().parent.parent.parent
LOG = ROOT / "bench" / "results" / "diag_plain.log"
sys.path.insert(0, str(ROOT / "reference" / "mlx-lm"))

from datasets import load_dataset
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

LETTERS = "ABCDEFGHIJ"
VALID = set(LETTERS)

def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
        f.flush()
    print(msg, flush=True)

def extract(text):
    m = re.search(r"answer is \(?([A-J])\)?", text, re.IGNORECASE)
    if m: return m.group(1).upper()
    cleaned = re.sub(r"<think[^>]*>.*?</think[^>]*>", "", text, flags=re.DOTALL)
    m = re.search(r"answer is \(?([A-J])\)?", cleaned, re.IGNORECASE)
    if m: return m.group(1).upper()
    ls = re.findall(r"\b([A-J])\b", cleaned)
    return ls[-1] if ls else None

def main():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("")

    log("═" * 60)
    log("  PLAIN TEXT BASELINE (no chat_template)")
    log("═" * 60)

    val_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
    test_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")

    # Build few-shot strings
    by_cat = defaultdict(list)
    for row in val_ds:
        by_cat[row["category"]].append(row)
    fewshots = {}
    for cat, rows in by_cat.items():
        s = ""
        for row in rows[:5]:
            s += "Question:\n" + row["question"] + "\nOptions:\n"
            for j, opt in enumerate(row["options"]):
                if j >= 10: break
                s += f"{LETTERS[j]}. {opt.strip()}\n"
            cot = row["cot_content"].replace(
                "A: Let's think step by step.", "Answer: Let's think step by step."
            )
            s += cot + "\n\n"
        fewshots[cat] = s

    # Sample 14 questions (1 per category)
    rng = random.Random(42)
    by_cat2 = defaultdict(list)
    for q in test_ds:
        by_cat2[q["category"]].append(q)
    questions = []
    for cat, qs in by_cat2.items():
        rng.shuffle(qs)
        questions.append(qs[0])
    rng.shuffle(questions)
    log(f"  {len(questions)} questions")

    model, tokenizer = load(str(ROOT / "models" / "Qwen3.5-9B-MLX-bf16"))
    sampler = make_sampler(temp=0.0)
    log("  Model loaded")

    correct = 0
    total = 0
    for i, q in enumerate(questions):
        cat = q["category"]
        prompt = fewshots.get(cat, "") + "Question:\n" + q["question"] + "\nOptions:\n"
        for j, opt in enumerate(q["options"]):
            if j >= 10: break
            prompt += f"{LETTERS[j]}. {opt.strip()}\n"
        prompt += "Answer: Let's think step by step."

        t0 = time.perf_counter()
        text = ""
        n_tok = 0
        for resp in stream_generate(model, tokenizer, prompt, max_tokens=2048, sampler=sampler):
            text += resp.text
            n_tok += 1
            if re.search(r"answer is \(?([A-J])\)?", text, re.IGNORECASE):
                break
            if n_tok >= 2048: break
        dt = time.perf_counter() - t0

        predicted = extract(text)
        answer = q["answer"]
        ok = predicted == answer and predicted in VALID
        if ok: correct += 1
        total += 1
        log(f"  {i+1}/{len(questions)} {cat}: pred={predicted} ans={answer} "
            f"{'✓' if ok else '✗'} | {dt:.1f}s {n_tok}tok")

    acc = correct/total*100 if total else 0
    se = math.sqrt(acc*(100-acc)/total) if total else 0
    log(f"\n  RESULT: {correct}/{total} = {acc:.1f}% ±{se:.1f}%")
    log("DONE")

if __name__ == "__main__":
    main()
