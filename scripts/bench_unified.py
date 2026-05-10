"""Yunshu Comprehensive Benchmark — lm-evaluation-harness compatible.

MMLU-Pro evaluation follows EleutherAI lm-evaluation-harness exactly:
  - 5-shot CoT from validation split (cot_content)
  - generate_until with max_tokens=4096, temperature=0.0
  - Official regex extraction: `answer is \(?([A-J])\)?`
  - Per-category accuracy, weighted by size

Supports batched inference via mlx-lm BatchGenerator for higher GPU utilization.

Additional benchmarks (HellaSwag, GSM8K, TruthfulQA) and non-LLM modalities
use best-effort evaluation.

Usage:
    PYTHONPATH=. uv run python scripts/bench_unified.py
    PYTHONPATH=. uv run python scripts/bench_unified.py --quick
    PYTHONPATH=. uv run python scripts/bench_unified.py --full
    PYTHONPATH=. uv run python scripts/bench_unified.py --bench mmlu_pro --samples 200
    PYTHONPATH=. uv run python scripts/bench_unified.py --framework mlx-lm yunshu --batch-size 8
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import io
import json
import math
import os
import re
import struct
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
REF_DIR = ROOT / "reference"
MODELS_DIR = ROOT / "models"

MODELS = {
    "llm": "Qwen3.5-9B-MLX-bf16",
    "vlm": "Qwen3-Omni-30B-A3B-Instruct-4bit",
    "tts": "Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
    "asr": "Qwen3-ASR-1.7B-bf16",
    "image": "Z-Image-Turbo-MLX-4bit",
}

VALID_ANSWERS = set("ABCDEFGHIJ")
LETTERS = list("ABCDEFGHIJ")

CACHE_DIR = ROOT / "bench" / "results" / "cache"


def _ensure_ref_path(name: str):
    p = str(REF_DIR / name)
    if p not in sys.path:
        sys.path.insert(0, p)


def _cache_key(framework: str, benchmark: str, n_samples: int, max_tokens: int) -> Path:
    return CACHE_DIR / f"{framework}_{benchmark}_n{n_samples}_mt{max_tokens}.json"


def load_cached(framework: str, benchmark: str, n_samples: int, max_tokens: int) -> BenchResult | None:
    p = _cache_key(framework, benchmark, n_samples, max_tokens)
    if p.exists():
        data = json.loads(p.read_text())
        return BenchResult(**data)
    return None


def save_cached(result: BenchResult, n_samples: int, max_tokens: int):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _cache_key(result.framework, result.benchmark, n_samples, max_tokens)
    data = {k: v for k, v in result.__dict__.items() if v is not None and v != {} and v != []}
    p.write_text(json.dumps(data, indent=2, default=str))


def P(msg: str):
    print(msg, flush=True)


def model_path(modality: str) -> str:
    name = MODELS.get(modality, "")
    p = MODELS_DIR / name
    return str(p) if p.exists() else name


def model_exists(modality: str) -> bool:
    return (MODELS_DIR / MODELS.get(modality, "")).exists()


def cleanup():
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    gc.collect()
    time.sleep(1.0)


def get_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1048576
    except Exception:
        return 0.0


@dataclass
class BenchResult:
    framework: str
    benchmark: str
    accuracy: float | None = None
    accuracy_stderr: float | None = None
    throughput: float | None = None
    throughput_std: float | None = None
    ttft_ms: float | None = None
    ttft_std: float | None = None
    memory_mb: float | None = None
    latency_ms: float | None = None
    extra: dict = field(default_factory=dict)
    per_category: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════
# MMLU-PRO: lm-evaluation-harness compatible
# ══════════════════════════════════════════════════════════════════

def build_mmlu_fewshots_by_category(val_ds, n_shot: int = 5) -> dict[str, str]:
    """Build per-category few-shot prompts from validation split (lm-eval-harness compatible)."""
    by_cat: dict[str, list] = defaultdict(list)
    for row in val_ds:
        by_cat[row["category"]].append(row)

    fewshots = {}
    for cat, rows in by_cat.items():
        prompt = ""
        for row in rows[:n_shot]:
            prompt += "Question:\n" + row["question"] + "\nOptions:\n"
            for j, opt in enumerate(row["options"]):
                if j >= len(LETTERS):
                    break
                prompt += f"{LETTERS[j]}. {opt.strip()}\n"
            cot = row["cot_content"].replace(
                "A: Let's think step by step.", "Answer: Let's think step by step."
            )
            prompt += cot + "\n\n"
        fewshots[cat] = prompt
    return fewshots


def get_fewshot_rows_by_category(val_ds, n_shot: int = 5) -> dict[str, list]:
    """Get few-shot row dicts grouped by category for chat-template mode."""
    by_cat: dict[str, list] = defaultdict(list)
    for row in val_ds:
        by_cat[row["category"]].append(row)
    return {cat: rows[:n_shot] for cat, rows in by_cat.items()}


def build_mmlu_messages(fewshot_rows: list[dict], test_row: dict) -> list[dict]:
    """Build chat messages: system + few-shot as user/assistant turns + test question."""
    messages = [{"role": "system", "content":
        "You are an expert at answering multiple choice questions. "
        "Think step by step, then answer with 'answer is (X)' where X is the letter."}]
    for row in fewshot_rows:
        user = "Question:\n" + row["question"] + "\nOptions:\n"
        for j, opt in enumerate(row["options"]):
            if j >= len(LETTERS):
                break
            user += f"{LETTERS[j]}. {opt.strip()}\n"
        cot = row["cot_content"].replace(
            "A: Let's think step by step.", "Answer: Let's think step by step."
        )
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": cot})
    test_user = "Question:\n" + test_row["question"] + "\nOptions:\n"
    for j, opt in enumerate(test_row["options"]):
        if j >= len(LETTERS):
            break
        test_user += f"{LETTERS[j]}. {opt.strip()}\n"
    messages.append({"role": "user", "content": test_user})
    return messages


def build_mmlu_test_prompt(fewshot: str, row: dict) -> str:
    prompt = fewshot + "Question:\n" + row["question"] + "\nOptions:\n"
    for j, opt in enumerate(row["options"]):
        if j >= len(LETTERS):
            break
        prompt += f"{LETTERS[j]}. {opt.strip()}\n"
    prompt += "Answer: Let's think step by step."
    return prompt


def extract_mmlu_pro_answer(text: str) -> str:
    r"""lm-eval-harness official extraction: answer is \(?([A-J])\)?"""
    m = re.search(r"answer is \(?([A-J])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1)
    # Strip thinking tags
    text = re.sub(r"<think[^>]*>.*?</think[^>]*>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think[^>]*>.*", "", text, flags=re.DOTALL)
    text = text.strip()
    # Fallback: last standalone letter in tail
    tail = text[-300:]
    last = None
    for m2 in re.finditer(r"\b([A-J])\b", tail):
        last = m2.group(1)
    return last or "?"


# Pre-compiled regex for early stopping
_ANSWER_RE = re.compile(r"answer is \(?([A-J])\)?", re.IGNORECASE)
_STOP_PATTERNS = ["Question:"]


# ══════════════════════════════════════════════════════════════════
# OTHER DATASETS
# ══════════════════════════════════════════════════════════════════

def extract_letter(text: str, max_option: int = 10) -> str:
    text = re.sub(r"<think[^>]*>.*?</think[^>]*>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think[^>]*>.*", "", text, flags=re.DOTALL)
    text = text.strip()
    m = re.search(r"(?:correct\s+)?answer\s*(?:is|:)\s*([A-J])", text, re.IGNORECASE)
    if m and m.group(1) <= chr(64 + max_option):
        return m.group(1)
    tail = text[-300:]
    last = None
    for m2 in re.finditer(r"\b([A-J])\b", tail):
        if m2.group(1) <= chr(64 + max_option):
            last = m2.group(1)
    if last:
        return last
    m = re.search(r"\b([A-J])\b", text)
    if m and m.group(1) <= chr(64 + max_option):
        return m.group(1)
    return "?"


def extract_number(text: str) -> float | None:
    text = re.sub(r"<think[^>]*>.*?</think[^>]*>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think[^>]*>.*", "", text, flags=re.DOTALL)
    text = text.strip()
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except ValueError:
            pass
    matches = re.findall(r"-?\d+\.?\d*", text)
    if matches:
        try:
            return float(matches[-1].replace(",", ""))
        except ValueError:
            pass
    return None


# ══════════════════════════════════════════════════════════════════
# MMLU-PRO RUNNER — thinking with logit bias + force-close fallback
# ══════════════════════════════════════════════════════════════════



def _generate_with_thinking(model, tokenizer, prompt: str, max_tokens: int, sampler, _think_budget: int | None = None) -> tuple[str, int]:
    """Two-phase generation: thinking then answer, with force-close for 4-bit."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler as _make_sampler

    THINK_BUDGET = _think_budget if _think_budget is not None else 1024

    text = ""
    n_tok = 0
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=THINK_BUDGET, sampler=sampler):
        text += resp.text
        n_tok += 1
        if "</think" in text:
            think_end = text.rfind("</think")
            post = text[think_end:]
            if _ANSWER_RE.search(post) or "Question:" in post:
                return text, n_tok
            if len(post) > 200:
                return text, n_tok

    # Force-close: model hit thinking budget without closing
    if n_tok > 50 and "</think" not in text:
        close_tag = "\n</think\n\n"
        # Use last 2000 chars of thinking as context to keep prompt manageable
        think_tail = text[-2000:] if len(text) > 2000 else text
        new_prompt = prompt + think_tail + close_tag
        clean_sampler = _make_sampler(temp=0.0)
        post_text = ""
        post_n = 0
        for resp in stream_generate(model, tokenizer, new_prompt, max_tokens=256, sampler=clean_sampler):
            post_text += resp.text
            post_n += 1
            if _ANSWER_RE.search(post_text) or "Question:" in post_text:
                break
            if post_n >= 200:
                break
        return text + close_tag + post_text, n_tok + post_n

    return text, n_tok

def run_mmlu_pro_mlxlm(
    model,
    tokenizer,
    test_ds,
    fewshot_rows_by_cat: dict[str, list],
    n_samples: int,
    max_tokens: int = 4096,
) -> BenchResult:
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)
    use_thinking = getattr(tokenizer, "has_thinking", False)

    questions = list(test_ds)
    if 0 < n_samples < len(questions):
        rng = __import__("random").Random(42)
        by_cat = defaultdict(list)
        for q in questions:
            by_cat[q["category"]].append(q)
        per_cat = max(1, n_samples // len(by_cat))
        sampled = []
        for cat, qs in by_cat.items():
            rng.shuffle(qs)
            sampled.extend(qs[:per_cat])
        rng.shuffle(sampled)
        questions = sampled[:n_samples]

    P(f"    MMLU-Pro: {len(questions)} questions, chat_template, max_tokens={max_tokens}")

    from mlx_lm import stream_generate

    # Warmup with chat template
    warmup_msgs = build_mmlu_messages(
        fewshot_rows_by_cat.get(questions[0]["category"], [])[:5], questions[0]
    )
    warmup_prompt = tokenizer.apply_chat_template(
        warmup_msgs, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    for _ in stream_generate(model, tokenizer, warmup_prompt, max_tokens=32, sampler=sampler):
        pass
    P(f"    Warmup done")

    cat_correct: dict[str, int] = Counter()
    cat_total: dict[str, int] = Counter()
    correct = 0
    total = 0
    times = []

    t_start = time.perf_counter()
    for i, q in enumerate(questions):
        cat = q["category"]
        msgs = build_mmlu_messages(
            fewshot_rows_by_cat.get(cat, [])[:5], q
        )
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        t0 = time.perf_counter()
        text = ""
        n_tok = 0
        for resp in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens, sampler=sampler):
            text += resp.text
            n_tok += 1
            if _ANSWER_RE.search(text):
                break
            if "Question:" in text:
                break
        dt = time.perf_counter() - t0
        times.append(dt)

        predicted = extract_mmlu_pro_answer(text)
        answer = q["answer"]
        is_correct = predicted == answer and predicted in VALID_ANSWERS
        if is_correct:
            correct += 1
        total += 1
        if is_correct:
            cat_correct[cat] += 1
        cat_total[cat] += 1

        P(f"    [{i+1}/{len(questions)}] {cat}: pred={predicted} ans={answer} {'✓' if is_correct else '✗'} "
          f"| {dt:.1f}s {n_tok}tok")

        if (i + 1) % 5 == 0 or (i + 1) == len(questions):
            avg = sum(times) / len(times)
            P(f"    ── {i+1}/{len(questions)} — {correct}/{total} = {correct/total*100:.1f}% "
              f"avg {avg:.1f}s/q")

    elapsed = time.perf_counter() - t_start
    accuracy = correct / total * 100 if total else 0
    stderr = math.sqrt(accuracy * (100 - accuracy) / total) if total > 0 else 0
    per_cat = {cat: {"correct": cat_correct.get(cat, 0), "total": cat_total[cat],
                      "accuracy": cat_correct.get(cat, 0) / cat_total[cat] * 100 if cat_total[cat] else 0}
               for cat in cat_total}

    return BenchResult(
        framework="mlx-lm",
        benchmark="mmlu_pro",
        accuracy=accuracy,
        accuracy_stderr=stderr,
        extra={
            "correct": correct, "total": total,
            "avg_time_per_q": round(sum(times) / len(times), 2),
            "total_time": round(elapsed, 1),
            "max_tokens": max_tokens,
            "method": "5-shot CoT + chat_template + non-thinking",
            "thinking_enabled": False,
        },
        per_category=per_cat,
    )


# ══════════════════════════════════════════════════════════════════
# YUNSHU BATCHED RUNNER
# ══════════════════════════════════════════════════════════════════

async def run_mmlu_pro_yunshu(
    test_ds,
    fewshot_rows_by_cat: dict[str, list],
    n_samples: int,
    max_tokens: int = 4096,
) -> BenchResult:
    from yunshu_engine.batched_engine import BatchedEngine

    engine = BatchedEngine(model_name=model_path("llm"))
    await engine.start()
    tokenizer = engine._tokenizer

    questions = list(test_ds)
    if 0 < n_samples < len(questions):
        rng = __import__("random").Random(42)
        by_cat = defaultdict(list)
        for q in questions:
            by_cat[q["category"]].append(q)
        per_cat = max(1, n_samples // len(by_cat))
        sampled = []
        for cat, qs in by_cat.items():
            rng.shuffle(qs)
            sampled.extend(qs[:per_cat])
        rng.shuffle(sampled)
        questions = sampled[:n_samples]

    P(f"    MMLU-Pro (Yunshu): {len(questions)} questions, chat_template + generate_until")

    warmup_msgs = build_mmlu_messages(
        fewshot_rows_by_cat.get(questions[0]["category"], [])[:5],
        questions[0],
    )
    warmup_prompt = tokenizer.apply_chat_template(
        warmup_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    await engine.generate(prompt=warmup_prompt, max_tokens=64, temperature=0.0)
    P(f"    Warmup done")

    cat_correct: dict[str, int] = Counter()
    cat_total: dict[str, int] = Counter()
    correct = 0
    total = 0
    times = []

    t_start = time.perf_counter()
    for i, q in enumerate(questions):
        cat = q["category"]
        msgs = build_mmlu_messages(
            fewshot_rows_by_cat.get(cat, [])[:5],
            q,
        )
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        t0 = time.perf_counter()
        r = await engine.generate(prompt=prompt, max_tokens=max_tokens, temperature=0.0)
        dt = time.perf_counter() - t0
        times.append(dt)

        text = r.text if hasattr(r, "text") else str(r)
        predicted = extract_mmlu_pro_answer(text)
        answer = q["answer"]
        is_correct = predicted == answer and predicted in VALID_ANSWERS
        if is_correct:
            correct += 1
        total += 1
        cat = q["category"]
        if is_correct:
            cat_correct[cat] += 1
        cat_total[cat] += 1

        if (i + 1) % 5 == 0 or (i + 1) == len(questions):
            avg = sum(times) / len(times)
            P(f"    {i+1}/{len(questions)} — {correct}/{total} = {correct/total*100:.1f}% "
              f"| {dt:.1f}s avg {avg:.1f}s/q")

    await engine.stop()

    elapsed = time.perf_counter() - t_start
    accuracy = correct / total * 100 if total else 0
    stderr = math.sqrt(accuracy * (100 - accuracy) / total) if total > 0 else 0
    per_cat = {cat: {"correct": cat_correct.get(cat, 0), "total": cat_total[cat],
                      "accuracy": cat_correct.get(cat, 0) / cat_total[cat] * 100 if cat_total[cat] else 0}
               for cat in cat_total}

    return BenchResult(
        framework="yunshu",
        benchmark="mmlu_pro",
        accuracy=accuracy,
        accuracy_stderr=stderr,
        extra={
            "correct": correct, "total": total,
            "avg_time_per_q": round(sum(times) / len(times), 2),
            "total_time": round(elapsed, 1),
            "max_tokens": max_tokens,
            "method": "5-shot CoT (per-category) + chat_template",
        },
        per_category=per_cat,
    )


# ══════════════════════════════════════════════════════════════════
# VLLM-MLX RUNNER
# ══════════════════════════════════════════════════════════════════

async def run_mmlu_pro_vllm_mlx(
    test_ds,
    fewshot_rows_by_cat: dict[str, list],
    n_samples: int,
    max_tokens: int = 4096,
) -> BenchResult:
    _ensure_ref_path("vllm-mlx")
    from vllm_mlx.engine.simple import SimpleEngine

    P(f"    MMLU-Pro (vllm-mlx): loading model...")
    engine = SimpleEngine(model_name=model_path("llm"))
    engine._is_mllm = False  # Qwen3.5- matches MLLM_PATTERNS but is text-only
    await engine.start()
    tokenizer = engine.tokenizer
    P(f"    Model loaded")

    questions = list(test_ds)
    if 0 < n_samples < len(questions):
        rng = __import__("random").Random(42)
        by_cat = defaultdict(list)
        for q in questions:
            by_cat[q["category"]].append(q)
        per_cat = max(1, n_samples // len(by_cat))
        sampled = []
        for cat, qs in by_cat.items():
            rng.shuffle(qs)
            sampled.extend(qs[:per_cat])
        rng.shuffle(sampled)
        questions = sampled[:n_samples]

    P(f"    MMLU-Pro (vllm-mlx): {len(questions)} questions")

    warmup_msgs = build_mmlu_messages(
        fewshot_rows_by_cat.get(questions[0]["category"], [])[:5], questions[0]
    )
    warmup_prompt = tokenizer.apply_chat_template(
        warmup_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    await engine.generate(warmup_prompt, max_tokens=32, temperature=0.0)
    P(f"    Warmup done")

    cat_correct: dict[str, int] = Counter()
    cat_total: dict[str, int] = Counter()
    correct = 0
    total = 0
    times = []

    t_start = time.perf_counter()
    for i, q in enumerate(questions):
        cat = q["category"]
        msgs = build_mmlu_messages(fewshot_rows_by_cat.get(cat, [])[:5], q)
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        t0 = time.perf_counter()
        result = await engine.generate(prompt, max_tokens=max_tokens, temperature=0.0)
        dt = time.perf_counter() - t0
        times.append(dt)

        text = result.text if hasattr(result, "text") else str(result)
        predicted = extract_mmlu_pro_answer(text)
        answer = q["answer"]
        is_correct = predicted == answer and predicted in VALID_ANSWERS
        if is_correct:
            correct += 1
        total += 1
        if is_correct:
            cat_correct[cat] += 1
        cat_total[cat] += 1
        n_tok = getattr(result, "completion_tokens", "?")

        P(f"    [{i+1}/{len(questions)}] {cat}: pred={predicted} ans={answer} "
          f"{'✓' if is_correct else '✗'} | {dt:.1f}s {n_tok}tok")

        if (i + 1) % 5 == 0 or (i + 1) == len(questions):
            avg = sum(times) / len(times)
            P(f"    ── {i+1}/{len(questions)} — {correct}/{total} = {correct/total*100:.1f}% "
              f"avg {avg:.1f}s/q")

    elapsed = time.perf_counter() - t_start
    accuracy = correct / total * 100 if total else 0
    stderr = math.sqrt(accuracy * (100 - accuracy) / total) if total > 0 else 0
    per_cat = {cat: {"correct": cat_correct.get(cat, 0), "total": cat_total[cat],
                      "accuracy": cat_correct.get(cat, 0) / cat_total[cat] * 100 if cat_total[cat] else 0}
               for cat in cat_total}

    await engine.stop()
    cleanup()

    return BenchResult(
        framework="vllm-mlx",
        benchmark="mmlu_pro",
        accuracy=accuracy,
        accuracy_stderr=stderr,
        extra={
            "correct": correct, "total": total,
            "avg_time_per_q": round(sum(times) / len(times), 2),
            "total_time": round(elapsed, 1),
            "max_tokens": max_tokens,
            "method": "5-shot CoT + chat_template + non-thinking",
            "thinking_enabled": False,
        },
        per_category=per_cat,
    )


# ══════════════════════════════════════════════════════════════════
# OMLX RUNNER
# ══════════════════════════════════════════════════════════════════

async def run_mmlu_pro_omlx(
    test_ds,
    fewshot_rows_by_cat: dict[str, list],
    n_samples: int,
    max_tokens: int = 4096,
) -> BenchResult:
    _ensure_ref_path("omlx")
    from omlx.engine import BatchedEngine

    P(f"    MMLU-Pro (omlx): loading model...")
    engine = BatchedEngine(model_name=model_path("llm"))
    await engine.start()
    tokenizer = engine.tokenizer
    P(f"    Model loaded")

    questions = list(test_ds)
    if 0 < n_samples < len(questions):
        rng = __import__("random").Random(42)
        by_cat = defaultdict(list)
        for q in questions:
            by_cat[q["category"]].append(q)
        per_cat = max(1, n_samples // len(by_cat))
        sampled = []
        for cat, qs in by_cat.items():
            rng.shuffle(qs)
            sampled.extend(qs[:per_cat])
        rng.shuffle(sampled)
        questions = sampled[:n_samples]

    P(f"    MMLU-Pro (omlx): {len(questions)} questions")

    warmup_msgs = build_mmlu_messages(
        fewshot_rows_by_cat.get(questions[0]["category"], [])[:5], questions[0]
    )
    warmup_prompt = tokenizer.apply_chat_template(
        warmup_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    await engine.generate(warmup_prompt, max_tokens=32, temperature=0.0)
    P(f"    Warmup done")

    cat_correct: dict[str, int] = Counter()
    cat_total: dict[str, int] = Counter()
    correct = 0
    total = 0
    times = []

    t_start = time.perf_counter()
    for i, q in enumerate(questions):
        cat = q["category"]
        msgs = build_mmlu_messages(fewshot_rows_by_cat.get(cat, [])[:5], q)
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        t0 = time.perf_counter()
        result = await engine.generate(prompt, max_tokens=max_tokens, temperature=0.0)
        dt = time.perf_counter() - t0
        times.append(dt)

        text = result.text if hasattr(result, "text") else str(result)
        predicted = extract_mmlu_pro_answer(text)
        answer = q["answer"]
        is_correct = predicted == answer and predicted in VALID_ANSWERS
        if is_correct:
            correct += 1
        total += 1
        if is_correct:
            cat_correct[cat] += 1
        cat_total[cat] += 1
        n_tok = getattr(result, "completion_tokens", "?")

        P(f"    [{i+1}/{len(questions)}] {cat}: pred={predicted} ans={answer} "
          f"{'✓' if is_correct else '✗'} | {dt:.1f}s {n_tok}tok")

        if (i + 1) % 5 == 0 or (i + 1) == len(questions):
            avg = sum(times) / len(times)
            P(f"    ── {i+1}/{len(questions)} — {correct}/{total} = {correct/total*100:.1f}% "
              f"avg {avg:.1f}s/q")

    elapsed = time.perf_counter() - t_start
    accuracy = correct / total * 100 if total else 0
    stderr = math.sqrt(accuracy * (100 - accuracy) / total) if total > 0 else 0
    per_cat = {cat: {"correct": cat_correct.get(cat, 0), "total": cat_total[cat],
                      "accuracy": cat_correct.get(cat, 0) / cat_total[cat] * 100 if cat_total[cat] else 0}
               for cat in cat_total}

    await engine.stop()
    cleanup()

    return BenchResult(
        framework="omlx",
        benchmark="mmlu_pro",
        accuracy=accuracy,
        accuracy_stderr=stderr,
        extra={
            "correct": correct, "total": total,
            "avg_time_per_q": round(sum(times) / len(times), 2),
            "total_time": round(elapsed, 1),
            "max_tokens": max_tokens,
            "method": "5-shot CoT + chat_template + non-thinking",
            "thinking_enabled": False,
        },
        per_category=per_cat,
    )


# ══════════════════════════════════════════════════════════════════
# SEQUENTIAL RUNNER (fallback, oMLX, and other benchmarks)
# ══════════════════════════════════════════════════════════════════

def run_sequential_benchmark(
    generate_fn,
    questions: list[dict],
    build_prompt_fn,
    extract_fn,
    check_fn,
    benchmark_name: str,
    max_tokens: int = 256,
) -> BenchResult:
    correct = 0
    total = 0
    cat_correct: dict[str, int] = Counter()
    cat_total: dict[str, int] = Counter()
    times = []

    # Warmup
    warmup_prompt = build_prompt_fn(questions[0])
    generate_fn(warmup_prompt, max_tokens=min(max_tokens, 64), temperature=0.0)
    P(f"    Warmup done")

    bench_start = time.perf_counter()
    for i, q in enumerate(questions):
        prompt = build_prompt_fn(q)
        t0 = time.perf_counter()
        text = generate_fn(prompt, max_tokens=max_tokens, temperature=0.0)
        dt = time.perf_counter() - t0
        times.append(dt)
        predicted = extract_fn(text, q)
        is_correct = check_fn(predicted, q)
        if is_correct:
            correct += 1
        total += 1
        cat = q.get("category", "unknown")
        if is_correct:
            cat_correct[cat] += 1
        cat_total[cat] += 1
        if (i + 1) % 5 == 0 or (i + 1) == len(questions):
            avg = sum(times) / len(times)
            P(f"    {i+1}/{len(questions)} — {correct}/{total} = {correct/total*100:.1f}% "
              f"| {dt:.1f}s/q (avg {avg:.1f}s)")

    elapsed = time.perf_counter() - bench_start
    accuracy = correct / total * 100 if total else 0
    stderr = math.sqrt(accuracy * (100 - accuracy) / total) if total > 0 else 0
    per_cat = {cat: {"correct": cat_correct.get(cat, 0), "total": cat_total[cat],
                      "accuracy": cat_correct.get(cat, 0) / cat_total[cat] * 100 if cat_total[cat] else 0}
               for cat in cat_total}

    return BenchResult(
        framework="sequential",
        benchmark=benchmark_name,
        accuracy=accuracy,
        accuracy_stderr=stderr,
        extra={
            "correct": correct, "total": total,
            "avg_time_per_q": round(sum(times) / len(times), 2) if times else 0,
            "total_time": round(elapsed, 1),
        },
        per_category=per_cat,
    )


# ══════════════════════════════════════════════════════════════════
# DATASET LOADERS
# ══════════════════════════════════════════════════════════════════

def load_dataset_samples(dataset_name: str, split: str, n_samples: int = 0):
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split=split)
    items = list(ds)
    if 0 < n_samples < len(items):
        rng = __import__("random").Random(42)
        rng.shuffle(items)
        items = items[:n_samples]
    return items


# ══════════════════════════════════════════════════════════════════
# THROUGHPUT BENCHMARK
# ══════════════════════════════════════════════════════════════════

def run_throughput_mlxlm(model, tokenizer, n_runs: int, max_tokens: int) -> BenchResult:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)
    prompt = "Write a detailed essay about the history of computing. " * 5

    generate(model, tokenizer, prompt=prompt, max_tokens=16, sampler=sampler, verbose=False)

    speeds = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        text = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                        sampler=sampler, verbose=False)
        elapsed = time.perf_counter() - t0
        n_tok = len(tokenizer.encode(text))
        speeds.append(n_tok / elapsed if elapsed > 0 else 0)

    mean = sum(speeds) / len(speeds)
    std = (sum((s - mean) ** 2 for s in speeds) / len(speeds)) ** 0.5
    return BenchResult(framework="mlx-lm", benchmark="throughput",
                       throughput=mean, throughput_std=std, memory_mb=get_rss_mb(),
                       extra={"n_runs": n_runs, "max_tokens": max_tokens})


async def run_throughput_yunshu(n_runs: int, max_tokens: int) -> BenchResult:
    from yunshu_engine.batched_engine import BatchedEngine

    engine = BatchedEngine(model_name=model_path("llm"))
    await engine.start()

    prompt = "Write a detailed essay about the history of computing. " * 5
    await engine.generate(prompt=prompt, max_tokens=16, temperature=0.0)

    speeds = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        r = await engine.generate(prompt=prompt, max_tokens=max_tokens, temperature=0.0)
        elapsed = time.perf_counter() - t0
        text = r.text if hasattr(r, "text") else str(r)
        n_tok = len(engine._tokenizer.encode(text)) if engine._tokenizer else max_tokens
        speeds.append(n_tok / elapsed if elapsed > 0 else 0)

    await engine.stop()
    mean = sum(speeds) / len(speeds)
    std = (sum((s - mean) ** 2 for s in speeds) / len(speeds)) ** 0.5
    return BenchResult(framework="yunshu", benchmark="throughput",
                       throughput=mean, throughput_std=std, memory_mb=get_rss_mb(),
                       extra={"n_runs": n_runs, "max_tokens": max_tokens})


# ══════════════════════════════════════════════════════════════════
# SUMMARY REPORTING
# ══════════════════════════════════════════════════════════════════

def print_summary(all_results: list[BenchResult]):
    P(f"\n{'═' * 90}")
    P(f"  BENCHMARK SUMMARY")
    P(f"{'═' * 90}")

    by_bench: dict[str, list[BenchResult]] = defaultdict(list)
    for r in all_results:
        by_bench[r.benchmark].append(r)

    for bench_name, results in by_bench.items():
        P(f"\n  ── {bench_name.upper()} ──")

        if bench_name == "throughput":
            P(f"  {'Framework':<12} {'tok/s':>10} {'±std':>8} {'vs mlx-lm':>10}")
            P(f"  {'─' * 12} {'─' * 10} {'─' * 8} {'─' * 10}")
            base = next((r for r in results if r.framework == "mlx-lm"), None)
            for r in results:
                tps = f"{r.throughput:.1f}" if r.throughput else "N/A"
                std = f"±{r.throughput_std:.1f}" if r.throughput_std else ""
                if r.framework == "mlx-lm":
                    ratio = "BASELINE"
                else:
                    ratio = f"{(r.throughput or 0) / (base.throughput or 1):.2f}x"
                P(f"  {r.framework:<12} {tps:>10} {std:>8} {ratio:>10}")

        elif bench_name == "mmlu_pro":
            base = next((r for r in results if r.framework == "mlx-lm"), None)
            P(f"  {'Framework':<12} {'Accuracy':>10} {'±stderr':>8} {'Correct':>10} {'Time':>10} {'vs Base':>8}")
            P(f"  {'─' * 12} {'─' * 10} {'─' * 8} {'─' * 10} {'─' * 10} {'─' * 8}")
            for r in results:
                acc = f"{r.accuracy:.1f}%" if r.accuracy is not None else "N/A"
                se = f"±{r.accuracy_stderr:.1f}" if r.accuracy_stderr else ""
                corr = f"{r.extra.get('correct', '?')}/{r.extra.get('total', '?')}"
                t = f"{r.extra.get('avg_time_per_q', 0):.1f}s/q"
                if r.framework == "mlx-lm":
                    diff = "BASELINE"
                else:
                    d = (r.accuracy or 0) - (base.accuracy if base else 0)
                    diff = f"{d:+.1f}%"
                P(f"  {r.framework:<12} {acc:>10} {se:>8} {corr:>10} {t:>10} {diff:>8}")

            for r in results:
                if r.per_category:
                    P(f"\n  Per-category ({r.framework}):")
                    for cat in sorted(r.per_category):
                        pc = r.per_category[cat]
                        P(f"    {cat:<22} {pc['correct']:>4}/{pc['total']:<4} = {pc['accuracy']:.1f}%")
                    break
        else:
            for r in results:
                P(f"  {r.framework:<12} {r.benchmark}: {r.extra}")

    P(f"\n{'═' * 90}")


# ══════════════════════════════════════════════════════════════════
# MULTIMODAL BENCHMARK (VLM, TTS, ASR, Image)
# ══════════════════════════════════════════════════════════════════

async def bench_vlm() -> dict:
    """VLM: text generation speed + throughput."""
    from yunshu_engine.vlm_engine import VLMEngine
    engine = VLMEngine(model_path("vlm"))
    await engine.start()
    rss = get_rss_mb()

    # Text-only warmup
    await engine.generate(messages=[{"role": "user", "content": "Hello"}], max_tokens=16, temperature=0.0)

    # Text-only benchmark
    prompts = [
        [{"role": "user", "content": "Explain gravity in one sentence."}],
        [{"role": "user", "content": "What is the capital of Japan?"}],
        [{"role": "user", "content": "Write a haiku about programming."}],
    ]
    times = []
    for msgs in prompts:
        t0 = time.perf_counter()
        r = await engine.generate(messages=msgs, max_tokens=64, temperature=0.0)
        dt = time.perf_counter() - t0
        # VLM returns dict with "text", not GenerationOutput
        if isinstance(r, dict):
            text = r.get("text", "")
        else:
            text = getattr(r, 'text', '')
        # Estimate tokens from text (rough: ~4 chars per token)
        n_tok = max(1, len(text) // 4)
        times.append({"dt": dt, "tokens": n_tok, "text_len": len(text)})

    avg_dt = sum(t["dt"] for t in times) / len(times)
    avg_tok = sum(t["tokens"] for t in times) / len(times)
    tok_s = avg_tok / avg_dt if avg_dt > 0 else 0

    result = {
        "modality": "VLM (text-only)", "rss_mb": round(rss, 0),
        "avg_latency_ms": round(avg_dt * 1000, 0),
        "avg_tokens": round(avg_tok, 1),
        "tok_per_s": round(tok_s, 1),
    }
    await engine.stop()
    return result


async def bench_tts() -> dict:
    """TTS: synthesis speed + RTF (Real-Time Factor)."""
    from yunshu_engine.audio_engine import TTSEngine
    engine = TTSEngine(model_path("tts"))
    await engine.start()
    rss = get_rss_mb()

    # Warmup
    await engine.synthesize("Hello world.", speed=1.0)

    # Benchmark
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Today is a beautiful day for a walk in the park.",
    ]
    durations = []
    for text in texts:
        t0 = time.perf_counter()
        wav_bytes = await engine.synthesize(text, speed=1.0)
        dt = time.perf_counter() - t0
        # Estimate audio duration (16kHz, 16-bit mono = 32000 bytes/s)
        audio_duration = len(wav_bytes) / 32000
        rtf = dt / audio_duration if audio_duration > 0 else 0
        durations.append({"gen_s": dt, "audio_s": audio_duration, "rtf": rtf, "bytes": len(wav_bytes)})

    avg_gen = sum(d["gen_s"] for d in durations) / len(durations)
    avg_audio = sum(d["audio_s"] for d in durations) / len(durations)
    avg_rtf = sum(d["rtf"] for d in durations) / len(durations)

    result = {
        "modality": "TTS", "rss_mb": round(rss, 0),
        "avg_latency_ms": round(avg_gen * 1000, 0),
        "avg_audio_s": round(avg_audio, 2),
        "rtf": round(avg_rtf, 3),
        "avg_output_bytes": round(sum(d["bytes"] for d in durations) / len(durations), 0),
    }
    await engine.stop()
    return result


async def bench_asr() -> dict:
    """ASR: transcription speed."""
    from yunshu_engine.audio_engine import ASREngine
    engine = ASREngine(model_path("asr"))
    await engine.start()
    rss = get_rss_mb()

    # Create a test WAV file (1s silence at 16kHz, 16-bit mono)
    import tempfile, os, struct, wave
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b'\x00\x00' * 16000)  # 1s silence
    tmp.close()

    # Warmup
    await engine.transcribe(tmp.name)

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(3):
        result_text = await engine.transcribe(tmp.name)
    dt = time.perf_counter() - t0

    os.unlink(tmp.name)
    avg_dt = dt / 3

    result_data = {
        "modality": "ASR", "rss_mb": round(rss, 0),
        "avg_latency_ms": round(avg_dt * 1000, 0),
        "transcription": str(result_text)[:100] if result_text else "",
    }
    await engine.stop()
    return result_data


async def bench_image() -> dict:
    """Image: generation speed + step latency."""
    from yunshu_engine.image_engine import ImageGenEngine
    engine = ImageGenEngine(model_path("image"))
    await engine.start()
    rss = get_rss_mb()

    # Benchmark
    t0 = time.perf_counter()
    img_bytes = await engine.generate_image("A cat sitting on a windowsill at sunset.", num_steps=4)
    dt = time.perf_counter() - t0

    result_data = {
        "modality": "Image", "rss_mb": round(rss, 0),
        "latency_ms": round(dt * 1000, 0),
        "output_bytes": len(img_bytes) if img_bytes else 0,
        "steps": 4,
    }
    await engine.stop()
    return result_data


async def run_multimodal_benchmarks() -> list[dict]:
    results = []

    if model_exists("vlm"):
        P(f"\n  ── VLM (text-only) ──")
        try:
            r = await bench_vlm()
            results.append(r)
            P(f"    {r['tok_per_s']:.1f} tok/s, {r['avg_latency_ms']:.0f}ms avg, {r['rss_mb']:.0f}MB")
        except Exception as e:
            P(f"    VLM FAILED: {e}")
            import traceback; traceback.print_exc()
        cleanup()

    if model_exists("tts"):
        P(f"\n  ── TTS ──")
        try:
            r = await bench_tts()
            results.append(r)
            P(f"    RTF={r['rtf']:.3f}, {r['avg_latency_ms']:.0f}ms, {r['avg_audio_s']:.2f}s audio, {r['rss_mb']:.0f}MB")
        except Exception as e:
            P(f"    TTS FAILED: {e}")
            import traceback; traceback.print_exc()
        cleanup()

    if model_exists("asr"):
        P(f"\n  ── ASR ──")
        try:
            r = await bench_asr()
            results.append(r)
            P(f"    {r['avg_latency_ms']:.0f}ms, {r['rss_mb']:.0f}MB")
        except Exception as e:
            P(f"    ASR FAILED: {e}")
            import traceback; traceback.print_exc()
        cleanup()

    if model_exists("image"):
        P(f"\n  ── Image ──")
        try:
            r = await bench_image()
            results.append(r)
            P(f"    {r['latency_ms']:.0f}ms ({r['steps']} steps), {r['output_bytes']}B, {r['rss_mb']:.0f}MB")
        except Exception as e:
            P(f"    Image FAILED: {e}")
            import traceback; traceback.print_exc()
        cleanup()

    if results:
        P(f"\n  ── MULTIMODAL SUMMARY ──")
        P(f"    {'Modality':<16} {'Latency':>10} {'Speed':>12} {'Output':>12} {'Memory':>10}")
        P(f"    {'─' * 16} {'─' * 10} {'─' * 12} {'─' * 12} {'─' * 10}")
        for r in results:
            mod = r["modality"]
            if "TTS" in mod:
                latency = f"{r['avg_latency_ms']:.0f}ms"
                speed = f"RTF={r['rtf']:.3f}"
                output = f"{r['avg_audio_s']:.2f}s audio"
            elif "Image" in mod:
                latency = f"{r['latency_ms']:.0f}ms"
                speed = f"{r['steps']} steps"
                output = f"{r['output_bytes']/1024:.0f}KB"
            elif "ASR" in mod:
                latency = f"{r['avg_latency_ms']:.0f}ms"
                speed = "—"
                output = "—"
            else:
                latency = f"{r['avg_latency_ms']:.0f}ms"
                speed = f"{r['tok_per_s']:.1f} tok/s"
                output = f"{r['avg_tokens']:.0f} tok"
            P(f"    {mod:<16} {latency:>10} {speed:>12} {output:>12} {r['rss_mb']:>8.0f}MB")

    return results


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

BENCHMARKS = ["mmlu_pro", "throughput", "perf", "multimodal"]


# ══════════════════════════════════════════════════════════════════
# PERFORMANCE BENCHMARK (TTFT, tok/s, Memory)
# ══════════════════════════════════════════════════════════════════

PERF_SCENARIOS = [
    ("short_prompt_short_gen",   5, 128),
    ("medium_prompt_medium_gen", 20, 256),
    ("long_prompt_long_gen",     80, 512),
]


def _build_perf_prompt(n_repeats: int) -> str:
    return "Write a detailed essay about the history of computing from the 1940s to present day. " * n_repeats


def run_perf_mlxlm(model, tokenizer) -> list[dict]:
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    from mlx_lm import stream_generate as mlx_stream_gen

    sampler = make_sampler(temp=0.0)
    rss = get_rss_mb()
    results = []

    # Warmup
    warmup_ids = mx.array(tokenizer.encode(_build_perf_prompt(1)))
    for _ in generate_step(warmup_ids, model, max_tokens=16, sampler=sampler):
        pass
    mx.synchronize()
    mx.clear_cache()

    for name, n_rep, max_tok in PERF_SCENARIOS:
        prompt = _build_perf_prompt(n_rep)
        input_ids = tokenizer.encode(prompt)
        prompt_toks = len(input_ids)

        # Non-streaming (generate_step — GPU-level TTFT)
        t0 = time.perf_counter()
        ids = mx.array(input_ids)
        first = True
        ttft = None
        tokens = []
        for token, _ in generate_step(ids, model, max_tokens=max_tok, sampler=sampler):
            if first:
                ttft = time.perf_counter() - t0
                first = False
            tokens.append(int(token))
            if hasattr(tokenizer, 'eos_token_id') and int(token) == tokenizer.eos_token_id:
                break
            if hasattr(tokenizer, 'eos_token_ids') and int(token) in tokenizer.eos_token_ids:
                break
        dt = time.perf_counter() - t0
        mx.synchronize()
        mx.clear_cache()

        gen_time = dt - (ttft or 0)
        tok_s = len(tokens) / gen_time if gen_time > 0 else 0
        results.append({
            "scenario": name, "ttft_ms": round((ttft or 0) * 1000, 1),
            "tok_per_s": round(tok_s, 1), "tokens": len(tokens),
            "prompt_tokens": prompt_toks, "total_s": round(dt, 2),
            "rss_mb": round(rss, 0),
        })

        # Streaming (stream_generate — end-to-end TTFT)
        t0 = time.perf_counter()
        ttft_s = None
        n_tok = 0
        text = ""
        for resp in mlx_stream_gen(model, tokenizer, prompt, max_tokens=max_tok, sampler=sampler):
            if ttft_s is None and resp.text:
                ttft_s = time.perf_counter() - t0
            text += resp.text
            n_tok += 1
        dt = time.perf_counter() - t0
        mx.synchronize()
        mx.clear_cache()
        gen_time = dt - (ttft_s or 0)
        tok_s = n_tok / gen_time if gen_time > 0 else 0
        results.append({
            "scenario": name + "_stream", "ttft_ms": round((ttft_s or 0) * 1000, 1),
            "tok_per_s": round(tok_s, 1), "tokens": n_tok,
            "prompt_tokens": prompt_toks, "total_s": round(dt, 2),
            "rss_mb": round(rss, 0),
        })

    return results


async def run_perf_yunshu() -> list[dict]:
    from yunshu_engine.batched_engine import BatchedEngine

    engine = BatchedEngine(model_name=model_path("llm"))
    await engine.start()
    tokenizer = engine._tokenizer
    rss = get_rss_mb()
    results = []

    for name, n_rep, max_tok in PERF_SCENARIOS:
        prompt = _build_perf_prompt(n_rep)
        input_ids = tokenizer.encode(prompt)
        prompt_toks = len(input_ids)

        # Warmup
        await engine.generate(prompt=_build_perf_prompt(1), max_tokens=16, temperature=0.0)

        t0 = time.perf_counter()
        r = await engine.generate(prompt=prompt, max_tokens=max_tok, temperature=0.0)
        dt = time.perf_counter() - t0

        ttft_ms = r.ttft_ms if hasattr(r, 'ttft_ms') else 0
        gen_time = (dt - ttft_ms / 1000) if ttft_ms > 0 else dt
        tok_s = r.completion_tokens / gen_time if gen_time > 0 else 0
        results.append({
            "scenario": name, "ttft_ms": ttft_ms,
            "tok_per_s": round(tok_s, 1), "tokens": r.completion_tokens,
            "prompt_tokens": prompt_toks, "total_s": round(dt, 2),
            "rss_mb": round(rss, 0),
        })

    # Streaming TTFT measurement
    stream_results = []
    for name, n_rep, max_tok in PERF_SCENARIOS:
        prompt = _build_perf_prompt(n_rep)
        input_ids = tokenizer.encode(prompt)
        prompt_toks = len(input_ids)

        t0 = time.perf_counter()
        ttft = None
        total_tokens = 0
        async for chunk in engine.stream_generate(prompt=prompt, max_tokens=max_tok, temperature=0.0):
            if ttft is None and chunk.new_text:
                ttft = time.perf_counter() - t0
            total_tokens = chunk.completion_tokens
        dt = time.perf_counter() - t0
        gen_time = dt - (ttft or 0)
        tok_s = total_tokens / gen_time if gen_time > 0 else 0
        stream_results.append({
            "scenario": name + "_stream", "ttft_ms": round((ttft or 0) * 1000, 1),
            "tok_per_s": round(tok_s, 1), "tokens": total_tokens,
            "prompt_tokens": prompt_toks, "total_s": round(dt, 2),
            "rss_mb": round(rss, 0),
        })

    await engine.stop()
    return results + stream_results


async def _perf_stream_any(engine, prompt: str, max_tokens: int, temperature: float = 0.0) -> tuple[float, float, int, float]:
    """Run streaming generation and measure TTFT, tok/s, token count, total time."""
    t0 = time.perf_counter()
    ttft = None
    total_tokens = 0
    async for chunk in engine.stream_generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature):
        if ttft is None and getattr(chunk, 'new_text', getattr(chunk, 'text', '')):
            ttft = time.perf_counter() - t0
        total_tokens = max(total_tokens, getattr(chunk, 'completion_tokens', total_tokens + 1))
    dt = time.perf_counter() - t0
    gen_time = dt - (ttft or 0)
    tok_s = total_tokens / gen_time if gen_time > 0 else 0
    ttft_ms = (ttft or 0) * 1000
    return ttft_ms, tok_s, total_tokens, dt


async def run_perf_vllm_mlx() -> list[dict]:
    _ensure_ref_path("vllm-mlx")
    from vllm_mlx.engine.simple import SimpleEngine

    engine = SimpleEngine(model_name=model_path("llm"))
    engine._is_mllm = False
    await engine.start()
    tokenizer = engine.tokenizer
    rss = get_rss_mb()
    results = []

    for name, n_rep, max_tok in PERF_SCENARIOS:
        prompt = _build_perf_prompt(n_rep)
        input_ids = tokenizer.encode(prompt)
        prompt_toks = len(input_ids)

        await engine.generate(_build_perf_prompt(1), max_tokens=16, temperature=0.0)

        # Non-streaming (generate = stream_generate accumulator, no separate TTFT)
        t0 = time.perf_counter()
        r = await engine.generate(prompt, max_tokens=max_tok, temperature=0.0)
        dt_ns = time.perf_counter() - t0
        n_tok = getattr(r, 'completion_tokens', 0) or len(tokenizer.encode(getattr(r, 'text', '')))
        tok_s_ns = n_tok / dt_ns if dt_ns > 0 else 0
        results.append({
            "scenario": name, "ttft_ms": 0,
            "tok_per_s": round(tok_s_ns, 1), "tokens": n_tok,
            "prompt_tokens": prompt_toks, "total_s": round(dt_ns, 2),
            "rss_mb": round(rss, 0),
        })

        # Streaming (has TTFT)
        ttft_ms, tok_s, tokens, dt = await _perf_stream_any(engine, prompt, max_tok)
        results.append({
            "scenario": name + "_stream", "ttft_ms": round(ttft_ms, 1),
            "tok_per_s": round(tok_s, 1), "tokens": tokens,
            "prompt_tokens": prompt_toks, "total_s": round(dt, 2),
            "rss_mb": round(rss, 0),
        })

    await engine.stop()
    cleanup()
    return results


async def run_perf_omlx() -> list[dict]:
    _ensure_ref_path("omlx")
    from omlx.engine import BatchedEngine

    engine = BatchedEngine(model_name=model_path("llm"))
    await engine.start()
    tokenizer = engine.tokenizer
    rss = get_rss_mb()
    results = []

    for name, n_rep, max_tok in PERF_SCENARIOS:
        prompt = _build_perf_prompt(n_rep)
        input_ids = tokenizer.encode(prompt)
        prompt_toks = len(input_ids)

        await engine.generate(_build_perf_prompt(1), max_tokens=16, temperature=0.0)

        # Non-streaming
        t0 = time.perf_counter()
        r = await engine.generate(prompt, max_tokens=max_tok, temperature=0.0)
        dt_ns = time.perf_counter() - t0
        n_tok = getattr(r, 'completion_tokens', 0) or len(tokenizer.encode(getattr(r, 'text', '')))
        tok_s_ns = n_tok / dt_ns if dt_ns > 0 else 0
        results.append({
            "scenario": name, "ttft_ms": 0,
            "tok_per_s": round(tok_s_ns, 1), "tokens": n_tok,
            "prompt_tokens": prompt_toks, "total_s": round(dt_ns, 2),
            "rss_mb": round(rss, 0),
        })

        # Streaming
        ttft_ms, tok_s, tokens, dt = await _perf_stream_any(engine, prompt, max_tok)
        results.append({
            "scenario": name + "_stream", "ttft_ms": round(ttft_ms, 1),
            "tok_per_s": round(tok_s, 1), "tokens": tokens,
            "prompt_tokens": prompt_toks, "total_s": round(dt, 2),
            "rss_mb": round(rss, 0),
        })

    await engine.stop()
    cleanup()
    return results


def print_perf_summary(perf_data: dict[str, list[dict]]):
    P(f"\n{'═' * 100}")
    P(f"  PERFORMANCE SUMMARY")
    P(f"{'═' * 100}")

    for mode, suffix in [("non-streaming", ""), ("streaming", "_stream")]:
        P(f"\n  ── {mode.upper()} ──")
        for scenario_name, _, _ in PERF_SCENARIOS:
            key = scenario_name + suffix
            rows = []
            best_ttft = float('inf')
            best_tps = 0
            best_mem = float('inf')
            for fw, results in perf_data.items():
                for r in results:
                    if r["scenario"] == key:
                        rows.append((fw, r))
                        if r["ttft_ms"] > 0:
                            best_ttft = min(best_ttft, r["ttft_ms"])
                        best_tps = max(best_tps, r["tok_per_s"])
                        best_mem = min(best_mem, r["rss_mb"])

            if not rows:
                continue

            P(f"\n    {scenario_name}:")
            P(f"    {'Framework':<14} {'TTFT':>10} {'tok/s':>10} {'Tokens':>8} {'Memory':>10}")
            P(f"    {'─' * 14} {'─' * 10} {'─' * 10} {'─' * 8} {'─' * 10}")

            for fw, r in sorted(rows, key=lambda x: -x[1]["tok_per_s"]):
                ttft_str = f"{r['ttft_ms']:.1f}ms" if r['ttft_ms'] > 0 else "—"
                tps = r["tok_per_s"]
                mem = r["rss_mb"]
                ttft_mark = " ★" if r["ttft_ms"] > 0 and best_ttft < float('inf') and r["ttft_ms"] <= best_ttft * 1.02 else ""
                tps_mark = " ★" if tps >= best_tps * 0.98 else ""
                mem_mark = " ★" if best_mem < float('inf') and mem <= best_mem * 1.02 else ""
                P(f"    {fw:<14} {ttft_str:>10}{ttft_mark} {tps:>10.1f}{tps_mark} {r['tokens']:>8} {mem:>8.0f}MB{mem_mark}")

    # Grand summary: best per metric across all scenarios
    P(f"\n  ── GRAND SUMMARY (best per framework across all scenarios) ──")
    P(f"    {'Framework':<14} {'TTFT best':>12} {'tok/s best':>12} {'tok/s avg':>12} {'Memory':>10}")
    P(f"    {'─' * 14} {'─' * 12} {'─' * 12} {'─' * 12} {'─' * 10}")

    for fw in ["mlx-lm", "yunshu", "vllm-mlx", "omlx"]:
        results = perf_data.get(fw, [])
        if not results:
            continue
        all_ttft = [r["ttft_ms"] for r in results if r["ttft_ms"] > 0]
        all_tps = [r["tok_per_s"] for r in results]
        all_mem = [r["rss_mb"] for r in results]
        best_ttft_str = f"{min(all_ttft):.0f}ms" if all_ttft else "—"
        P(f"    {fw:<14} {best_ttft_str:>12} {max(all_tps):>10.1f}  {sum(all_tps)/len(all_tps):>10.1f}  {min(all_mem):>8.0f}MB")

    P(f"\n{'═' * 100}")



async def main_async(args):
    batch_size = args.batch_size
    n_runs = 3 if args.quick else 5
    samples = args.samples
    if args.full:
        samples = 0

    frameworks = set(args.framework)
    if "all" in frameworks:
        frameworks = {"mlx-lm", "yunshu", "vllm-mlx", "omlx"}
    benchmarks = set(args.bench)
    if "all" in benchmarks:
        benchmarks = set(BENCHMARKS)

    P(f"{'═' * 90}")
    P(f"  Yunshu Benchmark — lm-evaluation-harness compatible")
    P(f"{'═' * 90}")
    P(f"  Extraction: answer is \\(?([A-J])\\)?")
    P(f"  Method: 5-shot CoT (lm-eval-harness) + chat_template")
    P(f"  Max tokens: 4096 | Temperature: 0.0 (greedy)")
    P(f"  Batch size: {batch_size}")
    if samples == 0:
        P(f"  Samples: FULL dataset (12,032)")
    else:
        P(f"  Samples: {samples}")
    P(f"  Frameworks: {', '.join(sorted(frameworks))}")
    P(f"")

    if not model_exists("llm"):
        P(f"  Model not found: {MODELS['llm']}")
        return

    all_results: list[BenchResult] = []

    # Load datasets
    from datasets import load_dataset
    P("Loading datasets...")
    val_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
    test_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    fewshot_rows_by_cat = get_fewshot_rows_by_category(val_ds, n_shot=5)
    P(f"  MMLU-Pro: {len(test_ds)} test, {len(val_ds)} validation (few-shot)")
    P(f"  Few-shot: {len(fewshot_rows_by_cat)} categories, "
      f"5 examples each")
    P("")

    # ── mlx-lm ──
    if "mlx-lm" in frameworks and "mmlu_pro" in benchmarks:
        cached = None if args.no_cache else load_cached("mlx-lm", "mmlu_pro", samples, 4096)
        if cached:
            P(f"\n  MMLU-PRO (mlx-lm): CACHED — {cached.accuracy:.1f}% "
              f"({cached.extra['correct']}/{cached.extra['total']})")
            all_results.append(cached)
        else:
            P(f"{'─' * 90}")
            P(f"  Framework: MLX-LM (stream_generate + generate_until)")
            P(f"{'─' * 90}")
            from mlx_lm import load
            P("  Loading model...")
            model, tokenizer = load(model_path("llm"))
            P(f"  Model loaded")
            P(f"\n  MMLU-PRO (5-shot CoT + chat_template):")
            result = await asyncio.to_thread(
                run_mmlu_pro_mlxlm, model, tokenizer, test_ds, fewshot_rows_by_cat, samples, max_tokens=4096,
            )
            result.framework = "mlx-lm"
            all_results.append(result)
            save_cached(result, samples, 4096)
            P(f"  → {result.accuracy:.1f}% ±{result.accuracy_stderr:.1f}% "
              f"({result.extra['correct']}/{result.extra['total']}) "
              f"[{result.extra['avg_time_per_q']}s/q, {result.extra['total_time']}s total]")
            del model, tokenizer
            cleanup()

    # ── Yunshu ──
    if "yunshu" in frameworks and "mmlu_pro" in benchmarks:
        cached = None if args.no_cache else load_cached("yunshu", "mmlu_pro", samples, 4096)
        if cached:
            P(f"\n  MMLU-PRO (yunshu): CACHED — {cached.accuracy:.1f}% "
              f"({cached.extra['correct']}/{cached.extra['total']})")
            all_results.append(cached)
        else:
            P(f"\n{'─' * 90}")
            P(f"  Framework: Yunshu (BatchedEngine)")
            P(f"{'─' * 90}")
            P(f"\n  MMLU-PRO (5-shot CoT + chat_template):")
            result = await run_mmlu_pro_yunshu(
                test_ds, fewshot_rows_by_cat, samples, max_tokens=4096,
            )
            all_results.append(result)
            save_cached(result, samples, 4096)
            P(f"  → {result.accuracy:.1f}% ±{result.accuracy_stderr:.1f}% "
              f"({result.extra['correct']}/{result.extra['total']}) "
              f"[{result.extra['avg_time_per_q']}s/q, {result.extra['total_time']}s total]")
            cleanup()

    # ── vllm-mlx ──
    if "vllm-mlx" in frameworks and "mmlu_pro" in benchmarks:
        cached = None if args.no_cache else load_cached("vllm-mlx", "mmlu_pro", samples, 4096)
        if cached:
            P(f"\n  MMLU-PRO (vllm-mlx): CACHED — {cached.accuracy:.1f}% "
              f"({cached.extra['correct']}/{cached.extra['total']})")
            all_results.append(cached)
        else:
            P(f"\n{'─' * 90}")
            P(f"  Framework: vllm-mlx (SimpleEngine)")
            P(f"{'─' * 90}")
            P(f"\n  MMLU-PRO (5-shot CoT + chat_template):")
            result = await run_mmlu_pro_vllm_mlx(
                test_ds, fewshot_rows_by_cat, samples, max_tokens=4096,
            )
            all_results.append(result)
            save_cached(result, samples, 4096)
            P(f"  → {result.accuracy:.1f}% ±{result.accuracy_stderr:.1f}% "
              f"({result.extra['correct']}/{result.extra['total']}) "
              f"[{result.extra['avg_time_per_q']}s/q, {result.extra['total_time']}s total]")
            cleanup()

    # ── omlx ──
    if "omlx" in frameworks and "mmlu_pro" in benchmarks:
        cached = None if args.no_cache else load_cached("omlx", "mmlu_pro", samples, 4096)
        if cached:
            P(f"\n  MMLU-PRO (omlx): CACHED — {cached.accuracy:.1f}% "
              f"({cached.extra['correct']}/{cached.extra['total']})")
            all_results.append(cached)
        else:
            P(f"\n{'─' * 90}")
            P(f"  Framework: omlx (BatchedEngine)")
            P(f"{'─' * 90}")
            P(f"\n  MMLU-PRO (5-shot CoT + chat_template):")
            result = await run_mmlu_pro_omlx(
                test_ds, fewshot_rows_by_cat, samples, max_tokens=4096,
            )
            all_results.append(result)
            save_cached(result, samples, 4096)
            P(f"  → {result.accuracy:.1f}% ±{result.accuracy_stderr:.1f}% "
              f"({result.extra['correct']}/{result.extra['total']}) "
              f"[{result.extra['avg_time_per_q']}s/q, {result.extra['total_time']}s total]")
            cleanup()

    # ── Performance (TTFT, tok/s, Memory) ──
    if "perf" in benchmarks:
        perf_data = {}

        if "mlx-lm" in frameworks:
            P(f"\n{'─' * 90}")
            P(f"  Performance: MLX-LM")
            P(f"{'─' * 90}")
            from mlx_lm import load
            P("  Loading model...")
            model, tokenizer = load(model_path("llm"))
            P(f"  Model loaded. RSS: {get_rss_mb():.0f} MB")
            # Run directly — generate_step must run on main thread for mlx-lm
            perf_data["mlx-lm"] = run_perf_mlxlm(model, tokenizer)
            for r in perf_data["mlx-lm"]:
                P(f"    {r['scenario']}: TTFT={r['ttft_ms']:.1f}ms, {r['tok_per_s']:.1f} tok/s, {r['tokens']}tok, {r['rss_mb']:.0f}MB")
            del model, tokenizer
            cleanup()

        if "yunshu" in frameworks:
            P(f"\n{'─' * 90}")
            P(f"  Performance: Yunshu")
            P(f"{'─' * 90}")
            P("  Loading model...")
            rss0 = get_rss_mb()
            perf_data["yunshu"] = await run_perf_yunshu()
            for r in perf_data["yunshu"]:
                P(f"    {r['scenario']}: TTFT={r['ttft_ms']:.1f}ms, {r['tok_per_s']:.1f} tok/s, {r['tokens']}tok, {r['rss_mb']:.0f}MB")
            cleanup()

        if "vllm-mlx" in frameworks:
            try:
                P(f"\n{'─' * 90}")
                P(f"  Performance: vllm-mlx")
                P(f"{'─' * 90}")
                P("  Loading model...")
                perf_data["vllm-mlx"] = await run_perf_vllm_mlx()
                for r in perf_data["vllm-mlx"]:
                    P(f"    {r['scenario']}: {r['tok_per_s']:.1f} tok/s, {r['tokens']}tok, {r['rss_mb']:.0f}MB")
                cleanup()
            except Exception as e:
                P(f"  vllm-mlx SKIPPED: {e}")

        if "omlx" in frameworks:
            try:
                P(f"\n{'─' * 90}")
                P(f"  Performance: omlx")
                P(f"{'─' * 90}")
                P("  Loading model...")
                perf_data["omlx"] = await run_perf_omlx()
                for r in perf_data["omlx"]:
                    P(f"    {r['scenario']}: {r['tok_per_s']:.1f} tok/s, {r['tokens']}tok, {r['rss_mb']:.0f}MB")
                cleanup()
            except Exception as e:
                P(f"    omlx SKIPPED: {e}")

        if perf_data:
            print_perf_summary(perf_data)

    # ── Multimodal benchmarks ──
    if "multimodal" in benchmarks:
        P(f"\n{'─' * 90}")
        P(f"  Multimodal Benchmarks")
        P(f"{'─' * 90}")
        mm_results = await run_multimodal_benchmarks()
        if mm_results:
            # Save multimodal results
            out_path = ROOT / "bench" / "results" / f"multimodal_{int(time.time())}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(mm_results, indent=2, default=str))
            P(f"\n  Results saved to {out_path}")

    # Summary
    if all_results:
        print_summary(all_results)
        out_path = ROOT / "bench" / "results" / f"bench_{int(time.time())}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json_data = [{k: v for k, v in r.__dict__.items() if v is not None and v != {} and v != []}
                     for r in all_results]
        out_path.write_text(json.dumps(json_data, indent=2, default=str))
        P(f"\n  Results saved to {out_path}")
    else:
        P("\nNo results collected.")


def main_sync(args):
    """Synchronous main for mlx-lm only — avoids asyncio overhead."""
    samples = args.samples
    if args.full:
        samples = 0

    benchmarks = set(args.bench)
    if "all" in benchmarks:
        benchmarks = set(BENCHMARKS)

    P(f"{'═' * 90}")
    P(f"  Yunshu Benchmark — lm-evaluation-harness compatible")
    P(f"{'═' * 90}")
    P(f"  Extraction: answer is \\(?([A-J])\\)?")
    P(f"  Method: 5-shot CoT (lm-eval-harness) + chat_template")
    P(f"  Max tokens: 4096 | Temperature: 0.0 (greedy)")
    if samples == 0:
        P(f"  Samples: FULL dataset (12,032)")
    else:
        P(f"  Samples: {samples}")
    P(f"  Frameworks: mlx-lm")
    P(f"")

    if not model_exists("llm"):
        P(f"  Model not found: {MODELS['llm']}")
        return

    all_results: list[BenchResult] = []

    from datasets import load_dataset
    P("Loading datasets...")
    val_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
    test_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    fewshot_rows_by_cat = get_fewshot_rows_by_category(val_ds, n_shot=5)
    P(f"  MMLU-Pro: {len(test_ds)} test, {len(val_ds)} validation (few-shot)")
    P(f"  Few-shot: {len(fewshot_rows_by_cat)} categories, 5 examples each")
    P("")

    P(f"{'─' * 90}")
    P(f"  Framework: MLX-LM (stream_generate + generate_until)")
    P(f"{'─' * 90}")

    from mlx_lm import load
    P("  Loading model...")
    rss_before = get_rss_mb()
    model, tokenizer = load(model_path("llm"))
    rss_after = get_rss_mb()
    P(f"  Model loaded. RSS: {rss_before:.0f} → {rss_after:.0f} MB (+{rss_after - rss_before:.0f} MB)")

    if "mmlu_pro" in benchmarks:
        P(f"\n  MMLU-PRO (5-shot CoT + chat_template):")
        result = run_mmlu_pro_mlxlm(
            model, tokenizer, test_ds, fewshot_rows_by_cat, samples,
            max_tokens=4096,
        )
        result.framework = "mlx-lm"
        all_results.append(result)
        P(f"  → {result.accuracy:.1f}% ±{result.accuracy_stderr:.1f}% "
          f"({result.extra['correct']}/{result.extra['total']}) "
          f"[{result.extra['avg_time_per_q']}s/q, {result.extra['total_time']}s total]")

    if "throughput" in benchmarks:
        n_runs = 3 if args.quick else 5
        P(f"\n  THROUGHPUT: {n_runs} runs × 128 tokens")
        result = run_throughput_mlxlm(model, tokenizer, n_runs, 128)
        all_results.append(result)
        P(f"  → {result.throughput:.1f} ±{result.throughput_std:.1f} tok/s")

    del model, tokenizer
    cleanup()

    if all_results:
        print_summary(all_results)
        out_path = ROOT / "bench" / "results" / f"bench_{int(time.time())}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json_data = [{k: v for k, v in r.__dict__.items() if v is not None and v != {} and v != []}
                     for r in all_results]
        out_path.write_text(json.dumps(json_data, indent=2, default=str))
        P(f"\n  Results saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Yunshu Benchmark — lm-eval-harness compatible")
    parser.add_argument("--quick", action="store_true", help="3 runs, 50 samples")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cached results")
    parser.add_argument("--full", action="store_true", help="Full dataset (12,032 questions)")
    parser.add_argument("--samples", type=int, default=200, help="Samples (0=full)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for parallel inference")
    parser.add_argument("--bench", nargs="+", choices=BENCHMARKS + ["all"], default=["all"])
    parser.add_argument("--framework", nargs="+",
                        choices=["mlx-lm", "yunshu", "vllm-mlx", "omlx", "all"], default=["all"])
    args = parser.parse_args()
    if args.quick:
        args.samples = min(args.samples, 50)

    frameworks = set(args.framework)
    if "all" in frameworks:
        frameworks = {"mlx-lm", "yunshu", "vllm-mlx", "omlx"}

    # Sync path for mlx-lm only (avoids asyncio event loop slowing down stream_generate)
    if frameworks == {"mlx-lm"}:
        main_sync(args)
    else:
        asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
