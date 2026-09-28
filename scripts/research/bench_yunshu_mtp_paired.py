"""Paired greedy AR/MTP probe using one existing local Qwen3.8 checkpoint.

Run with the candidate mlx-vlm environment; this script does not download or
export a second model. Raw token IDs and timings are written as JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx

from yunshu_engine.mlxvlm_mtp import MLXVLMMtp

TASKS = [
    ("short_zh", "只回答數字：17 + 25 = ?"),
    ("short_en", "Reply with exactly one word: cobalt."),
    (
        "reasoning",
        "A box has 12 red and 8 blue balls. Remove 3 red and add 5 blue. Give the final red:blue ratio.",
    ),
    (
        "code",
        "Write a Python function that returns the first duplicate in a list, or None.",
    ),
    ("json", 'Return only JSON with keys "city" and "count" for Taipei and 7.'),
    ("unicode", "把『晚安，臺北！』翻譯成英文，只給譯文。"),
    ("instruction", "List three concise steps to safely close a file in Python."),
    ("math", "What is 23 * 19? Explain in one sentence."),
]


def _prompt(tokenizer, user_text: str) -> str:
    messages = [{"role": "user", "content": user_text}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )


def _run(model, prompt: str, use_mtp: bool, max_tokens: int) -> dict:
    start = time.perf_counter()
    first = None
    token_ids = []
    for token in model.iter_token_ids(
        [], max_tokens=max_tokens, temperature=0.0, prompt=prompt, use_mtp=use_mtp
    ):
        if first is None:
            first = time.perf_counter()
        token_ids.append(token)
    end = time.perf_counter()
    return {
        "mode": "mtp" if use_mtp else "ar",
        "first_token_s": round(first - start, 6) if first else None,
        "complete_s": round(end - start, 6),
        "token_ids": token_ids,
        "text": model.tokenizer.decode(token_ids),
        "completion_tokens": len(token_ids),
        "mlx_active_bytes": mx.get_active_memory(),
        "mlx_peak_bytes": mx.get_peak_memory(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="local Qwen3.5-family checkpoint directory"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()
    model = MLXVLMMtp(args.model)
    model.load()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as file:
        for round_index in range(args.rounds):
            for index, (task, user_text) in enumerate(TASKS):
                prompt = _prompt(model.tokenizer, user_text)
                order = (
                    (False, True) if (index + round_index) % 2 == 0 else (True, False)
                )
                results = {
                    use_mtp: _run(model, prompt, use_mtp, args.max_tokens)
                    for use_mtp in order
                }
                row = {
                    "round": round_index + 1,
                    "task": task,
                    "prompt": user_text,
                    "prompt_tokens": len(model._encode_text(prompt)),
                    "ar": results[False],
                    "mtp": results[True],
                    "token_equal": results[False]["token_ids"]
                    == results[True]["token_ids"],
                }
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                file.flush()
                print(
                    round_index + 1,
                    task,
                    "equal=" + str(row["token_equal"]),
                    "ar=" + str(results[False]["complete_s"]),
                    "mtp=" + str(results[True]["complete_s"]),
                    flush=True,
                )


if __name__ == "__main__":
    main()
