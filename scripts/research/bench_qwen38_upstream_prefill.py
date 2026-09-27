"""Compare upstream mlx-vlm chunked prefill on one existing Qwen3.8 model."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--prefix-repeats", type=int, default=300)
    parser.add_argument(
        "--prompt-kind", choices=("history", "synthetic_raw"), default="history"
    )
    args = parser.parse_args()
    model = args.model.expanduser().resolve()
    if (
        not model.is_relative_to(Path("/Volumes/P5Plus/models"))
        or not (model / "config.json").is_file()
    ):
        parser.error("Use an existing model under /Volumes/P5Plus/models")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    async def run() -> None:
        import mlx.core as mx
        from mlx_vlm.generate.ar import generate_step

        from yunshu_engine.mrope import clear_rope_state
        from yunshu_engine.vlm_engine import VLMEngine

        engine = VLMEngine(str(model))
        await engine.start()
        try:
            messages = [
                {
                    "role": "user",
                    "content": (
                        "The archive record is neutral background text.\n"
                        * args.prefix_repeats
                    )
                    + "\nThe current code is ALPHA. Reply only with the current code.",
                }
            ]
            if args.prompt_kind == "synthetic_raw":
                token_ids = engine._tokenizer.encode(
                    "The quick brown fox jumps over the lazy dog. " * 60
                )
            else:
                token_ids = engine._tokenize_with_cache(
                    messages, enable_thinking=False
                ).tolist()

            def one(step: int | None, round_index: int) -> dict:
                ids = mx.array(token_ids)[None]
                clear_rope_state(engine._model)
                mx.reset_peak_memory()
                start = time.perf_counter()
                tokens = []
                first = None
                for item in generate_step(
                    ids,
                    engine._model,
                    None,
                    None,
                    max_tokens=24,
                    temperature=0,
                    prefill_step_size=step,
                ):
                    if first is None:
                        first = time.perf_counter() - start
                    token = item[0] if isinstance(item, tuple) else item
                    tokens.append(
                        int(token.item() if hasattr(token, "item") else token)
                    )
                    if tokens[-1] in engine._get_eos_ids():
                        break
                return {
                    "round": round_index,
                    "prompt_kind": args.prompt_kind,
                    "prefill_step_size": step,
                    "prompt_tokens": int(ids.shape[1]),
                    "first_token_s": first,
                    "complete_s": time.perf_counter() - start,
                    "token_ids": tokens,
                    "text": engine._tokenizer.decode(tokens, skip_special_tokens=True),
                    "peak_bytes": mx.get_peak_memory(),
                }

            with args.output.open("w") as file:
                for round_index in range(args.rounds):
                    for step in (None, 256) if round_index % 2 == 0 else (256, None):
                        row = await asyncio.get_running_loop().run_in_executor(
                            engine._executor, one, step, round_index
                        )
                        file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        file.flush()
                        print(
                            round_index,
                            step,
                            round(row["first_token_s"], 3),
                            round(row["complete_s"], 3),
                            repr(row["text"]),
                            flush=True,
                        )
        finally:
            await engine.stop()

    asyncio.run(run())


if __name__ == "__main__":
    main()
