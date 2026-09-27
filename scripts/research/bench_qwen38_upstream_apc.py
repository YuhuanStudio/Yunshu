"""Paired Qwen3.8 multi-turn APC probe using an existing local checkpoint."""

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
    parser.add_argument("--apc", choices=("on", "off"), required=True)
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--memory-max-gb", type=float)
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
        from mlx_vlm.apc import APCManager, semantic_extra_hash
        from mlx_vlm.generate.ar import BatchGenerator

        from yunshu_engine.mrope import clear_rope_state
        from yunshu_engine.vlm_engine import VLMEngine

        engine = VLMEngine(str(model))
        await engine.start()
        try:

            def probe() -> None:
                apc = (
                    APCManager(
                        num_blocks=512,
                        block_size=16,
                        disk=None,
                        overrides=(
                            {"memory_max_gb": args.memory_max_gb}
                            if args.memory_max_gb is not None
                            else None
                        ),
                    )
                    if args.apc == "on"
                    else None
                )
                gen = BatchGenerator(
                    engine._model.language_model,
                    engine._processor,
                    max_tokens=24,
                    apc_manager=apc,
                    greedy_sampling=True,
                    compute_logprobs=False,
                    prefill_step_size=256,
                )
                try:
                    semantic_hash = semantic_extra_hash(
                        image_hash=0,
                        media={"audio": None, "video": None},
                        model=engine._model.language_model,
                        processor=engine._processor,
                    )
                    history = []
                    base = "The archive record is neutral background text.\n" * 300
                    with args.output.open("w") as file:
                        for turn in range(args.turns):
                            prepare_start = time.perf_counter()
                            expected = "ALPHA" if turn < args.turns // 2 else "COBALT"
                            if turn == 0:
                                user = (
                                    base
                                    + "\nThe current code is ALPHA. Reply only with the current code."
                                )
                            elif turn == args.turns // 2:
                                user = "Update the current code to COBALT. Reply only with the current code."
                            else:
                                user = "What is the current code? Reply only with the current code."
                            messages = history + [{"role": "user", "content": user}]
                            ids = engine._tokenize_with_cache(
                                messages, enable_thinking=False
                            ).tolist()
                            clear_rope_state(engine._model)
                            embeds = engine._model.get_input_embeddings(
                                mx.array(ids)[None], None, mask=None
                            )
                            prepare_s = time.perf_counter() - prepare_start
                            mx.reset_peak_memory()
                            before = (
                                apc.stats.snapshot(apc.num_blocks, apc.block_size)
                                if apc is not None
                                else None
                            )
                            prompt_kwargs = embeds.to_dict()
                            prompt_kwargs["_apc_semantic_hash"] = semantic_hash
                            uid = gen.insert(
                                [ids],
                                max_tokens=24,
                                prompt_kwargs=[prompt_kwargs],
                            )[0]
                            start = time.perf_counter()
                            first = None
                            token_ids = []
                            finish = None
                            for _ in range(500):
                                _, responses = gen.next()
                                for response in responses:
                                    if response.uid != uid:
                                        continue
                                    if first is None:
                                        first = time.perf_counter() - start
                                    token_ids.append(response.token)
                                    finish = response.finish_reason or finish
                                if finish is not None:
                                    break
                            text = engine._tokenizer.decode(
                                token_ids, skip_special_tokens=True
                            )
                            after = (
                                apc.stats.snapshot(apc.num_blocks, apc.block_size)
                                if apc is not None
                                else None
                            )
                            row = {
                                "mode": args.apc,
                                "memory_max_gb": args.memory_max_gb,
                                "turn": turn + 1,
                                "expected": expected,
                                "text": text,
                                "correct": text.strip() == expected,
                                "prompt_tokens": len(ids),
                                "prepare_s": prepare_s,
                                "first_token_s": first,
                                "complete_s": time.perf_counter() - start,
                                "first_total_s": prepare_s + first if first else None,
                                "peak_bytes": mx.get_peak_memory(),
                                "active_bytes": mx.get_active_memory(),
                                "finish_reason": finish,
                                "token_ids": token_ids,
                                "apc_before": before,
                                "apc_after": after,
                            }
                            file.write(json.dumps(row, ensure_ascii=False) + "\n")
                            file.flush()
                            print(
                                args.apc,
                                turn + 1,
                                row["correct"],
                                round(first or 0, 3),
                                after["matched_tokens"] if after else None,
                                flush=True,
                            )
                            history.extend(
                                [
                                    {"role": "user", "content": user},
                                    {"role": "assistant", "content": expected},
                                ]
                            )
                finally:
                    gen.close()

            await asyncio.get_running_loop().run_in_executor(engine._executor, probe)
        finally:
            await engine.stop()

    asyncio.run(run())


if __name__ == "__main__":
    main()
