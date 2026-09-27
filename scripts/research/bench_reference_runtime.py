"""Actual local-model runs; use separate processes/environments per backend.

Writes JSONL immediately, including failures. Does not claim cache equivalence
or benchmark validity merely because a framework returns output.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import traceback
from contextlib import suppress
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["mlx-vlm", "yunshu-vlm"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", nargs="+", dest="selected_cases")
    parser.add_argument("--apc", action="store_true")
    parser.add_argument("--prefill-step-size", type=int)
    parser.add_argument("--kv-bits", type=int)
    args = parser.parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_dir() or not (model_path / "config.json").is_file():
        parser.error(
            "--model must be an existing local model directory with config.json; downloads are disabled"
        )
    if not model_path.is_relative_to(Path("/Volumes/P5Plus")):
        parser.error(
            "Research models must use existing model directories on /Volumes/P5Plus"
        )
    args.model = str(model_path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.backend != "mlx-vlm" and (
        args.apc or args.prefill_step_size or args.kv_bits
    ):
        parser.error("APC/prefill/KV flags are only implemented by the mlx-vlm probe")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def record(row):
        with args.output.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    versions = {}
    for name in ["mlx", "mlx-lm", "mlx-vlm", "transformers", "yunshu"]:
        with suppress(importlib.metadata.PackageNotFoundError):
            versions[name] = importlib.metadata.version(name)
    record(
        dict(
            event="environment",
            backend=args.backend,
            model=args.model,
            python=platform.python_version(),
            packages=versions,
            apc=args.apc,
            prefill_step_size=args.prefill_step_size,
            kv_bits=args.kv_bits,
        )
    )
    import mlx.core as mx
    from PIL import Image, ImageDraw

    image_path = args.output.parent / "red-left-blue-right.png"
    im = Image.new("RGB", (256, 128), "blue")
    ImageDraw.Draw(im).rectangle((0, 0, 127, 127), fill="red")
    im.save(image_path)
    cases = [
        (
            "text_cold",
            "List the integers from 1 to 30, separated by commas. No explanation.",
            None,
            96,
        ),
        (
            "text_repeat",
            "List the integers from 1 to 30, separated by commas. No explanation.",
            None,
            96,
        ),
        (
            "long_prefill",
            ("The archive entry has code ALPHA and status OPEN.\n" * 300)
            + "\nWhat is the code? Reply only with the code.",
            None,
            24,
        ),
        (
            "vision",
            "Which half is red, left or right? Reply with one word.",
            str(image_path.resolve()),
            24,
        ),
    ]
    cases.insert(3, ("long_repeat", *cases[2][1:]))
    cases.insert(
        4,
        (
            "long_edited_tail",
            ("The archive entry has code ALPHA and status OPEN.\n" * 300)
            + "\nThe code changed to COBALT. What is the current code? Reply only with the code.",
            None,
            24,
        ),
    )
    long_32k = (
        "The archive entry has code ALPHA and status OPEN.\n" * 3000
    ) + "\nWhat is the code? Reply only with the code."
    cases.extend(
        [
            ("long_32k_prefill", long_32k, None, 24),
            ("long_32k_repeat", long_32k, None, 24),
            (
                "long_32k_edited_tail",
                ("The archive entry has code ALPHA and status OPEN.\n" * 3000)
                + "\nThe code changed to COBALT. What is the current code? Reply only with the code.",
                None,
                24,
            ),
            (
                "vision_after_32k",
                "Which half is red, left or right? Reply with one word.",
                str(image_path.resolve()),
                24,
            ),
        ]
    )
    if args.selected_cases:
        cases = [c for c in cases if c[0] in args.selected_cases]
        if not cases:
            parser.error("No matching cases")
    if args.backend == "mlx-vlm":
        from mlx_lm.sample_utils import make_sampler
        from mlx_vlm import load, stream_generate
        from mlx_vlm.apc import APCManager

        apc = APCManager(num_blocks=128, block_size=256) if args.apc else None
        t = time.perf_counter()
        try:
            model, processor = load(args.model)
        except Exception:
            record(dict(event="load_error", traceback=traceback.format_exc()))
            return
        record(
            dict(
                event="loaded",
                seconds=time.perf_counter() - t,
                active_bytes=mx.get_active_memory(),
            )
        )
        for name, text, image_file, maximum in cases:
            parts = [{"type": "text", "text": text}]
            if image_file:
                parts.insert(0, {"type": "image"})
            messages = [{"role": "user", "content": parts}]
            try:
                prompt = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                kwargs = dict(max_tokens=maximum, sampler=make_sampler(temp=0.0))
                if apc is not None:
                    kwargs["apc_manager"] = apc
                if args.prefill_step_size:
                    kwargs["prefill_step_size"] = args.prefill_step_size
                if args.kv_bits:
                    kwargs["kv_bits"] = args.kv_bits
                    kwargs["quantized_kv_start"] = 0
                if image_file:
                    kwargs["image"] = image_file
                mx.reset_peak_memory()
                start = time.perf_counter()
                first = None
                text_out = []
                ids = []
                last = None
                for item in stream_generate(model, processor, prompt, **kwargs):
                    last = item
                    if item.text and first is None:
                        first = time.perf_counter() - start
                    text_out.append(item.text)
                    if item.token is not None:
                        ids.append(int(item.token))
                record(
                    dict(
                        event="result",
                        case=name,
                        input_sha256=hashlib.sha256(text.encode()).hexdigest(),
                        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                        first_text_s=first,
                        wall_s=time.perf_counter() - start,
                        output="".join(text_out),
                        token_ids=ids,
                        generation_tokens=last.generation_tokens if last else 0,
                        prompt_tokens=last.prompt_tokens if last else 0,
                        finish_reason=last.finish_reason if last else None,
                        reported_prompt_tps=last.prompt_tps if last else None,
                        reported_generation_tps=last.generation_tps if last else None,
                        reported_cached_tokens=last.cached_tokens if last else None,
                        peak_bytes=mx.get_peak_memory(),
                        active_bytes=mx.get_active_memory(),
                        apc_stats=apc.stats_snapshot() if apc is not None else None,
                    )
                )
            except Exception:
                record(
                    dict(
                        event="request_error",
                        case=name,
                        traceback=traceback.format_exc(),
                    )
                )
    else:

        async def run():
            from yunshu_engine.vlm_engine import VLMEngine

            engine = VLMEngine(args.model)
            t = time.perf_counter()
            try:
                await engine.start()
                record(
                    dict(
                        event="loaded",
                        seconds=time.perf_counter() - t,
                        loaded=engine.is_loaded,
                        text_prefix_safe=engine._text_prefix_reuse_safe(
                            engine._model.language_model
                        ),
                        hybrid_prefix_safe=engine._text_hybrid_reuse_safe(
                            engine._model.language_model
                        ),
                        hybrid_probe=engine._hybrid_reuse_probe_ok,
                    )
                )
                for name, text, image_file, maximum in cases:
                    parts = [{"type": "text", "text": text}]
                    if image_file:
                        import base64

                        data = base64.b64encode(Path(image_file).read_bytes()).decode()
                        parts.insert(
                            0,
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64," + data},
                            },
                        )
                    messages = [{"role": "user", "content": parts}]
                    mx.reset_peak_memory()
                    start = time.perf_counter()
                    first = None
                    chunks = []
                    last = None
                    try:
                        async for item in engine.generate_stream(
                            messages=messages,
                            max_tokens=maximum,
                            temperature=0,
                            enable_thinking=False,
                        ):
                            last = item
                            if item.new_text and first is None:
                                first = time.perf_counter() - start
                            chunks.append(item.new_text)
                        record(
                            dict(
                                event="result",
                                case=name,
                                input_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                first_text_s=first,
                                wall_s=time.perf_counter() - start,
                                output="".join(chunks),
                                generation_tokens=last.completion_tokens if last else 0,
                                prompt_tokens=last.prompt_tokens if last else 0,
                                cached_tokens=last.cached_tokens if last else 0,
                                finish_reason=last.finish_reason if last else None,
                                engine_stats=engine.get_stats(),
                                text_kv_stats=engine._text_kv_prefix_cache.get_stats()
                                if engine._text_kv_prefix_cache is not None
                                else None,
                                peak_bytes=mx.get_peak_memory(),
                                active_bytes=mx.get_active_memory(),
                            )
                        )
                    except Exception:
                        record(
                            dict(
                                event="request_error",
                                case=name,
                                traceback=traceback.format_exc(),
                            )
                        )
            except Exception:
                record(dict(event="load_error", traceback=traceback.format_exc()))
            finally:
                await engine.stop()

        asyncio.run(run())


if __name__ == "__main__":
    main()
