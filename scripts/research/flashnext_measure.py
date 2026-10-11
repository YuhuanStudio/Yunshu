"""gpuq-only served-runner memory/TTFT/decode and modest paired accuracy probe.

Times exclude HTTP and loading. Runs both unique cold prompts and exact repeats
for APC; never describes serial quantization arms as interleaved A/B evidence.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import logging
import os
import re
import resource
import subprocess
import time
from collections import Counter
from pathlib import Path


def preflight(model):
    journal = model / "FLASHNEXT_QUANTIZATION.json"
    if journal.exists() and json.loads(journal.read_text())["status"] != "complete":
        raise ValueError(
            "conversion is unfinished; refusing to load mixed shard policy"
        )
    from yunshu_engine.model_manager import ModelType, _detect_model_type

    if _detect_model_type(str(model)) != ModelType.VLM:
        raise ValueError("expected the Yunshu VLM batch runner")


def model_processes():
    rows = subprocess.check_output(
        ["ps", "-axo", "pid,ppid,rss,command"], text=True
    ).splitlines()[1:]
    result = []
    for row in rows:
        fields = row.split(None, 3)
        if len(fields) == 4 and Path(fields[3].split()[0]).name.startswith(
            ("python", "uvicorn", "yunshu", "mlx")
        ):
            result.append(row)
    return result


def parse_choice(text, allowed):
    match = re.fullmatch(r"([A-J])", text.strip())
    value = match.group(1) if match else None
    if value not in allowed:
        raise ValueError("choice constraint did not emit a valid complete answer")
    return value


async def measure(args):
    import mlx.core as mx

    from yunshu_engine.types import EngineConfig
    from yunshu_engine.vlm_engine import VLMEngine

    preflight(args.model)
    from yunshu_engine import settings

    apc_gib = getattr(args, "apc_gib", 1.0)
    settings.set_override("YUNSHU_VLM_APC_MEMORY_GB", apc_gib)
    settings.set_override(
        "YUNSHU_VLM_APC_DISK_DIR",
        str(
            Path.home()
            / "Library/Caches/yunshu-flashnext/apc"
            / (args.model.name + "-" + getattr(args, "cache_tag", args.out.stem))
        ),
    )
    settings.set_override("YUNSHU_VLM_APC_DISK_GB", 8.0)
    engine = VLMEngine(
        str(args.model), EngineConfig(prefill_step_size=512, completion_batch_size=1)
    )
    processes = model_processes()
    if "Flash" in str(args.model) or "flashnext" in str(args.model):
        conflicting = [
            row
            for row in processes
            if "27B" in row
            and any(x in row for x in ("serve", "server", "bench", "realmodel"))
        ]
        if conflicting:
            raise RuntimeError(
                "27B-free measurement required; existing non-owned processes are not touched: "
                + repr(conflicting)
            )
    captured = {}
    original_events = engine._runner_events

    def capture_events(*positional, **keywords):
        captured["stats"] = keywords["stats"]
        yield from original_events(*positional, **keywords)

    engine._runner_events = capture_events
    started = time.perf_counter()
    await engine.start()
    load_s = time.perf_counter() - started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as stream:

        def record(row):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()

        record(
            {
                "kind": "load",
                "model": str(args.model),
                "load_seconds": load_s,
                "active_bytes": mx.get_active_memory(),
                "peak_bytes": mx.get_peak_memory(),
                "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "rss_bytes": int(
                    subprocess.check_output(
                        ["ps", "-o", "rss=", "-p", str(os.getpid())],
                        text=True,
                    ).strip()
                )
                * 1024,
                "engaged_mode": "Yunshu VLMBatchRunner; speculative off (qwen4_exp not in _SPEC_MODEL_TYPES)",
                "processes": model_processes(),
                "apc_memory_gib": apc_gib,
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ("mlx", "mlx-lm", "mlx-vlm")
                },
                "expert_quantization": dict(
                    Counter(
                        f"{module.bits}/{module.group_size}/{getattr(module, 'mode', 'affine')}"
                        for path, module in engine._model.named_modules()
                        if ".mlp.switch_mlp." in path and hasattr(module, "bits")
                    )
                ),
                "source_sha256": {
                    name: hashlib.sha256(
                        (
                            Path(__file__).resolve().parents[2] / "python" / name
                        ).read_bytes()
                    ).hexdigest()
                    for name in (
                        "yunshu_engine/vlm_engine.py",
                        "yunshu_engine/kernels/ragged_kv.py",
                    )
                },
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
            }
        )
        try:

            async def request(prompt, max_tokens, **extra):
                t0 = time.perf_counter()
                first = None
                ids = []
                content = []
                final = None
                async for output in engine.generate_stream(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0,
                    enable_thinking=False,
                    **extra,
                ):
                    if output.error:
                        raise RuntimeError(output.error)
                    if output.new_token_ids and first is None:
                        first = time.perf_counter()
                    ids.extend(output.new_token_ids)
                    content.append(output.new_text)
                    final = output
                end = time.perf_counter()
                if first is None or not ids:
                    raise RuntimeError("no generated token")
                stats = captured["stats"]
                return {
                    "runner_first_token_seconds": stats.first_token_s,
                    "runner_decode_tok_s": (stats.generated - 1)
                    / (stats.t_last - stats.t_first)
                    if stats.generated > 1 and stats.t_last > stats.t_first
                    else None,
                    "runner_generated_tokens": stats.generated,
                    "runner_used_apc": stats.used_apc,
                    "runner_used_draft": stats.used_draft,
                    "runner_spec_mode": stats.spec_mode,
                    "apc_snapshot": engine.apc_snapshot(),
                    "ttft_seconds": first - t0,
                    "wall_seconds": end - t0,
                    "decode_tok_s": (len(ids) - 1) / (end - first)
                    if len(ids) > 1
                    else None,
                    "tokens": ids,
                    "digest": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                    "text": "".join(content),
                    "prompt_tokens": final.prompt_tokens,
                    "cached_tokens": final.cached_tokens,
                    "active_bytes": mx.get_active_memory(),
                    "peak_bytes": mx.get_peak_memory(),
                    "rss_peak_bytes": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss,
                    "rss_bytes": int(
                        subprocess.check_output(
                            ["ps", "-o", "rss=", "-p", str(os.getpid())],
                            text=True,
                        ).strip()
                    )
                    * 1024,
                }

            # Tiny smoke before the timed lengths/quality loop.
            record({"kind": "smoke", **await request("Reply with exactly OK.", 4)})
            if not args.smoke_only:
                for repeat in range(getattr(args, "repeats", 3)):
                    prompt = f"Run {repeat}. Explain how a Python LRU cache works, with a short code example."
                    record(
                        {
                            "kind": "timing",
                            "repeat": repeat,
                            "cache": "cold-unique",
                            **await request(prompt, 96),
                        }
                    )
                    record(
                        {
                            "kind": "timing",
                            "repeat": repeat,
                            "cache": "warm-repeat",
                            **await request(prompt, 96),
                        }
                    )
                long_prompt = (
                    "def add(a, b):\n    return a + b\n" * 180
                ) + "\nExplain this code in one sentence."
                if not getattr(args, "no_long", False):
                    record({"kind": "prefill", **await request(long_prompt, 16)})
            if args.quality:
                correct = 0
                rows = [
                    json.loads(line) for line in args.quality.read_text().splitlines()
                ]
                for row in rows:
                    options = "\n".join(
                        f"{chr(65 + i)}. {choice}"
                        for i, choice in enumerate(row["choices"])
                    )
                    prompt = f"{row['question']}\n{options}\nAnswer with only the correct option letter."
                    allowed = [chr(65 + i) for i in range(len(row["choices"]))]
                    result = await request(
                        prompt, 8, json_schema={"type": "choice", "choices": allowed}
                    )
                    predicted = parse_choice(result["text"], allowed)
                    ok = predicted == row["answer"]
                    correct += ok
                    record(
                        {
                            "kind": "quality",
                            "id": row["id"],
                            "answer": row["answer"],
                            "predicted": predicted,
                            "correct": ok,
                            **result,
                        }
                    )
                record(
                    {
                        "kind": "quality_summary",
                        "method": "choice-constrained single letter; thinking off; greedy",
                        "correct": correct,
                        "total": len(rows),
                        "dataset_sha256": hashlib.sha256(
                            args.quality.read_bytes()
                        ).hexdigest(),
                    }
                )
            record({"complete": True})
        finally:
            await engine.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quality", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--apc-gib", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-long", action="store_true")
    parser.add_argument("--cache-tag", default="paired")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)
    preflight(args.model)
    if not args.check:
        asyncio.run(measure(args))


if __name__ == "__main__":
    main()
