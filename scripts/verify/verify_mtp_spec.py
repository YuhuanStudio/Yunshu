"""Native MTP gate on the served VLM runner; run only through gpuq.

Uses one loaded target and its invariant kernels for MTP on/off. Missing native
heads fail closed; token identities, not prose coherence, determine parity.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path


async def verify(engine) -> bool:
    runner = engine._batch_runner
    if runner is None or runner.drafter is None or runner.draft_kind != "mtp":
        raise RuntimeError("native MTP is not engaged in the VLM batch runner")
    drafter = runner.drafter
    ok = True
    try:
        for prompt in (
            "In one sentence, what lives in the ocean?",
            "Write a Python Fibonacci function.",
        ):
            arms = []
            for enabled in (False, True):
                runner.drafter = drafter if enabled else None
                tokens = []
                finished = False
                async for output in engine.generate_stream(
                    prompt=prompt,
                    max_tokens=48,
                    temperature=0.0,
                    enable_thinking=False,
                ):
                    tokens.extend(output.new_token_ids or [])
                    finished = output.finished
                    if output.error:
                        raise RuntimeError(output.error)
                if not finished or not tokens:
                    raise RuntimeError("incomplete VLM token stream")
                arms.append(tokens)
            equal = arms[0] == arms[1]
            print(f"{'OK' if equal else 'BAD'} native MTP token parity: {prompt!r}")
            ok &= equal
    finally:
        runner.drafter = drafter
    print("PASS" if ok else "FAIL")
    return ok


async def main(model: str) -> int:
    if not Path(model).is_dir():
        print("SKIP: model not mounted")
        return 0
    from yunshu_engine import settings
    from yunshu_engine.vlm_engine import VLMEngine

    settings.set_override("YUNSHU_VLM_DRAFT", "mtp")
    engine = VLMEngine(model)
    try:
        await engine.start()
        return 0 if await verify(engine) else 1
    finally:
        await engine.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./models/Jundot/Qwen3.8-27B-oQ4e-mtp")
    raise SystemExit(asyncio.run(main(ap.parse_args().model)))
