"""Where the warm / turn-2 TTFT goes at long context: a timeline of the admit path, APC lookup / restore /
checkpoint stores and prefill steps for cold -> warm -> turn-2 requests, in-process on the real engine.

    turn2_timeline.py --ctx 32768 [--kind prose] [--out F.jsonl]
"""

import argparse
import asyncio
import importlib
import json
import sys
import threading
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
PROMPTS = Path("/Volumes/P5Plus/yunshu-build/tfnew/prompts")
M = "/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
EV: list = []
T0 = [0.0]

TARGETS = [
    ("yunshu_engine.vlm_batch_runner", "VLMBatchRunner._admit"),
    ("yunshu_engine.vlm_batch_runner", "VLMBatchRunner._step_generator"),
    ("mlx_vlm.generate.ar", "BatchGenerator.insert"),
    ("mlx_vlm.generate.ar", "BatchGenerator._next"),
    ("mlx_vlm.generate.ar", "PromptProcessingBatch.prompt_step"),
    ("mlx_vlm.generate.ar", "PromptProcessingBatch._store_apc_exact_checkpoints"),
    ("mlx_vlm.apc", "APCManager.lookup_exact_cache"),
    ("mlx_vlm.apc", "APCManager.store_exact_cache"),
    ("mlx_vlm.apc", "APCCoordinator.store_checkpoint"),
    ("mlx_vlm.apc", "APCCoordinator.observe_cache"),
    ("mlx_vlm.apc", "make_warm_batch_exact_cache_multi"),
    ("mlx_vlm.apc", "_clone_prompt_cache_for_apc"),
]


def wrap(modname, path):
    try:
        mod = importlib.import_module(modname)
        obj = mod
        parts = path.split(".")
        for p in parts[:-1]:
            obj = getattr(obj, p)
        fn = getattr(obj, parts[-1])
    except Exception:
        return
    name = path

    def run(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            EV.append((name, t - T0[0], time.perf_counter() - t))

    setattr(obj, parts[-1], run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--kind", default="prose")
    ap.add_argument("--out")
    a = ap.parse_args()
    from yunshu_engine.vlm_engine import VLMEngine

    engine = VLMEngine(M)
    asyncio.run(engine.start())
    runner = engine._batch_runner
    for m, p in TARGETS:
        wrap(m, p)
    text = (PROMPTS / f"{a.kind}-{a.ctx}.txt").read_text()
    kw = {"enable_thinking": False}
    reply = ""

    def run(msgs, label):
        ids, pkw, salt = engine._executor.submit(
            engine._runner_input, msgs, [], [], False, kw
        ).result()
        EV.clear()
        out, first = [], []
        T0[0] = t0 = time.perf_counter()

        def consume():
            for tok in runner.iter_tokens(
                ids,
                max_tokens=256,
                temperature=0.0,
                prompt_kwargs=pkw,
                apc_semantic_hash=salt,
            ):
                if not first:
                    first.append(time.perf_counter() - t0)
                out.append(tok)

        th = threading.Thread(target=consume)
        th.start()
        th.join()
        rec = dict(
            label=label,
            prompt_tokens=len(ids),
            ttft=round(first[0], 3),
            events=[(n, round(s, 3), round(d, 3)) for n, s, d in EV],
        )
        print(json.dumps(rec), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return out

    msgs = [{"role": "user", "content": text}]
    toks = run(msgs, "cold")
    reply = engine._tokenizer.decode([t for t in toks if isinstance(t, int)]) or "ok"
    run(msgs, "warm")
    run(msgs, "warm2")
    msgs2 = msgs + [
        {"role": "assistant", "content": reply},
        {"role": "user", "content": "Continue with the next part, at the same length."},
    ]
    run(msgs2, "turn2")
    run(msgs2, "turn2-repeat")
    mx.synchronize()


if __name__ == "__main__":
    main()
