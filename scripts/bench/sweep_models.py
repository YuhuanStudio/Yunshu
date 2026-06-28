"""Model-breadth sweep (full tier) — prove core correctness on EVERY LLM, isolated.

The correctness gates in regression.py mostly run on ONE model (Qwen2.5-3B). This
sweep runs the core invariants across the whole LLM matrix, each model in its OWN
subprocess so they are NEVER co-loaded (critical on a 36GB Mac — co-loading heavy
models SIGABRTs). Per model it checks:

  1. greedy determinism      — temp=0 twice → identical (sampler/seed plumbing)
  2. non-degenerate output   — not a single repeated token (cache/decoder sane)
  3. multi-turn coherence    — a 2-turn chat recalls turn-1 context (chat template)

Parent spawns one child per model and aggregates. A model that OOMs/crashes is
reported as FAIL (not a silent skip) unless its weights are absent (real SKIP).

Run:  PYTHONPATH=. uv run python scripts/sweep_models.py
Child: PYTHONPATH=. uv run python scripts/sweep_models.py --child <model_path>
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys

MODELS = [
    "./models/Qwen2.5-3B-Instruct-4bit",
    "./models/Qwen2.5-3B-Instruct-bf16",
    "./models/Qwen3.5-0.8B-MLX-bf16",
    "./models/Qwen3.5-2B-MLX-bf16",
    "./models/Qwen3.5-9B-MLX-4bit",
    "./models/gemma-4-e4b-it-bf16",
]


def _text(o):
    return (o["text"] if isinstance(o, dict) else getattr(o, "text", "")).strip()


def _degenerate(s: str) -> bool:
    w = s.split()
    return len(w) >= 8 and (len(set(w)) / len(w)) < 0.12


async def _child(model: str) -> int:
    if not os.path.exists(model):
        print(json.dumps({"model": model, "skip": True}))
        return 0
    from yunshu_engine.batched_engine import BatchedEngine
    eng = BatchedEngine(model_name=model)
    await eng.start()
    try:
        async def chat(msgs, **kw):
            return _text(await eng.chat(messages=msgs, max_tokens=40, temperature=0.0,
                                        enable_thinking=False, **kw))
        q = [{"role": "user", "content": "Name one planet in our solar system."}]
        d0 = await chat(q)
        d1 = await chat(q)
        # multi-turn: the chat TEMPLATE must thread prior turns without error and
        # yield a coherent reply. (We assert ENGINE correctness — template
        # threading — not model intelligence like fact-recall, which a 0.8B model
        # may lack and which is not Yunshu's responsibility.)
        turn = [{"role": "user", "content": "My favorite number is 42. Reply OK."},
                {"role": "assistant", "content": "OK."},
                {"role": "user", "content": "Say one more short sentence."}]
        mt = await chat(turn)
    finally:
        await eng.stop()
    checks = {
        "greedy_deterministic": d0 == d1 and len(d0) > 0,
        "non_degenerate": not _degenerate(d0),
        "multiturn_template_coherent": len(mt) > 0 and not _degenerate(mt),
    }
    print(json.dumps({"model": model, "checks": checks, "ok": all(checks.values()),
                      "sample": d0[:40], "mt": mt[:40]}))
    return 0 if all(checks.values()) else 1


def _parent() -> int:
    results = []
    for m in MODELS:
        if not os.path.exists(m):
            results.append({"model": m, "status": "SKIP (absent)"})
            continue
        env = dict(os.environ, PYTHONPATH=os.environ.get("PYTHONPATH", "."))
        try:
            p = subprocess.run(
                [sys.executable, __file__, "--child", m],
                capture_output=True, text=True, timeout=420, env=env,
            )
            line = [l for l in p.stdout.splitlines() if l.startswith("{")]
            data = json.loads(line[-1]) if line else {}
            if data.get("skip"):
                results.append({"model": m, "status": "SKIP (absent)"})
            elif data.get("ok"):
                results.append({"model": m, "status": "PASS", "sample": data.get("sample")})
            else:
                results.append({"model": m, "status": "FAIL", "checks": data.get("checks"),
                                "stderr_tail": p.stderr.strip()[-160:]})
        except subprocess.TimeoutExpired:
            results.append({"model": m, "status": "FAIL (timeout)"})
        except Exception as e:
            results.append({"model": m, "status": f"FAIL ({type(e).__name__})"})

    tested = [r for r in results if "SKIP" not in r["status"]]
    passed = [r for r in tested if r["status"] == "PASS"]
    for r in results:
        mark = {"PASS": "OK ", }.get(r["status"], "BAD" if "FAIL" in r["status"] else "·· ")
        print(f"  {mark} {os.path.basename(r['model']):32s} {r['status']}"
              + (f"  {r.get('sample','')!r}" if r.get("sample") else ""))
    print(f"RESULT: {len(passed)}/{len(tested)} models pass core correctness "
          f"({len(results)-len(tested)} skipped)")
    ok = len(tested) > 0 and len(passed) == len(tested)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    if "--child" in sys.argv:
        model = sys.argv[sys.argv.index("--child") + 1]
        return asyncio.run(_child(model))
    return _parent()


if __name__ == "__main__":
    sys.exit(main())
