"""Logprobs + stop-sequence generation-contract gate (direct engine).

Two parts of the generation output contract that no other gate covers:

  - logprobs: chat(logprobs=True, top_logprobs=5) returns one entry per
    generated token, each a {token, logprob, top_logprobs:[...]} with logprob
    ≤ 0, the requested number of alternatives, and — at temperature 0 — the
    chosen token IS the argmax (its logprob == the max of its top_logprobs).
  - stop: a deterministic two-pass test. Generate at temp 0 (reproducible),
    pick a word that appears mid-output, re-generate with that word as a stop
    string, and assert the run stops early (finish_reason="stop", strictly
    shorter, and does not run past the stop point).

Run: PYTHONPATH=. uv run python scripts/verify_logprobs_stop.py
"""
from __future__ import annotations

import asyncio
import os
import sys

MODEL = os.environ.get("YUNSHU_BENCH_MODEL", "./models/Qwen3.5-0.8B-MLX-bf16")


def _txt(o):
    return (o["text"] if isinstance(o, dict) else o.text)


def _fr(o):
    return o.get("finish_reason") if isinstance(o, dict) else getattr(o, "finish_reason", None)


def _lps(o):
    return o.get("logprobs") if isinstance(o, dict) else getattr(o, "logprobs", None)


async def main() -> int:
    if not os.path.exists(MODEL):
        print("SKIP: model not mounted")
        return 0
    from yunshu_engine.batched_engine import BatchedEngine
    eng = BatchedEngine(model_name=MODEL)
    await eng.start()

    checks: dict[str, bool] = {}
    detail: list[str] = []
    try:
        # ── logprobs ────────────────────────────────────────────────────────
        o = await eng.chat(
            messages=[{"role": "user", "content": "Reply with exactly one word: hello"}],
            max_tokens=8, temperature=0.0, enable_thinking=False,
            logprobs=True, top_logprobs=5,
        )
        lps = _lps(o)
        has = bool(lps)
        checks["logprobs: present & non-empty"] = has
        if has:
            e0 = lps[0]
            lp = e0.get("logprob")
            tops = e0.get("top_logprobs") or []
            checks["logprobs: logprob ≤ 0"] = isinstance(lp, (int, float)) and lp <= 1e-6
            checks["logprobs: top_logprobs has 5 alts"] = len(tops) == 5
            # temp 0 → chosen token is the argmax → its logprob == max alt logprob
            max_alt = max((t.get("logprob", -1e9) for t in tops), default=-1e9)
            checks["logprobs: chosen==argmax at temp0"] = abs(lp - max_alt) < 1e-4
            detail.append(f"logprobs[0] token={e0.get('token')!r} lp={lp:.4f} max_alt={max_alt:.4f} nalt={len(tops)}")
        else:
            detail.append("logprobs missing")

        # ── stop sequence (deterministic two-pass) ───────────────────────────
        prompt = [{"role": "user", "content": "List the first eight planets, one per line."}]
        base = await eng.chat(messages=prompt, max_tokens=80, temperature=0.0, enable_thinking=False)
        base_txt = _txt(base)
        # pick a word that appears past the first line so a stop genuinely truncates
        words = [w.strip(".,") for w in base_txt.split() if len(w.strip(".,")) >= 4]
        stop_word = words[len(words) // 2] if words else None
        if stop_word and stop_word in base_txt:
            cut = await eng.chat(messages=prompt, max_tokens=80, temperature=0.0,
                                 enable_thinking=False, stop=[stop_word])
            cut_txt = _txt(cut)
            checks["stop: finish_reason == 'stop'"] = _fr(cut) == "stop"
            checks["stop: output strictly shorter"] = len(cut_txt) < len(base_txt)
            # the stop string must not appear past where it first occurs (engine
            # cuts at/before it — the stop token text is not emitted into output)
            checks["stop: does not run past stop word"] = stop_word not in cut_txt
            detail.append(f"stop={stop_word!r} base_len={len(base_txt)} cut_len={len(cut_txt)} fr={_fr(cut)}")
        else:
            checks["stop: usable stop word found in baseline"] = False
            detail.append(f"no usable stop word (base={base_txt[:60]!r})")
    finally:
        await eng.stop()

    for k, v in checks.items():
        print(f"  {'OK ' if v else 'BAD'} {k}")
    for d in detail:
        print(f"     {d}")
    ok = all(checks.values())
    print(f"RESULT: {sum(checks.values())}/{len(checks)} passed")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
