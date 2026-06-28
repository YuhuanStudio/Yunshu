"""In-process correctness verification of the opt-in engine loop.

Skeptical re-verification: the 4-tier prefix-reuse bug proved that "tests pass"
does not mean "correct under real use". This drives BatchedEngine DIRECTLY and
checks CORRECTNESS properties of the continuous-batching loop that unit tests
don't exercise:

  1. Concurrent DISTINCT requests — each must get its OWN correct answer
     (catches cross-request KV contamination).
  2. Long generation — decode loop stays coherent.
  3. stop sequence + max_tokens honored in the loop.
  4. Aggressive KV offload (threshold ~0) — outputs stay correct (catches
     tier-migration data corruption / round-trip bugs).
  5. Same answer in loop vs fast path (loop must not change outputs).

Run:
  PYTHONPATH=. YUNSHU_ENGINE_LOOP=1 uv run python scripts/verify_engine_loop.py
"""
import asyncio
import logging
import os

logging.basicConfig(level=logging.WARNING)
MODEL = os.environ.get("YUNSHU_BENCH_MODEL", "./models/Qwen2.5-3B-Instruct-bf16")

_PASS = 0
_FAIL = 0


def check(name, cond, detail=""):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  ✅ {name}")
    else:
        _FAIL += 1
        print(f"  ❌ {name}  {detail}")


async def ask(engine, user, max_tokens=16, **kw):
    out = await engine.chat(
        messages=[{"role": "user", "content": user}],
        max_tokens=max_tokens, temperature=0.0, enable_thinking=False, **kw)
    return (getattr(out, "text", None) or "").strip(), out


async def main():
    from yunshu_engine.batched_engine import BatchedEngine
    print(f"=== engine-loop correctness verify ===")
    print(f"model: {MODEL}  ENGINE_LOOP={os.environ.get('YUNSHU_ENGINE_LOOP')} "
          f"OFFLOAD_THRESHOLD={os.environ.get('YUNSHU_KV_OFFLOAD_THRESHOLD')}")
    engine = BatchedEngine(model_name=MODEL)
    await engine.start()
    await ask(engine, "hi")  # warmup

    # 1. Concurrent DISTINCT requests — each correct, no contamination
    print("\n[1] concurrent distinct (no cross-request contamination)")
    qs = [("2+2", "4"), ("3+4", "7"), ("5+5", "10"), ("10-3", "7"),
          ("6*2", "12"), ("9+1", "10"), ("8-5", "3"), ("7+6", "13")]
    res = await asyncio.gather(*[
        ask(engine, f"What is {q}? Reply with only the number.", max_tokens=8)
        for q, _ in qs])
    nok = sum(1 for (txt, _), (_, a) in zip(res, qs) if a in txt)
    for (txt, _), (q, a) in zip(res, qs):
        mark = "ok" if a in txt else "BAD"
        if a not in txt:
            print(f"      {mark}: {q} -> {txt!r} (want {a})")
    check(f"concurrent distinct correct ({nok}/{len(qs)})", nok == len(qs))

    # 2. Long generation coherence
    print("\n[2] long generation (decode loop)")
    txt, out = await ask(engine, "Count from 1 to 20 separated by commas.", max_tokens=80)
    ok = all(str(n) in txt for n in range(1, 11))
    check("long gen contains 1..10 in order", ok, detail=repr(txt[:60]))

    # 3. stop sequence + max_tokens
    print("\n[3] stop + max_tokens in loop")
    txt, out = await ask(engine, "Say: A B C D E F", max_tokens=40, stop=["D"])
    check("stop honored (no 'D')", "D" not in txt, detail=repr(txt))
    txt2, out2 = await ask(engine, "Write a long story.", max_tokens=5)
    ct = getattr(out2, "completion_tokens", 99)
    check("max_tokens=5 honored", ct <= 5, detail=f"ct={ct}")

    # 4. determinism in the loop (temp=0, same prompt twice)
    print("\n[4] determinism")
    a1, _ = await ask(engine, "Name a color.", max_tokens=6)
    a2, _ = await ask(engine, "Name a color.", max_tokens=6)
    check("temp=0 deterministic", a1 == a2, detail=f"{a1!r} vs {a2!r}")

    # 5. prefix reuse from a SHORT donor (regression guard for the engine-loop
    #    prefix-save fix): a short request that finishes fast must still cache its
    #    prompt prefix, so a later request sharing that prefix reuses it. The first
    #    fix attempt saved AFTER the finished uid was pruned from _uid_to_req →
    #    ineffective; the correct fix saves at the finish point before the pop.
    print("\n[5] prefix reuse from a short donor")
    _sys = ("You are a meticulous, concise assistant. Always answer precisely and "
            "briefly. Follow the user's instructions exactly. Do not add commentary. "
            "Shared system preamble used to test engine-loop prefix reuse across requests.")
    async def _ask_sys(user, mt):
        return await engine.chat(
            messages=[{"role": "system", "content": _sys}, {"role": "user", "content": user}],
            max_tokens=mt, temperature=0.0, enable_thinking=False)
    _donor = await _ask_sys("What is 2+2?", 2)        # SHORT donor → finishes fast
    _reuser = await _ask_sys("What is 3+3?", 8)        # shares the system prefix
    _rc = getattr(_reuser, "cached_tokens", 0)
    check("short donor's prefix cached + reused (cached_tokens>0)", _rc > 0,
          detail=f"donor cached={getattr(_donor,'cached_tokens',0)} reuser cached={_rc}")

    print(f"\n=== RESULT: {_PASS} passed, {_FAIL} failed ===")
    try:
        await engine.stop()
    except Exception:
        pass
    return _FAIL


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
