"""In-process 4-tier KV prefix-reuse tracer + benchmark.

Drives BatchedEngine DIRECTLY (no gateway / HTTP / middleware) so the live
request path is unambiguous, then measures whether re-requesting a shared
prefix actually skips prefill in the engine-loop / tiered config.

Why in-process: server-side instrumentation gave empty/contradictory traces
(a request produced content with no trace through generate/_generate_fast/
scheduler.step), strongly implying an HTTP-layer response cache was masking the
path. Driving the engine directly removes that ambiguity.

Run:
  PYTHONPATH=. uv run python scripts/bench_tier_reuse.py
Env (set before run to exercise the 4-tier engine loop):
  YUNSHU_ENGINE_LOOP=1 YUNSHU_KV_OFFLOAD=1 YUNSHU_SSD_CACHE=1
  YUNSHU_SSD_CACHE_DIR=/tmp/yunshu_ssd
"""
import asyncio
import logging
import os
import threading
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")

MODEL = os.environ.get("YUNSHU_BENCH_MODEL", "./models/Qwen2.5-3B-Instruct-bf16")

# Ordered trace of method entries (name, t, thread)
_TRACE: list[tuple[str, float, str]] = []
_T0 = time.perf_counter()
_trace_on = False


def _rec(name: str) -> None:
    if _trace_on:
        _TRACE.append((name, time.perf_counter() - _T0, threading.current_thread().name))


def _patch():
    """Monkey-patch the candidate request-path methods to record entry."""
    from yunshu_engine import batched_engine as BE
    from yunshu_engine import engine_core as EC
    from yunshu_engine import scheduler as SCH

    def wrap_async(cls, meth):
        orig = getattr(cls, meth, None)
        if orig is None:
            return
        async def w(self, *a, **k):
            _rec(f"{cls.__name__}.{meth}")
            return await orig(self, *a, **k)
        setattr(cls, meth, w)

    def wrap_sync(cls, meth):
        orig = getattr(cls, meth, None)
        if orig is None:
            return
        def w(self, *a, **k):
            _rec(f"{cls.__name__}.{meth}")
            return orig(self, *a, **k)
        setattr(cls, meth, w)

    for m in ("generate", "_generate_fast", "_stream_generate_fast", "chat"):
        wrap_async(BE.BatchedEngine, m)
    wrap_async(EC.EngineCore, "generate")
    wrap_sync(SCH.Scheduler, "step")
    # PagedScheduler is optional (only with paged KV)
    try:
        from yunshu_engine import paged_scheduler as PS
        wrap_sync(PS.PagedScheduler, "step")
        wrap_sync(PS.PagedScheduler, "add_request")
    except Exception:
        pass
    # Tiered allocate (the radix prefix match)
    try:
        from yunshu_kv import tiered as TI
        wrap_sync(TI.TieredKVCacheManager, "allocate_for_prefill")
    except Exception:
        pass


def _mkprompt(tag: str, n: int = 110) -> list[dict]:
    sys = f"KB doc {tag}. " + ("Photosynthesis converts sunlight into chemical energy in plant cells. " * n)
    return sys


async def _one(engine, system: str, user: str, max_tokens: int = 1):
    global _TRACE
    _TRACE = []
    t0 = time.perf_counter()
    out = await engine.chat(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens, temperature=0.0,
    )
    dt = time.perf_counter() - t0
    pt = getattr(out, "prompt_tokens", None)
    ct = getattr(out, "cached_tokens", None)
    return dt, pt, ct, list(_TRACE)


def _cache_stats(engine):
    try:
        return engine.get_kv_cache_stats()
    except Exception as e:
        return {"error": str(e)}


async def main():
    global _trace_on
    from yunshu_engine.batched_engine import BatchedEngine
    print(f"=== in-process 4-tier reuse bench ===")
    print(f"model: {MODEL}")
    print(f"env: ENGINE_LOOP={os.environ.get('YUNSHU_ENGINE_LOOP')} "
          f"KV_OFFLOAD={os.environ.get('YUNSHU_KV_OFFLOAD')} "
          f"SSD={os.environ.get('YUNSHU_SSD_CACHE')}")
    _patch()
    engine = BatchedEngine(model_name=MODEL)
    t0 = time.perf_counter()
    await engine.start()
    print(f"engine started in {time.perf_counter()-t0:.1f}s\n")

    sysp = _mkprompt("REUSE")
    # warmup (discard)
    await _one(engine, _mkprompt("WARM"), "hi")

    _trace_on = True
    print("--- SERIAL: prime prefix (16 tok) → warm TTFT vs never-seen cold TTFT ---")
    # PRIME: generate enough tokens so the prefix is saved during generation.
    pr_dt, _, _, _ = await _one(engine, sysp, "prime please", max_tokens=16)
    print(f"PRIME (16 tok, saves prefix): {pr_dt:.3f}s")
    # WARM: same prefix, TTFT only (should skip prefill if the save worked).
    warm = []
    last_trace = None
    last_ct = None
    for i in range(3):
        dt, pt, ct, tr = await _one(engine, sysp, f"w{i}", max_tokens=1)
        warm.append(dt); last_trace = tr; last_ct = ct
    avg = sum(warm) / len(warm)
    # COLD: a brand-new prefix never seen → full prefill baseline (TTFT only).
    cold_dt, _, cct, cold_trace = await _one(engine, _mkprompt("COLDREF"), "c0", max_tokens=1)
    print(f"COLD (never-seen, TTFT): {cold_dt:.3f}s  cached_tokens={cct}")
    print(f"WARM avg (primed, TTFT): {avg:.3f}s  cached_tokens={last_ct}")
    print(f"  warm path: {[n for n,_,_ in last_trace]}")
    print(f"  >>> REUSE SPEEDUP: {cold_dt/avg:.2f}x  "
          f"{'PREFILL SKIPPED' if cold_dt/avg > 1.5 else 'NO SKIP'}")
    _trace_on = False

    # CORRECTNESS: same full prompt cold (1st, populates) vs warm (2nd, reuses).
    # temp=0 → must be IDENTICAL if prefill-skip is lossless.
    print("\n--- CORRECTNESS: cold vs warm output must be IDENTICAL (lossless) ---")
    # Substantive question + long output so a single divergent token shows up
    # (boilerplate-only outputs would falsely pass).
    csys = _mkprompt("CORRECT")
    cuser = ("Write a detailed 80-word paragraph about the water cycle, "
             "then list 5 planets. Be specific. /no_think")
    async def _gen():
        return await engine.chat(
            messages=[{"role": "system", "content": csys}, {"role": "user", "content": cuser}],
            max_tokens=96, temperature=0.0, enable_thinking=False)
    out_cold = await _gen()
    out_warm = await _gen()
    tc = getattr(out_cold, "text", None) or getattr(out_cold, "output_text", "") or ""
    tw = getattr(out_warm, "text", None) or getattr(out_warm, "output_text", "") or ""
    ok = tc == tw and len(tc) > 20
    print(f"  cold[{len(tc)}]: {tc[:70]!r}")
    print(f"  warm[{len(tw)}]: {tw[:70]!r}")
    # show first divergence if any
    if tc != tw:
        for i, (a, b) in enumerate(zip(tc, tw)):
            if a != b:
                print(f"  first diff @char {i}: cold={tc[i:i+15]!r} warm={tw[i:i+15]!r}")
                break
    print(f"  >>> LOSSLESS: {'YES ✅' if ok else 'NO ❌ — CORRUPTION!'}")

    import json
    print(f"\n--- cache stats after serial ---")
    print(json.dumps(_cache_stats(engine), indent=1, default=str)[:900])

    # CONCURRENT: fire N same-prefix requests at once to activate the loop
    print(f"\n--- CONCURRENT: 6 same-prefix requests at once ---")
    t0 = time.perf_counter()
    results = await asyncio.gather(*[
        _one(engine, sysp, f"c{i}", max_tokens=5) for i in range(6)
    ])
    cdt = time.perf_counter() - t0
    print(f"6 concurrent done in {cdt:.3f}s; per-req: {[f'{r[0]:.2f}' for r in results]}")
    print(f"  cached_tokens seen: {[r[2] for r in results]}")

    try:
        await engine.stop()
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
