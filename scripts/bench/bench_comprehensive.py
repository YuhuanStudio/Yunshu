"""Comprehensive per-model sweep — the dimensions the simple matrix lacked:
PROMPT LENGTH and CACHE-HIT-RATIO (plus TTFT / prefill tok/s / pure-decode tok/s).
Concurrency × framework lives in bench_matrix.py; cache TIERS in bench_4tier.py.

In-process, Yunshu production fast path (default serving), engine-agnostic
(BatchedEngine for LLM, VLMEngine for VLM — same routing as bench_4tier).

Run (one model):
  PYTHONPATH=. YUNSHU_BENCH_MODEL=./models/Qwen2.5-3B-Instruct-bf16 \
    uv run python scripts/bench_comprehensive.py
Sweep all (sequential subprocesses): see bench_comprehensive_all (bottom).
"""
import asyncio
import importlib.util
import json
import logging
import os
import time

logging.basicConfig(level=logging.ERROR)
MODEL = os.environ.get("YUNSHU_BENCH_MODEL", "./models/Qwen2.5-3B-Instruct-bf16")
os.environ.setdefault("YUNSHU_VLM_KV_PREFIX", "1")
os.environ.setdefault("YUNSHU_SSD_CACHE", "0")

_FILLER = "Photosynthesis converts sunlight into chemical energy stored in glucose. "


def _is_vlm(path):
    try:
        mt = json.loads(open(os.path.join(path, "config.json")).read()).get("model_type")
        return mt and importlib.util.find_spec(f"mlx_lm.models.{mt}") is None
    except Exception:
        return False


_IS_VLM = _is_vlm(MODEL)


class _Out:
    __slots__ = ("text", "prompt_tokens", "completion_tokens", "cached_tokens", "ttft_ms")

    def __init__(self, raw):
        g = raw.get if isinstance(raw, dict) else (lambda k, d=0: getattr(raw, k, d))
        self.text = g("text", "") or ""
        self.prompt_tokens = g("prompt_tokens", 0)
        self.completion_tokens = g("completion_tokens", 0)
        self.cached_tokens = g("cached_tokens", 0)
        self.ttft_ms = g("ttft_ms", 0) or 0


async def _chat(engine, system, user, max_tokens):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    t0 = time.perf_counter()
    if _IS_VLM:
        raw = await engine.generate(messages=msgs, max_tokens=max_tokens, temperature=0.0, enable_thinking=False)
    else:
        raw = await engine.chat(messages=msgs, max_tokens=max_tokens, temperature=0.0, enable_thinking=False)
    return time.perf_counter() - t0, _Out(raw)


def _doc(approx_tokens, tag=""):
    # ~0.75 token/word for this filler; over-generate then the tokenizer truncates
    reps = max(1, int(approx_tokens / 9))
    return f"Doc {tag}. " + _FILLER * reps


def _pc(engine):
    return getattr(engine, "_kv_prefix_cache", None) or getattr(engine, "_text_kv_prefix_cache", None)


async def _pure_decode_tps(engine, sysmsg, pc=None):
    """Single-point pure decode: generate N tokens, subtract prefill via the
    engine's TTFT → decode_tps = (completion-1)/(wall - ttft). Robust to unstable
    completion counts (the old two-point method broke when an instruct model
    generated different token counts per call). Tries generation-forcing prompts
    and returns None if the model won't emit enough tokens to measure."""
    for user in ("Write the numbers from 1 to 60 separated by commas, then stop.",
                 "Repeat the word 'cat' fifty times separated by spaces."):
        if pc is not None:
            pc.clear()
        dt, o = await _chat(engine, sysmsg, user, 64)
        ttft_s = (o.ttft_ms / 1000.0) if o.ttft_ms else 0.0
        dec_time = dt - ttft_s
        if o.completion_tokens >= 12 and dec_time > 0.02:
            return (o.completion_tokens - 1) / dec_time
    return None


async def main():
    name = os.path.basename(MODEL.rstrip("/"))
    print(f"\n╔══ COMPREHENSIVE SWEEP: {name} ({'VLM' if _IS_VLM else 'LLM'}) ══")
    if _IS_VLM:
        from yunshu_engine.types import EngineConfig
        from yunshu_engine.vlm_engine import VLMEngine
        engine = VLMEngine(MODEL, EngineConfig())
    else:
        from yunshu_engine.batched_engine import BatchedEngine
        engine = BatchedEngine(model_name=MODEL)
    t0 = time.perf_counter()
    await engine.start()
    print(f"║ loaded in {time.perf_counter()-t0:.1f}s")
    pc = _pc(engine)
    await _chat(engine, "warmup", "hi", 4)  # JIT

    result = {"model": name, "engine": "vlm" if _IS_VLM else "lm"}

    # ── 1. LENGTH sweep (cold single-req) ──
    print("╠══ LENGTH sweep (cold, single-request) ══")
    print(f"║  {'prompt_tok':>10} {'TTFT_ms':>9} {'prefill_t/s':>11} {'decode_t/s':>10}")
    length_rows = []
    for L in (128, 512, 1024, 2048, 4096):
        if pc is not None:
            pc.clear()
        sysmsg = _doc(L, f"L{L}")
        dt, out = await _chat(engine, sysmsg, "Summarize in one word.", 1)
        ptok = out.prompt_tokens
        prefill_tps = ptok / dt if dt > 0 else 0
        dec = await _pure_decode_tps(engine, _doc(L, f"D{L}"), pc)
        dec_str = f"{dec:.1f}" if dec is not None else "n/a"
        length_rows.append({"target": L, "prompt_tok": ptok, "ttft_ms": round(dt * 1000, 1),
                            "prefill_tps": round(prefill_tps),
                            "decode_tps": (round(dec, 1) if dec is not None else None)})
        print(f"║  {ptok:>10} {dt*1000:>9.1f} {prefill_tps:>11.0f} {dec_str:>10}")
    result["length_sweep"] = length_rows

    # ── 2. CACHE-HIT-RATIO sweep ──
    # Warm a fixed base prefix (~1024 tok). Query = base + suffix of varying length;
    # hit_ratio = base_tok / prompt_tok. Measure TTFT, cached, speedup vs cold.
    # speedup is measured vs the cold prefill of the SAME full prompt (apples to
    # apples) — NOT vs the short base alone. Dividing by the base understated
    # reuse at low hit ratio (the reuse rows prefill a base + a long suffix, so a
    # base-denominator looks <1× even though reuse beats cold-prefilling the same
    # prompt). The honest metric stays ≥1× and grows with prompt length.
    print("╠══ CACHE-HIT-RATIO sweep (reuse vs cold-prefill of a same-length prompt) ══")
    print(f"║  {'hit_ratio':>9} {'prompt_tok':>10} {'cached':>7} {'cold_ms':>9} {'reuse_ms':>9} {'speedup':>8}")
    hit_rows = []
    BASE = _doc(1024, "BASE")
    # distinct content, ~same length — used as the honest cold baseline so it can
    # NOT be served from the primed BASE cache (a same-prompt cold run leaks into
    # the reuse lookup and pins hit_ratio at 1.0).
    COLDREF = _doc(1024, "COLDREF")
    if pc is not None:
        pc.clear()
    cold_dt, cold_out = await _chat(engine, BASE, "Q-cold-unique-0.", 1)
    base_tok = cold_out.prompt_tokens
    for suffix_words in (4, 60, 250, 1100):
        suffix = "Question: " + ("explain detail " * suffix_words)
        # reuse: prime the base prefix, then issue base+suffix. As the suffix grows
        # the cached fraction (hit ratio) falls — the scenario we want to measure.
        if pc is not None:
            pc.clear()
        await _chat(engine, BASE, "Prime this document.", 4)
        dt, out = await _chat(engine, BASE, suffix, 1)
        ptok = out.prompt_tokens
        cached = out.cached_tokens
        hr = cached / ptok if ptok else 0
        # honest cold baseline: cold prefill of a DISTINCT prompt of ~the same
        # length (distinct so it can't hit the primed cache). Prefill cost is ~linear
        # in token count, so same length = fair same-prompt-cold proxy.
        if pc is not None:
            pc.clear()
        cdt, _ = await _chat(engine, COLDREF, suffix + " cold-ref unique", 1)
        speedup = cdt / dt if dt > 0 else 0  # reuse vs same-length cold prefill
        hit_rows.append({"suffix_words": suffix_words, "prompt_tok": ptok, "cached": cached,
                         "hit_ratio": round(hr, 3), "cold_same_ms": round(cdt * 1000, 1),
                         "ttft_ms": round(dt * 1000, 1), "speedup_vs_same_cold": round(speedup, 2)})
        print(f"║  {hr:>9.2f} {ptok:>10} {cached:>7} {cdt*1000:>9.1f} {dt*1000:>9.1f} {speedup:>8.2f}x")
    result["hit_ratio_sweep"] = hit_rows
    result["cold_base_tok"] = base_tok
    result["cold_base_ttft_ms"] = round(cold_dt * 1000, 1)

    # ── 3. CONCURRENCY sweep (aggregate throughput) ──
    # Fire N concurrent requests (distinct prompts) and measure aggregate
    # completion tok/s. Measured on BOTH paths so the report shows the real story:
    #  - fast path: serialises on the MLX executor (max_workers=1) → agg ≈ single
    #    stream, TTFT grows with N. The right default for single-user latency.
    #  - engine-loop (YUNSHU_ENGINE_LOOP=1): true continuous batching → agg scales.
    async def _conc_sweep(eng, tag):
        print(f"╠══ CONCURRENCY sweep ({tag}, aggregate tok/s) ══")
        print(f"║  {'N':>4} {'agg_tok/s':>10} {'mean_TTFT_ms':>12}")
        rows = []
        for N in (1, 8, 16, 32):
            prompts = [(_doc(256, f"C{N}_{i}"), f"Write 30 words, variant {i}.") for i in range(N)]
            t0 = time.perf_counter()
            outs = await asyncio.gather(*[_chat(eng, s, u, 30) for s, u in prompts])
            wall = time.perf_counter() - t0
            total_tok = sum(o.completion_tokens for _, o in outs)
            agg = total_tok / wall if wall > 0 else 0
            mean_ttft = sum(dt for dt, _ in outs) / len(outs) * 1000
            rows.append({"N": N, "agg_tps": round(agg, 1), "mean_ttft_ms": round(mean_ttft, 1),
                         "total_tok": total_tok, "wall_s": round(wall, 2)})
            print(f"║  {N:>4} {agg:>10.1f} {mean_ttft:>12.1f}")
        return rows

    result["concurrency_sweep"] = await _conc_sweep(engine, "fast path")
    # Engine-loop concurrency (the real multi-user path). LLM-only — VLM has no
    # engine-loop. Load it AFTER stopping the fast engine so we never hold two
    # model copies in memory at once (matters for 9B-4bit / 30B).
    if not _IS_VLM:
        try:
            await engine.stop()
        except Exception:
            pass
        os.environ["YUNSHU_ENGINE_LOOP"] = "1"
        from yunshu_engine.batched_engine import BatchedEngine as _BE
        engine = _BE(model_name=MODEL)
        await engine.start()
        await _chat(engine, "warmup", "hi", 4)
        result["concurrency_sweep_loop"] = await _conc_sweep(engine, "engine-loop")

    print(f"╚{'═'*54}")
    print("@@COMPREHENSIVE@@ " + json.dumps(result))
    try:
        await engine.stop()
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
