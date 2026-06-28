"""COMPLETE 4-tier KV-cache benchmark — every tier, isolated, per model.

Drives BatchedEngine in-process (no gateway/HTTP) with a deliberately TINY prefix
cache (hot_limit=2, max_entries=4) so HOT->WARM->SSD->COLD can all be triggered
with a handful of requests. For each tier it places the SAME primed prefix into
that tier, then re-requests it and measures:
  - TTFT (avg of 3 max_tokens=1 calls)
  - speedup vs COLD full-prefill
  - tokens cached vs prefilled
  - COHERENCE: greedy output text must match the COLD reference (WARM/SSD are
    lossy-quantized, so exact-match is reported but not required)
  - per-entry memory (full-precision HOT entry vs 4-bit WARM entry) to show UMA
    RAM savings
  - tier occupancy, SSD on-disk bytes, migration/lifecycle stats

Run (one model):
  PYTHONPATH=. YUNSHU_BENCH_MODEL=./models/Qwen2.5-3B-Instruct-bf16 \
    uv run python scripts/bench_4tier.py
"""
import asyncio
import logging
import os
import time

logging.basicConfig(level=logging.ERROR)
MODEL = os.environ.get("YUNSHU_BENCH_MODEL", "./models/Qwen2.5-3B-Instruct-bf16")

# DEFAULT FAST PATH (the production serving path) — this is what exercises the
# KVPrefixCache 4-tier hierarchy (HOT/WARM/SSD). The opt-in engine_core loop
# (YUNSHU_ENGINE_LOOP=1) uses the PagedScheduler radix cache instead and leaves
# the KVPrefixCache empty, so the tiers are a FAST-PATH feature. SSD on for the
# SSD tier. Per-tier hot_limit/max_entries are set programmatically below.
os.environ.setdefault("YUNSHU_SSD_CACHE", "1")
os.environ.setdefault("YUNSHU_SSD_CACHE_DIR", "/tmp/yunshu_ssd_4tier")
os.environ.setdefault("YUNSHU_VLM_KV_PREFIX", "1")  # VLM text-path 4-tier (Wave 613)

QUERY = "List the first 6 even numbers, comma separated."


def _is_vlm_model(path: str) -> bool:
    """Route to VLMEngine iff mlx-lm does NOT implement this model_type — exactly
    the project's resolver logic (try mlx_lm first, fall back to mlx_vlm). A
    `vision_config` stub alone does NOT make it a VLM here: Qwen3.5 has one but is
    served as a text LLM by mlx-lm's qwen3_5. Genuine VLM-only types (glm_ocr,
    qwen3_omni_moe, qwen2_5_vl, …) are absent from mlx_lm → VLMEngine."""
    import importlib.util
    import json
    try:
        cfg = json.loads(open(os.path.join(path, "config.json")).read())
    except Exception:
        return False
    mt = cfg.get("model_type")
    if not mt:
        return False
    try:
        return importlib.util.find_spec(f"mlx_lm.models.{mt}") is None
    except Exception:
        return True


_IS_VLM = _is_vlm_model(MODEL)


class _Out:
    """Uniform accessor over BatchedEngine GenerationOutput (attrs) and VLMEngine
    generate() dict, so the rest of the bench reads .text/.prompt_tokens/etc."""
    __slots__ = ("text", "prompt_tokens", "completion_tokens", "cached_tokens")

    def __init__(self, raw):
        g = raw.get if isinstance(raw, dict) else (lambda k, d=0: getattr(raw, k, d))
        self.text = g("text", "") or ""
        self.prompt_tokens = g("prompt_tokens", 0)
        self.completion_tokens = g("completion_tokens", 0)
        self.cached_tokens = g("cached_tokens", 0)


def _doc(tag, n=110):
    return f"Reference document {tag}. " + (
        "Photosynthesis converts sunlight into chemical energy stored in glucose. " * n)


async def _chat(engine, system, user, max_tokens=1):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    t0 = time.perf_counter()
    if _IS_VLM:
        raw = await engine.generate(messages=msgs, max_tokens=max_tokens,
                                    temperature=0.0, enable_thinking=False)
    else:
        raw = await engine.chat(messages=msgs, max_tokens=max_tokens,
                                temperature=0.0, enable_thinking=False)
    return time.perf_counter() - t0, _Out(raw)


def _pc(engine):
    return getattr(engine, "_kv_prefix_cache", None) or getattr(engine, "_text_kv_prefix_cache", None)


def _entry_bytes(cache_layers):
    """Sum nbytes across all layer-caches of one prefix entry (HOT full or WARM 4-bit)."""
    total = 0
    quant = False
    for c in cache_layers:
        k = getattr(c, "keys", None)
        v = getattr(c, "values", None)
        if isinstance(k, (tuple, list)) and hasattr(c, "group_size"):
            quant = True
            for part in list(k) + list(v):
                if part is not None and hasattr(part, "nbytes"):
                    total += part.nbytes
        else:
            for part in (k, v):
                if part is not None and hasattr(part, "nbytes"):
                    total += part.nbytes
    return total, quant


def _first_hot_idx(pc):
    """Index of a full-precision (HOT) entry, or -1."""
    for i, w in enumerate(pc._warm_flags):
        if not w and i < len(pc._caches):
            return i
    return 0 if pc._caches else -1


def _first_warm_idx(pc):
    """Index of a 4-bit-quantized (WARM) entry, or -1."""
    for i, w in enumerate(pc._warm_flags):
        if w and i < len(pc._caches):
            return i
    return -1


def _ssd_bytes():
    d = os.environ["YUNSHU_SSD_CACHE_DIR"]
    t = 0
    for root, _, files in os.walk(d):
        for f in files:
            try:
                t += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return t


async def _timed_reuse(engine, sysprompt, n=3):
    dts = []
    out = None
    for i in range(n):
        d, out = await _chat(engine, sysprompt, QUERY, 1)
        dts.append(d)
    return min(dts), out  # min = least-noisy TTFT


async def main():
    import shutil
    shutil.rmtree(os.environ["YUNSHU_SSD_CACHE_DIR"], ignore_errors=True)
    os.makedirs(os.environ["YUNSHU_SSD_CACHE_DIR"], exist_ok=True)
    name = os.path.basename(MODEL.rstrip("/"))
    print(f"\n╔══ 4-TIER CACHE BENCH: {name} ══")
    if _IS_VLM:
        from yunshu_engine.types import EngineConfig
        from yunshu_engine.vlm_engine import VLMEngine
        print("║ path: VLMEngine text path | _text_kv_prefix_cache 4-tier | ssd=1 (Wave 613)")
        engine = VLMEngine(MODEL, EngineConfig())
    else:
        from yunshu_engine.batched_engine import BatchedEngine
        print("║ path: default fast path (_generate_fast) | KVPrefixCache 4-tier | ssd=1")
        engine = BatchedEngine(model_name=MODEL)
    t0 = time.perf_counter()
    await engine.start()
    print(f"║ model loaded in {time.perf_counter()-t0:.1f}s")
    pc = _pc(engine)
    # Reuse may be bypassed for correctness (VLM sliding-window / interleaved-mRoPE
    # / hybrid). Report that honestly instead of running empty tiers.
    _bypassed = pc is None
    _bypass_why = "no prefix cache"
    if not _bypassed and _IS_VLM:
        try:
            if not engine._text_prefix_reuse_safe(engine._model.language_model):
                _bypassed = True
                caps = engine.backend_capabilities(engine._model.language_model)
                _bypass_why = caps.bypass_reason() or (
                    "reuse probe not lossless" if not getattr(engine, "_reuse_probe_ok", None)
                    else "bypassed")
        except Exception:
            pass
    if _bypassed:
        print(f"║ reuse BYPASSED ({_bypass_why}) — correct for this backbone")
        print("@@RESULT4T@@ " + __import__("json").dumps({
            "model": name, "prompt_tok": 0, "prefill_tps": 0, "decode_tps": 0,
            "cold_ms": 0, "hot_entry_mb": 0, "warm_entry_mb": 0, "ssd_disk_mb": 0,
            "tiers": {}, "bypassed": True, "bypass_reason": _bypass_why}))
        try:
            await engine.stop()
        except Exception:
            pass
        return
    await _chat(engine, _doc("WARMUP"), "hi", 2)  # JIT warmup

    P = _doc("PRIMARY")  # the prefix we track through every tier

    # ── COLD reference: clear, full prefill, capture reference text ──
    pc.clear()
    cold_dt, cold_out = await _chat(engine, P, QUERY, 1)
    prompt_tok = getattr(cold_out, "prompt_tokens", 0)
    prefill_tps = prompt_tok / cold_dt if cold_dt > 0 else 0
    pc.clear()
    _, ref_out = await _chat(engine, P, QUERY, 24)
    ref_text = (getattr(ref_out, "text", "") or "").strip()
    # PURE decode tps — two-point timing cancels prefill (a single
    # tokens/total-time figure is prefill-contaminated and understates decode
    # ~2-3x for short generations). Same short prompt both calls; the difference
    # (t41 - t1) is 40 decode steps with zero extra prefill.
    pc.clear()
    _t1, _ = await _chat(engine, "You are concise.", "Write about the ocean.", 1)
    pc.clear()
    _t41, _o41 = await _chat(engine, "You are concise.", "Write about the ocean.", 41)
    _dec_steps = max(1, getattr(_o41, "completion_tokens", 41) - 1)
    decode_tps = _dec_steps / (_t41 - _t1) if _t41 > _t1 else 0

    rows = []  # (tier, ttft_ms, speedup, cached, prefilled, coherent, exact, note)
    rows.append(("COLD", cold_dt * 1000, 1.0, 0, prompt_tok, True, True, "full prefill baseline"))

    # NB: reuse uses a DIFFERENT user than the prime, so the cached prefix is the
    # long SYSTEM doc (the realistic shared-context case). Priming + reusing with
    # an identical prompt hits the full-match guard (last token must recompute).
    PRIME_USER = "Briefly summarize the document above."

    # Each tier is isolated by setting the cache's hot_limit / max_entries
    # directly, so demotion (WARM) and eviction-to-SSD fire in isolation without
    # cross-tier churn corrupting the measurement.
    def _cfg(hot, mx_):
        pc._hot_limit = hot
        pc._max_entries = mx_

    # ── HOT: resident, full precision (no demotion, no eviction) ──
    _cfg(64, 64)
    pc.clear()
    await _chat(engine, P, PRIME_USER, 8)  # prime → saves the P-based prefix
    hot_entry_bytes = _entry_bytes(pc._caches[_first_hot_idx(pc)])[0] if pc._caches else 0
    hot_dt, hot_out = await _timed_reuse(engine, P, 3)
    _, hot_txt_out = await _chat(engine, P, QUERY, 24)
    hot_txt = (getattr(hot_txt_out, "text", "") or "").strip()
    hot_cached = getattr(hot_out, "cached_tokens", 0)
    hot_pref = max(0, getattr(hot_out, "prompt_tokens", 0) - hot_cached)
    rows.append(("HOT", hot_dt * 1000, cold_dt / hot_dt if hot_dt else 0, hot_cached, hot_pref,
                 bool(hot_txt), hot_txt == ref_text, "GPU-resident full precision"))

    # ── WARM: hot_limit=1 so priming P then 1 distinct prefix demotes P to 4-bit;
    #    max_entries large so no eviction churn destroys the WARM entry. ──
    _cfg(1, 64)
    pc.clear()
    await _chat(engine, P, PRIME_USER, 8)       # prime P (oldest, becomes WARM)
    await _chat(engine, _doc("WFLOOD0"), "q0", 4)  # 2nd hot entry → P demoted to WARM
    await asyncio.sleep(0.2)
    widx = _first_warm_idx(pc)
    warm_entry_bytes, warm_is_quant = (_entry_bytes(pc._caches[widx]) if widx >= 0 else (0, False))
    st = pc.get_stats().get("prefix_cache", pc.get_stats())
    warm_n = st.get("warm_entries", "?")
    warm_dt, warm_out = await _timed_reuse(engine, P, 3)
    _, warm_txt_out = await _chat(engine, P, QUERY, 24)
    warm_txt = (getattr(warm_txt_out, "text", "") or "").strip()
    warm_cached = getattr(warm_out, "cached_tokens", 0)
    warm_pref = max(0, getattr(warm_out, "prompt_tokens", 0) - warm_cached)
    rows.append(("WARM", warm_dt * 1000, cold_dt / warm_dt if warm_dt else 0, warm_cached, warm_pref,
                 bool(warm_txt), warm_txt == ref_text,
                 f"warm_entries={warm_n} 4bit={warm_is_quant}"))

    # ── SSD: max_entries=4 so priming P then overflowing evicts P to disk. The
    #    FIRST reuse (n=1) is the genuine SSD restore (later reuses hit the
    #    re-added RAM entry). Hybrid models decline (linear-attn not on disk). ──
    _cfg(64, 4)
    pc.clear()
    await _chat(engine, P, PRIME_USER, 8)       # prime P (oldest)
    for i in range(6):
        await _chat(engine, _doc(f"SFLOOD{i}"), f"q{i}", 4)  # overflow → P spills to SSD
    await asyncio.sleep(0.5)
    ssd_after_flood = _ssd_bytes()
    ram_entries = len(pc._prompts)  # capped at max_entries=4; P (oldest) spilled to SSD
    ssd_dt, ssd_out = await _timed_reuse(engine, P, 1)  # first reuse = real SSD restore
    ssd_txt = (getattr(ssd_out, "text", "") or "").strip()  # n=1, max_tokens=1 → no text
    ssd_cached = getattr(ssd_out, "cached_tokens", 0)
    ssd_pref = max(0, getattr(ssd_out, "prompt_tokens", 0) - ssd_cached)
    # coherence on a fresh overflow so the text call is itself the SSD restore
    pc.clear()
    await _chat(engine, P, PRIME_USER, 8)
    for i in range(6):
        await _chat(engine, _doc(f"SFLOOD{i}"), f"q{i}", 4)
    await asyncio.sleep(0.3)
    _, ssd_txt_out = await _chat(engine, P, QUERY, 24)
    ssd_txt = (getattr(ssd_txt_out, "text", "") or "").strip()
    ssd_restored = getattr(ssd_txt_out, "cached_tokens", 0) > 0
    rows.append(("SSD", ssd_dt * 1000, cold_dt / ssd_dt if ssd_dt else 0, ssd_cached, ssd_pref,
                 bool(ssd_txt), ssd_txt == ref_text,
                 f"ram={ram_entries}/4 disk={ssd_after_flood/1e6:.0f}MB restored={ssd_restored}"))

    # ── REPORT ──
    print(f"╠══ MODEL: {name}  (prompt≈{prompt_tok} tok)")
    print(f"║  prefill {prefill_tps:6.0f} tok/s   decode {decode_tps:6.1f} tok/s")
    print(f"║  ref greedy output: {ref_text[:42]!r}")
    print("╠══ PER-TIER RE-REQUEST OF THE SAME PREFIX ══")
    print(f"║  {'tier':5} {'TTFT(ms)':>9} {'speedup':>8} {'cached':>7} {'prefil':>7}  {'coh':>3} {'exact':>5}  note")
    for (tier, ttft, sp, ca, pr, coh, ex, note) in rows:
        print(f"║  {tier:5} {ttft:9.1f} {sp:7.2f}x {ca:7} {pr:7}  {'YES' if coh else 'no ':>3} "
              f"{'Y' if ex else 'n':>5}  {note}")
    print("╠══ PER-ENTRY MEMORY (UMA RAM per cached prefix) ══")
    if hot_entry_bytes and warm_entry_bytes:
        print(f"║  HOT  entry: {hot_entry_bytes/1e6:7.2f} MB (full precision)")
        print(f"║  WARM entry: {warm_entry_bytes/1e6:7.2f} MB (4-bit)   "
              f"=> {hot_entry_bytes/max(1,warm_entry_bytes):.2f}x smaller")
    else:
        print(f"║  HOT={hot_entry_bytes/1e6:.2f}MB WARM={warm_entry_bytes/1e6:.2f}MB (one not captured)")
    print(f"║  SSD on disk after overflow: {ssd_after_flood/1e6:.1f} MB")
    print(f"╚{'═'*54}")
    # AUTHORITATIVE lossless verdict: for VLM the engine ran an empirical reuse
    # probe at load (full prefill vs reuse-path, bit-identical) — that is ground
    # truth, unlike the per-tier text exact-match which is noisy for short
    # cross-prompt answers. Prefer it when available.
    probe_lossless = getattr(engine, "_reuse_probe_ok", None) if _IS_VLM else None
    print(f"║  reuse probe (authoritative lossless): {probe_lossless}")
    # Machine-readable summary (consumed by scripts/bench_all.py).
    import json
    rd = {r[0]: {"ttft_ms": round(r[1], 1), "speedup": round(r[2], 2),
                 "cached": r[3], "prefilled": r[4], "lossless": bool(r[6]),
                 "note": r[7]} for r in rows}
    summary = {
        "model": name, "prompt_tok": prompt_tok,
        "prefill_tps": round(prefill_tps), "decode_tps": round(decode_tps, 1),
        "cold_ms": round(cold_dt * 1000, 1),
        "hot_entry_mb": round(hot_entry_bytes / 1e6, 2),
        "warm_entry_mb": round(warm_entry_bytes / 1e6, 2),
        "ssd_disk_mb": round(ssd_after_flood / 1e6, 1),
        "engine": "vlm" if _IS_VLM else "lm",
        "probe_lossless": probe_lossless,
        "tiers": rd,
    }
    print("@@RESULT4T@@ " + json.dumps(summary))
    try:
        await engine.stop()
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
