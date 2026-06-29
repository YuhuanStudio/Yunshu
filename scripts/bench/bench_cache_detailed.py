"""Detailed per-TIER, per-MODEL KV-cache performance benchmark (in-process).

Measures, for one model with the full 4-tier engine loop:
  - COLD (full prefill): TTFT + prefill throughput + decode throughput
  - HOT reuse (prefix-cache resident): TTFT, tokens reused vs prefilled, speedup
  - Capacity stress: drive distinct long prefixes to force HOT->WARM->SSD demotion
  - Demoted reuse: re-request an early prefix; does it load back from SSD?
  - Full per-tier stats: migration counts, tier occupancy (blocks+bytes), SSD I/O

Run (one model):
  PYTHONPATH=. YUNSHU_BENCH_MODEL=./models/Qwen2.5-3B-Instruct-bf16 \
    uv run python scripts/bench_cache_detailed.py
"""
import asyncio
import logging
import os
import time

logging.basicConfig(level=logging.ERROR)
MODEL = os.environ.get("YUNSHU_BENCH_MODEL", "./models/Qwen2.5-3B-Instruct-bf16")

# Force the 4-tier engine loop + aggressive offload so demotion actually fires.
os.environ.setdefault("YUNSHU_ENGINE_LOOP", "1")
os.environ.setdefault("YUNSHU_KV_OFFLOAD", "1")
os.environ.setdefault("YUNSHU_KV_OFFLOAD_THRESHOLD", "0.02")
os.environ.setdefault("YUNSHU_KV_OFFLOAD_INTERVAL", "1")
os.environ.setdefault("YUNSHU_SSD_CACHE", "1")
os.environ.setdefault("YUNSHU_SSD_CACHE_DIR", "/tmp/yunshu_ssd_detailed")


def _mkprompt(tag, n=110):
    return f"KB doc {tag}. " + ("Photosynthesis converts sunlight into chemical energy in plant cells. " * n)


async def _chat(engine, system, user, max_tokens=1):
    t0 = time.perf_counter()
    out = await engine.chat(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens, temperature=0.0, enable_thinking=False)
    dt = time.perf_counter() - t0
    return dt, out


def _migstats(engine):
    core = getattr(engine, "_engine_core", None)
    out = {}
    if core is not None:
        lf = getattr(core, "_kv_lifecycle", None)
        if lf is not None:
            try:
                out["lifecycle"] = lf.get_stats()
            except Exception:
                pass
    try:
        kc = engine.get_kv_cache_stats()
        out["prefix_cache"] = kc.get("prefix_cache", {})
        out["paged_kv"] = kc.get("paged_kv", {})
    except Exception:
        pass
    return out


def _ssd_disk_bytes():
    d = os.environ["YUNSHU_SSD_CACHE_DIR"]
    total = 0
    for root, _, files in os.walk(d):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


async def main():
    import shutil
    shutil.rmtree(os.environ["YUNSHU_SSD_CACHE_DIR"], ignore_errors=True)
    os.makedirs(os.environ["YUNSHU_SSD_CACHE_DIR"], exist_ok=True)
    from yunshu_engine.batched_engine import BatchedEngine

    print(f"\n╔══ DETAILED CACHE BENCH: {MODEL} ══")
    print(f"║ config: ENGINE_LOOP=1 OFFLOAD=1 thresh={os.environ['YUNSHU_KV_OFFLOAD_THRESHOLD']} SSD=1")
    engine = BatchedEngine(model_name=MODEL)
    t0 = time.perf_counter()
    await engine.start()
    print(f"║ loaded in {time.perf_counter()-t0:.1f}s")
    await _chat(engine, _mkprompt("WARM"), "hi", 1)  # warmup

    # ── COLD: never-seen prefix (full prefill) ──
    cold_dt, cold_out = await _chat(engine, _mkprompt("COLD1"), "q", 1)
    pt = getattr(cold_out, "prompt_tokens", 0)
    prefill_tps = pt / cold_dt if cold_dt > 0 else 0
    # decode throughput on a fresh prefix
    dec_dt, dec_out = await _chat(engine, _mkprompt("COLDDEC"), "Write 40 words about rivers.", 40)
    dtoks = getattr(dec_out, "completion_tokens", 0)
    decode_tps = dtoks / dec_dt if dec_dt > 0 else 0

    # ── HOT reuse: prime then re-request (resident in prefix cache) ──
    sysh = _mkprompt("HOT")
    await _chat(engine, sysh, "prime", 16)  # prime (saves prefix during gen)
    hot_dts = []
    hot_out = None
    for i in range(3):
        d, hot_out = await _chat(engine, sysh, f"w{i}", 1)
        hot_dts.append(d)
    hot_dt = sum(hot_dts) / len(hot_dts)
    hot_cached = getattr(hot_out, "cached_tokens", 0)
    hot_prompt = getattr(hot_out, "prompt_tokens", 0)
    hot_prefilled = max(0, hot_prompt - hot_cached)

    # ── Capacity stress: distinct long prefixes → force HOT->WARM->SSD ──
    mig_before = _migstats(engine)
    for i in range(20):
        await _chat(engine, _mkprompt(f"FLOOD{i}"), f"r{i}", 2)
    await asyncio.sleep(0.5)
    mig_after = _migstats(engine)

    # ── Demoted reuse: re-request an early flood prefix (likely demoted) ──
    ssd_g2s_before = mig_after.get("migration", {}).get("ssd_to_gpu_count", 0)
    dem_dt, dem_out = await _chat(engine, _mkprompt("FLOOD0"), "again", 1)
    mig_final = _migstats(engine)
    ssd_g2s_after = mig_final.get("migration", {}).get("ssd_to_gpu_count", 0)
    dem_cached = getattr(dem_out, "cached_tokens", 0)

    # ── REPORT ──
    M = mig_final.get("migration", {})
    L = mig_final.get("lifecycle", {})
    PC = mig_final.get("prefix_cache", {})
    print(f"╠══ LATENCY / THROUGHPUT (prompt≈{pt} tok) ══")
    print(f"║  COLD (full prefill)   TTFT {cold_dt*1000:7.1f} ms   prefill {prefill_tps:6.0f} tok/s")
    print(f"║  decode throughput                          {decode_tps:6.1f} tok/s")
    print(f"║  HOT reuse             TTFT {hot_dt*1000:7.1f} ms   speedup {cold_dt/hot_dt:5.2f}x  "
          f"(cached {hot_cached}, prefilled {hot_prefilled})")
    print(f"║  demoted re-request    TTFT {dem_dt*1000:7.1f} ms   cached {dem_cached}  "
          f"ssd->gpu reloads this call: {ssd_g2s_after - ssd_g2s_before}")
    print(f"╠══ TIER OCCUPANCY (after 20-prefix stress) ══")
    tb = L.get("tier_blocks", {}); tby = L.get("tier_bytes", {})
    for tier in ("HOT", "WARM", "SSD", "COLD"):
        blks = tb.get(tier, 0); byts = tby.get(tier, 0)
        print(f"║  {tier:5}: {blks:5} blocks  {byts/1e6:8.2f} MB")
    print(f"║  SSD on disk: {_ssd_disk_bytes()/1e6:.2f} MB")
    print(f"╠══ MIGRATIONS (cumulative) ══")
    print(f"║  total={M.get('total_migrations',0)}  gpu→ssd={M.get('gpu_to_ssd_count',0)}  "
          f"ssd→gpu={M.get('ssd_to_gpu_count',0)}  gpu→cpu={M.get('gpu_to_cpu_count',0)}  "
          f"cpu→ssd={M.get('cpu_to_ssd_count',0)}")
    print(f"║  bytes transferred={M.get('total_bytes_transferred',0)/1e6:.2f} MB  "
          f"avg {M.get('avg_migration_time_s',0)*1000:.3f} ms  failed={M.get('failed_migrations',0)}")
    print(f"║  tier_counts(migration-tracked)={M.get('tier_counts',{})}")
    print(f"╠══ PREFIX CACHE ══")
    print(f"║  entries={PC.get('entries',0)}  cached_tokens={PC.get('total_cached_tokens',0)}  "
          f"blocks={PC.get('total_cached_blocks',0)}  lookups={PC.get('total_lookups',0)}  "
          f"hits={PC.get('total_hits',0)}")
    print(f"╚{'═'*50}")
    try:
        await engine.stop()
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
