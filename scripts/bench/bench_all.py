"""Unified KV-cache benchmark — ONE table, ALL models, ALL tiers, BOTH Yunshu
paths, plus a live oMLX head-to-head. Everything runs SEQUENTIALLY (each measure
is its own subprocess, never concurrent) so numbers don't interfere.

Per model it runs, in order:
  1. Yunshu fast path 4-tier  (scripts/bench/bench_4tier.py        -> @@RESULT4T@@)
  2. Yunshu engine-loop radix (scripts/_bench_engineloop.py  -> @@RESULTEL@@)
  3. oMLX head-to-head        (scripts/_bench_omlx.py         -> @@RESULTOMLX@@)

Run:
  PYTHONPATH=. uv run python scripts/bench_all.py
  PYTHONPATH=. uv run python scripts/bench_all.py ./models/Qwen3.5-2B-MLX-bf16 ...
"""
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# arch tag: standard=full-attn LLM, hybrid=GatedDeltaNet LLM, vlm-*=multimodal
# (routed through VLMEngine text path; -mrope/-sw/-hybrid notes the cache class).
DEFAULT_MODELS = [
    ("./models/Qwen3.5-0.8B-MLX-bf16", "hybrid"),
    ("./models/Qwen3.5-2B-MLX-bf16", "hybrid"),
    ("./models/Qwen3.5-9B-MLX-4bit", "hybrid"),
    ("./models/Qwen2.5-3B-Instruct-bf16", "standard"),
    ("./models/gemma-4-e4b-it-bf16", "gemma4"),
    ("./models/GLM-OCR-bf16", "vlm-mrope"),
    ("./models/Qwen3-Omni-30B-A3B-Instruct-4bit", "vlm-imrope"),
]


def _run(cmd, env, tag):
    """Run a sub-bench to completion (blocking) and parse its @@tag@@ json line."""
    try:
        p = subprocess.run(cmd, env=env, cwd=REPO, capture_output=True, text=True, timeout=1200)
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT"}
    out = p.stdout + p.stderr
    m = re.search(rf"@@{tag}@@ (\{{.*\}})", out)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return {"status": "PARSE ERR"}
    tail = "\n".join(out.strip().splitlines()[-3:])
    return {"status": f"NO RESULT ({tail[:90]})"}


def _is_vlm_path(path):
    """Route to VLMEngine iff mlx-lm lacks the model_type (the resolver's rule).
    A vision_config stub alone is NOT enough (Qwen3.5 has one but is a text LLM)."""
    import importlib.util
    try:
        cfg = json.loads(open(os.path.join(REPO, path, "config.json")).read())
    except Exception:
        return False
    mt = cfg.get("model_type")
    if not mt:
        return False
    try:
        return importlib.util.find_spec(f"mlx_lm.models.{mt}") is None
    except Exception:
        return True


def bench_model(path, arch):
    name = os.path.basename(path.rstrip("/"))
    base = dict(os.environ, PYTHONPATH=".")
    # VLM models route the text path through VLMEngine; the engine-loop (radix)
    # and oMLX head-to-head are BatchedEngine/LLM-only, so skip them for VLMs.
    is_vlm = _is_vlm_path(path)  # resolver truth (mlx_lm lacks the type)
    # mlx-vlm must be importable for VLM models.
    if is_vlm and "reference/mlx-vlm" not in base.get("PYTHONPATH", ""):
        base["PYTHONPATH"] = "reference/mlx-vlm:."
    ssd = f"/tmp/benchall_{name}"
    subprocess.run(["rm", "-rf", ssd], cwd=REPO)

    print(f"  [{name}] {'1/1' if is_vlm else '1/3'} {'VLM ' if is_vlm else 'fast-path '}4-tier ...", flush=True)
    r4 = _run([sys.executable, "scripts/bench/bench_4tier.py"],
              dict(base, YUNSHU_BENCH_MODEL=path, YUNSHU_SSD_CACHE_DIR=ssd + "_ft"), "RESULT4T")
    if is_vlm:
        subprocess.run(["rm", "-rf", ssd + "_ft"], cwd=REPO)
        return {"name": name, "arch": arch, "ft": r4,
                "el": {"status": "n/a (VLM)"}, "omlx": {"status": "n/a (VLM)"}}
    print(f"  [{name}] 2/3 engine-loop radix ...", flush=True)
    rel = _run([sys.executable, "scripts/bench/_bench_engineloop.py"],
               dict(base, YBENCH_MODEL=path, YUNSHU_SSD_CACHE_DIR=ssd + "_el"), "RESULTEL")
    print(f"  [{name}] 3/3 oMLX head-to-head ...", flush=True)
    # Prefer oMLX's NATIVE venv (its exact pinned mlx-lm/mlx-vlm) for a correct,
    # artifact-free comparison: set OMLX_PYTHON=/path/to/omlxenv/bin/python and
    # the no-stub native bench is used. Falls back to the in-env stubbed bench.
    omlx_py = os.environ.get("OMLX_PYTHON")
    if omlx_py:
        romlx = _run([omlx_py, "scripts/bench/_bench_omlx_native.py"],
                     dict(base, PYTHONPATH="./reference/omlx", OMLX_MODEL=path,
                          OMLX_SSD=ssd + "_omlx"), "RESULTOMLX")
    else:
        romlx = _run([sys.executable, "scripts/bench/_bench_omlx.py"],
                     dict(base, PYTHONPATH=".:./reference/omlx", OMLX_MODEL=path,
                          OMLX_SSD=ssd + "_omlx"), "RESULTOMLX")
    subprocess.run(["rm", "-rf", ssd + "_ft", ssd + "_el", ssd + "_omlx"], cwd=REPO)
    return {"name": name, "arch": arch, "ft": r4, "el": rel, "omlx": romlx}


def _tier(ft, t, k, fmt, default="—"):
    try:
        return fmt(ft["tiers"][t][k])
    except Exception:
        return default


def render(rows):
    H = ["Model", "Arch", "pTok", "pf t/s", "dec t/s", "COLD ms",
         "F-HOT", "F-WARM", "F-SSD", "WARMram", "LOOP", "oMLX", "loss/note"]
    W = [30, 10, 5, 6, 7, 8, 7, 7, 8, 7, 7, 8, 20]

    def fmt_row(cells):
        return " ".join(str(c)[:w].ljust(w) for c, w in zip(cells, W, strict=False))

    print("\n" + "=" * 118)
    print("UNIFIED KV-CACHE BENCHMARK — all sequential (no concurrent runs)")
    print("=" * 130)
    print(fmt_row(H))
    print("-" * 130)
    for r in rows:
        ft, el, om = r["ft"], r["el"], r["omlx"]
        # Reuse intentionally bypassed (VLM sliding-window / interleaved-mRoPE / hybrid).
        if ft.get("bypassed"):
            why = ft.get("bypass_reason", "bypassed")[:30]
            print(fmt_row([r["name"], r["arch"][:9], "", "", "", "bypass",
                           "—", "—", "—", "—", "—", "—", why]))
            continue
        if "tiers" not in ft:
            print(fmt_row([r["name"], r["arch"][:9], "", "", "", ft.get("status", "ERR"),
                           "", "", "", "", "", "", ""]))
            continue
        hotx = _tier(ft, "HOT", "speedup", lambda v: f"{v:.2f}x")
        warmx = _tier(ft, "WARM", "speedup", lambda v: f"{v:.2f}x")
        ssdx = _tier(ft, "SSD", "speedup", lambda v: f"{v:.2f}x")
        ssd_restored = _tier(ft, "SSD", "note", lambda v: "✓" if "restored=True" in v else "✗", "?")
        warmram = (f"{ft['hot_entry_mb']/ft['warm_entry_mb']:.2f}x"
                   if ft.get("warm_entry_mb") else "1.0x")
        loopx = f"{el['speedup']:.2f}x" if "speedup" in el else el.get("status", "?")[:6]
        omx = (f"{om['speedup']:.2f}x" if om.get("status") == "ok" else om.get("status", "?")[:8])
        # VLM: the engine's empirical reuse probe is authoritative (the per-tier
        # text exact-match is noisy for short cross-prompt answers). LLM: per-tier.
        if ft.get("engine") == "vlm" and ft.get("probe_lossless") is not None:
            lossless = bool(ft["probe_lossless"])
        else:
            lossless = all(ft["tiers"][t].get("lossless") for t in ft["tiers"]
                           if t in ("HOT", "WARM", "SSD"))
        print(fmt_row([
            r["name"], r["arch"][:9], ft["prompt_tok"], ft["prefill_tps"], ft["decode_tps"],
            f"{ft['cold_ms']:.0f}", hotx, warmx, f"{ssdx}{ssd_restored}", warmram,
            loopx, omx, "Y" if lossless else "n",
        ]))
    print("=" * 130)
    print("F-* = Yunshu fast path (default). LOOP = Yunshu engine-loop (radix, opt-in, HOT only).")
    print("oMLX = reference/omlx live head-to-head (prefix cache on, block=128). loss = all Yunshu tiers lossless.")
    print("F-SSD trailing ✓/✗ = whether the SSD tier actually restored (vs full-prefill fallback).")
    print("WARMram = HOT-entry / WARM-entry RAM ratio (4-bit saving).")


def main():
    models = ([(m, "?") for m in sys.argv[1:]] if len(sys.argv) > 1 else DEFAULT_MODELS)
    # Per-MODEL thermal tag: the cache benches run sequentially (~15-20 min) so the
    # LAST model runs much hotter than the first. The regression snapshot's single
    # end-of-run gpu_tflops can't reflect that → a model benched hot looks like a
    # regression. Measure the GPU's fp16 ceiling right after EACH model's cache bench
    # and stash it in that model's ft, so the report can thermal-discount per model.
    try:
        import sys as _sys
        _sys.path.insert(0, os.path.join(REPO, "scripts"))
        from perf_history import _gpu_tflops
    except Exception:
        _gpu_tflops = lambda *a, **k: None  # noqa: E731
    rows = []
    for path, arch in models:
        if not os.path.exists(os.path.join(REPO, path)):
            print(f"  SKIP (missing): {path}")
            continue
        print(f"\n>>> {path}")
        _row = bench_model(path, arch)
        _tf = _gpu_tflops(1.5)
        if _tf is not None and isinstance(_row.get("ft"), dict):
            _row["ft"]["_gpu_tflops"] = _tf
            print(f"    [{_row.get('name')}] thermal tag: {_tf} TFLOP/s")
        rows.append(_row)
        print("    raw:", json.dumps(rows[-1])[:200])
    render(rows)
    # also dump full json for the record
    with open("/tmp/bench_all_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nfull json -> /tmp/bench_all_results.json")


if __name__ == "__main__":
    main()
