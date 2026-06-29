"""verify the VLM text 4-tier KV prefix cache.

For mRoPE full-attention VLMs (GLM-OCR, Qwen-VL) reuse is now BYTE-LOSSLESS — the
text path supplies explicit sequential position_ids so a reused prefix resumes at
`matched`. For non-resumable backbones (sliding-window gemma RotatingKVCache,
hybrid Qwen3.5 ArraysCache) `_text_prefix_reuse_safe` returns False and the engine
bypasses (output unchanged vs cache-off). See docs/guides/VLM_TEXT_KV_PREFIX.md.

This script checks BOTH contracts on whatever VLMs are available:
- reuse-capable model: warm long-gen output == no-cache control (byte-lossless),
  full-match AND partial-match, non-streaming AND streaming.
- bypassed model: cache-on output == cache-off (never corrupts).

Usage:
    PYTHONPATH=reference/mlx-vlm:. uv run python scripts/verify_vlm_text_kv_prefix.py \
        /Volumes/P5Plus/models/GLM-OCR-bf16 /Volumes/P5Plus/models/gemma-4-e4b-it-bf16
"""
import asyncio
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath("python"))
from yunshu_engine.types import EngineConfig  # noqa: E402

CTX = "You are a helpful assistant.\n\n" + "".join(
    f"Fact {i}: city-{i:03d} has population {i * 1234 % 99999}.\n" for i in range(40))
FULL = [{"role": "user", "content": CTX + "\n\nWrite a detailed multi-paragraph summary."}]
PROBE = [{"role": "user", "content": CTX + "\n\nList the three largest cities in detail."}]


async def gen(eng, m):
    return (await eng.generate(messages=m, max_tokens=150, temperature=0.0, seed=0))["text"]


async def stream(eng, m):
    t = ""
    async for o in eng.generate_stream(messages=m, max_tokens=150, temperature=0.0, seed=0):
        if getattr(o, "new_text", None):
            t += o.new_text
    return t


async def fresh(model_path, on):
    os.environ["YUNSHU_VLM_KV_PREFIX"] = "1" if on else "0"
    from yunshu_engine.vlm_engine import VLMEngine
    eng = VLMEngine(model_path, EngineConfig())
    if not on:
        eng._text_kv_prefix_cache = None
    await eng.start()
    return eng


async def check(model_path):
    name = model_path.rstrip("/").split("/")[-1]
    # capability probe
    eng = await fresh(model_path, True)
    safe = eng._text_prefix_reuse_safe(eng._model.language_model)
    await eng.stop()

    # controls (cache off)
    eng = await fresh(model_path, False)
    cf, cp = await gen(eng, FULL), await gen(eng, PROBE)
    cfs, cps = await stream(eng, FULL), await stream(eng, PROBE)
    await eng.stop()

    # cache on
    eng = await fresh(model_path, True)
    await gen(eng, FULL); wf = await gen(eng, FULL)          # full match (ns)
    await gen(eng, FULL); wp = await gen(eng, PROBE)         # partial match (ns)
    await stream(eng, FULL); wfs = await stream(eng, FULL)   # full match (stream)
    await stream(eng, FULL); wps = await stream(eng, PROBE)  # partial match (stream) — #1
    hits = eng._text_kv_prefix_cache._total_hits if eng._text_kv_prefix_cache else 0
    await eng.stop()

    if safe:
        ok = (wf == cf) and (wp == cp) and (wfs == cfs) and (wps == cps) and hits > 0
        detail = (f"LOSSLESS reuse: full={wf==cf} partial={wp==cp} "
                  f"stream_full={wfs==cfs} stream_partial={wps==cps} hits={hits}")
    else:
        ok = (wf == cf) and (wfs == cfs)
        detail = f"bypassed (not reuse-safe): cache-on==cache-off={wf==cf and wfs==cfs}"
    print(f"{name}: reuse_safe={safe} => {'OK' if ok else 'FAIL'} | {detail}")
    return ok


DEFAULT_PATHS = [
    "/Volumes/P5Plus/models/GLM-OCR-bf16",          # mRoPE full-attn (lossless reuse)
    "/Volumes/P5Plus/models/gemma-4-e4b-it-bf16",   # sliding-window (bypassed safely)
    "/Volumes/P5Plus/models/Qwen3.5-0.8B-MLX-bf16",  # HYBRID GatedDeltaNet (A2 boundary-snapshot)
]


def main():
    args = sys.argv[1:]
    # Leaf mode: check exactly ONE model in this process (spawned by the parent).
    if args and args[0] == "--_one":
        ok = asyncio.run(check(args[1]))
        sys.exit(0 if ok else 1)

    paths = args or DEFAULT_PATHS
    # MEMORY SAFETY (default): isolate each model in its OWN subprocess so MLX's
    # Metal/wired GPU memory is fully reclaimed by the OS on exit. This script
    # loads each model ~3x (probe + cache-off controls + cache-on); doing three
    # bf16 VLMs (GLM-OCR + gemma-4 + Qwen3.5) in one process accumulated GPU
    # memory → signal 6 (Metal abort) on the 36GB Mac. Same fix as the modality
    # smoke. Set VLM_KV_NO_ISOLATE=1 for the legacy single process.
    if os.environ.get("VLM_KV_NO_ISOLATE") in ("1", "true", "yes") or len(paths) == 1:
        results = [asyncio.run(check(p)) for p in paths]
    else:
        results = []
        for i, p in enumerate(paths):
            rc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--_one", p],
                env={**os.environ},
            ).returncode
            results.append(rc == 0)
            if i < len(paths) - 1:
                time.sleep(20)  # GPU/mem settle between isolated model loads
    print(f"\n{'PASS' if all(results) else 'FAIL'}: VLM text KV prefix cache")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
