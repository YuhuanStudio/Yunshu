"""Realistic-scenario cache benchmark — how the KV cache actually performs in
real usage patterns (not synthetic prefix sweeps):

  1. MULTI-TURN CHAT: an 8-turn conversation. Each turn's prompt = the whole prior
     transcript + the new user message, so the shared prefix GROWS every turn.
     Measures per-turn cache-hit-rate (cached/prompt), TTFT, and TTFT vs no-cache.
  2. RAG / shared-doc: one long document (system) reused across N distinct user
     questions — the classic prompt-caching win. Measures hit-rate + TTFT speedup.

Yunshu production fast path, in-process. Engine-agnostic (LLM/VLM).

Run: PYTHONPATH=. YUNSHU_BENCH_MODEL=./models/Qwen2.5-3B-Instruct-bf16 \
       uv run python scripts/bench_realistic_cache.py
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


def _is_vlm(path):
    try:
        mt = json.loads(open(os.path.join(path, "config.json")).read()).get("model_type")
        return mt and importlib.util.find_spec(f"mlx_lm.models.{mt}") is None
    except Exception:
        return False


_IS_VLM = _is_vlm(MODEL)


class _Out:
    __slots__ = ("text", "prompt_tokens", "completion_tokens", "cached_tokens", "ttft_ms")

    def __init__(self, raw, dt):
        g = raw.get if isinstance(raw, dict) else (lambda k, d=0: getattr(raw, k, d))
        self.text = g("text", "") or ""
        self.prompt_tokens = g("prompt_tokens", 0)
        self.completion_tokens = g("completion_tokens", 0)
        self.cached_tokens = g("cached_tokens", 0)
        self.ttft_ms = dt * 1000


async def _chat(engine, messages, max_tokens):
    t0 = time.perf_counter()
    if _IS_VLM:
        raw = await engine.generate(messages=messages, max_tokens=max_tokens, temperature=0.0, enable_thinking=False)
    else:
        raw = await engine.chat(messages=messages, max_tokens=max_tokens, temperature=0.0, enable_thinking=False)
    return _Out(raw, time.perf_counter() - t0)


def _pc(engine):
    return getattr(engine, "_kv_prefix_cache", None) or getattr(engine, "_text_kv_prefix_cache", None)


_DOC = ("Knowledge base. " + "".join(
    f"Entry {i}: product P{i:03d} has SKU {i*7%1000}, price ${i*3%500}, stock {i*11%200} units, "
    f"category C{i%12}, rating {i%5+1} stars. " for i in range(120)))

_QUESTIONS = [
    "What is the price of product P042?", "Which category is P017 in?",
    "How many units of P099 are in stock?", "What is the SKU of P003?",
    "List products in category C5.", "What is the rating of P077?",
    "Compare P010 and P020 by price.", "Summarize the catalog in one sentence.",
]


async def main():
    name = os.path.basename(MODEL.rstrip("/"))
    print(f"\n╔══ REALISTIC CACHE BENCH: {name} ══")
    if _IS_VLM:
        from yunshu_engine.types import EngineConfig
        from yunshu_engine.vlm_engine import VLMEngine
        engine = VLMEngine(MODEL, EngineConfig())
    else:
        from yunshu_engine.batched_engine import BatchedEngine
        engine = BatchedEngine(model_name=MODEL)
    await engine.start()
    pc = _pc(engine)
    await _chat(engine, [{"role": "user", "content": "hi"}], 4)
    result = {"model": name}

    # ── 1. MULTI-TURN CHAT (growing shared prefix) ──
    print("╠══ 1. MULTI-TURN CHAT — prefix grows each turn ══")
    print(f"║  {'turn':>4} {'prompt_tok':>10} {'cached':>7} {'hit%':>6} {'TTFT_ms':>9}")
    if pc is not None:
        pc.clear()
    convo = [{"role": "system", "content": "You are a concise assistant."}]
    user_turns = ["Tell me about photosynthesis.", "What pigment is involved?",
                  "And the wavelength it absorbs?", "How does that relate to leaf color?",
                  "What about C4 plants?", "Give an example C4 crop.",
                  "What is its water-use efficiency?", "Summarize this whole chat."]
    mt_rows = []
    for ti, u in enumerate(user_turns):
        convo.append({"role": "user", "content": u})
        o = await _chat(engine, convo, 48)
        convo.append({"role": "assistant", "content": o.text})
        hit = o.cached_tokens / o.prompt_tokens if o.prompt_tokens else 0
        mt_rows.append({"turn": ti + 1, "prompt_tok": o.prompt_tokens, "cached": o.cached_tokens,
                        "hit_pct": round(hit * 100, 1), "ttft_ms": round(o.ttft_ms, 1)})
        print(f"║  {ti+1:>4} {o.prompt_tokens:>10} {o.cached_tokens:>7} {hit*100:>5.1f}% {o.ttft_ms:>9.1f}")
    result["multi_turn"] = mt_rows
    agg_hit = sum(r["cached"] for r in mt_rows[1:]) / max(1, sum(r["prompt_tok"] for r in mt_rows[1:]))
    print(f"║  aggregate hit-rate (turns 2+): {agg_hit*100:.1f}%")

    # ── 2. RAG / shared-doc (long doc reused across questions) ──
    print("╠══ 2. RAG — one long doc, N distinct questions ══")
    print(f"║  {'q#':>3} {'prompt_tok':>10} {'cached':>7} {'hit%':>6} {'TTFT_ms':>9} {'vs_cold':>8}")
    if pc is not None:
        pc.clear()
    # cold reference: first question, no cache
    rag_rows = []
    cold_ttft = None
    for qi, q in enumerate(_QUESTIONS):
        msgs = [{"role": "system", "content": _DOC}, {"role": "user", "content": q}]
        o = await _chat(engine, msgs, 32)
        if cold_ttft is None:
            cold_ttft = o.ttft_ms  # first = cold (doc not yet cached)
        hit = o.cached_tokens / o.prompt_tokens if o.prompt_tokens else 0
        speedup = cold_ttft / o.ttft_ms if o.ttft_ms else 0
        rag_rows.append({"q": qi + 1, "prompt_tok": o.prompt_tokens, "cached": o.cached_tokens,
                         "hit_pct": round(hit * 100, 1), "ttft_ms": round(o.ttft_ms, 1),
                         "speedup_vs_q1": round(speedup, 2)})
        print(f"║  {qi+1:>3} {o.prompt_tokens:>10} {o.cached_tokens:>7} {hit*100:>5.1f}% {o.ttft_ms:>9.1f} {speedup:>7.2f}x")
    result["rag"] = rag_rows
    warm_hit = sum(r["cached"] for r in rag_rows[1:]) / max(1, sum(r["prompt_tok"] for r in rag_rows[1:]))
    print(f"║  aggregate hit-rate (q2+): {warm_hit*100:.1f}%")

    print(f"╚{'═'*54}")
    print("@@REALISTIC@@ " + json.dumps(result))
    try:
        await engine.stop()
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
