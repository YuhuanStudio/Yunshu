#!/usr/bin/env python3
"""Yunshu coverage matrix — the honest map of what is actually verified.

This is the ANTITHESIS of docs/archive/VALIDATION_REPORT.md (a 577-wave narrative
of uncertain reliability). It is pure metadata (no model loading — runs in <1s,
crashes nothing) that maps every technique × model × interface to its EVIDENCE:

  ✅ GATED   — a scripts/regression.py section verifies it on EVERY run.
  🟢 VERIFIED— proven by a real-model run (cited), but NOT yet in the regression
               harness → a gap to close (add a gate so it can't regress).
  🟡 OPT-IN  — works, but off by default, with the stated reason.
  ❌ GAP     — the capability exists in code but is NOT systematically tested.
  🗑️ REMOVED — deleted with evidence (e.g. benchmarked slower).
  🚫 BLOCKED — needs an artifact/hardware we don't have.
  ⛔ N/A     — does not apply to this cell (with reason).

The `gate` field of each technique is checked against the LIVE section list in
regression.py, so this map cannot silently drift from the harness. Run:

  PYTHONPATH=. uv run python scripts/coverage_matrix.py            # full matrix
  PYTHONPATH=. uv run python scripts/coverage_matrix.py --gaps     # only the gaps
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_REGRESSION = _ROOT / "scripts" / "regression.py"


def _live_gate_sections() -> list[str]:
    """Parse the section NAMES regression.py actually registers (so we can't drift)."""
    txt = _REGRESSION.read_text(encoding="utf-8")
    # sections are tuples: ("name", "tier", gate_bool, [cmd], {env}, parser)
    return re.findall(r'\(\s*"([^"]+)",\s*"(?:smoke|standard|full)",\s*(?:True|False)', txt)


# ── Models physically present, grouped by modality (≈25 on /Volumes/P5Plus) ──
MODELS = {
    "LLM (text)": [
        "Qwen2.5-3B-Instruct-4bit", "Qwen2.5-3B-Instruct-bf16",
        "Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B-4bit", "Qwen3.5-9B-bf16",
        "Qwen3.6-27B-MTP-4bit", "gemma-4-e4b-it", "gemma-4-E4B-assistant",
    ],
    "VLM (vision)": ["GLM-OCR", "gemma-4-e4b (vision)", "Qwen3-Omni-30B-A3B"],
    "Image": ["Z-Image-Turbo-MLX-4bit", "Z-Image-ControlNet"],
    "Audio": ["Qwen3-TTS-1.7B-VoiceDesign", "Qwen3-ASR-1.7B"],
    "Video": ["Wan2.2-TI2V-5B", "Lance-3B-Video"],
    "Embeddings": ["(BGE-class embedding model)"],
}

# ── Techniques: status, the regression gate (must match a LIVE section), the
#    models it's actually been verified on, and an honest note. ──
# status ∈ {GATED, VERIFIED, OPT-IN, GAP, REMOVED, BLOCKED}
TECHNIQUES = [
    ("fast-path single-request decode", "GATED", "unit-tests",
     ["all LLM"], "Default serving (_generate_fast). Exercised by ~40 verify_* sections."),
    ("engine-loop continuous batching", "GATED", "cache-lossless: engine-loop",
     ["Qwen2.5-3B"], "Opt-in YUNSHU_ENGINE_LOOP=1. Gated lossless; only 1 model."),
    ("KV-prefix cache HOT", "GATED", "cache-tier matrix (all models)",
     ["Qwen2.5-3B", "Qwen3.5", "gemma-4", "GLM-OCR"], "bench_all sweeps the tier×model matrix."),
    ("KV-prefix cache WARM (4-bit)", "GATED", "cache-tier matrix (all models)",
     ["Qwen2.5-3B"], "3.56x less RAM lossless; gemma-4 WARM saves no RAM (sliding window)."),
    ("KV-prefix cache SSD tier", "GATED", "cache-tier matrix (all models)",
     ["Qwen2.5-3B"], "Net-negative for fast-prefill models (documented)."),
    ("KV-prefix hybrid (Qwen3.5)", "GATED", "cache-lossless: VLM text (GLM-OCR + gemma)",
     ["Qwen3.5"], "Boundary-snapshot reuse; opt-in YUNSHU_HYBRID_PREFIX=1."),
    ("gemma-4 assistant-drafter spec decode", "GATED", "spec-decode lossless (gemma-4)",
     ["gemma-4-assistant"], "THE production spec win — an EAGLE-style ASSISTANT DRAFTER "
     "(NOT MTP), 2.13x lossless greedy measured live (W969). Note: 'MTP' proper (home-grown "
     "+ mlx-vlm) is measured SLOWER / proof-script-only — do not conflate the two."),
    ("n-gram spec decode", "GATED", "n-gram spec (W686: hybrid guard==greedy + dense non-degenerate)",
     ["Qwen2.5-3B (W686)", "Qwen3.5 (W686 guarded→fast)"],
     "W686 fix gated: Qwen3.5 non-trimmable guard → bit-identical to greedy; Qwen2.5 "
     "spec non-degenerate. SLOW on Apple Silicon (0.61-0.83x) so stays opt-in."),
    ("cross-model spec decode", "GATED", "unit-tests",
     ["model-free"], "Logic gated via test_speculative_decoder (unit suite). Gated to "
     "greedy (W678) + slower on Apple Silicon, so opt-in; no live section needed."),
    ("MTP (mlx-vlm / Qwen3.6-27B)", "GATED", "MTP spec decode (Qwen3.6-27B production path, coherent)",
     ["Qwen3.6-27B-MTP-4bit"], "Gated (full tier): the 27B-4bit MTP backend loads in 36GB "
     "(single ~14GB load) + produces coherent output. HONESTY (W969): the '1.82x' figure is "
     "from a standalone PROOF script self-marked 'INTEGRATION TODO' — NOT a served, "
     "regression-gated number; the wired mlx-vlm MTP path is single-backend + drops "
     "top_p/json_schema/penalties + non-streaming. NOT a shipped prod win. The real spec win "
     "is the gemma-4 assistant drafter above. Small Qwen3.5 keep MTP in a separate "
     "mtp-weights.safetensors that is_mtp_capable's index-check misses (known limitation)."),
    ("LoRA (diffusion / image) — THE product LoRA", "GATED", "image diffusion-LoRA (load/effect/restore)",
     ["Z-Image"], "Gated: ComfyUI lora_down/lora_up/alpha loader, load/effect/restore. "
     "This is Yunshu's actual LoRA use case."),
    ("LoRA serving (text adapters) — secondary", "OPT-IN", None,
     [], "lora_manager.py is a secondary text-LLM path (NOT the product — image LoRA is). "
     "W687 fixed a real crash there: mx.array(x, stream=mx.cpu) is invalid API → broke "
     "base-model snapshot on every real model (unit tests use mocks). Left ungated by intent."),
    ("KV-quant (KIVI 2-bit/4-bit)", "GATED", "KV-cache quant round-trip (4/8-bit bounded + compression)",
     ["model-free"], "mx-based KVQuantizer round-trip gated: 4-bit 3.6% err, 8-bit 0.23%, "
     "compression 32/bits. (The Metal kivi kernel was deleted W685.)"),
    ("priority / FAIR scheduling", "GATED", "priority/FAIR scheduling (W683: policy plumbed + ordering)",
     ["model-free"], "W683 gated: config→policy plumbing (priority/fair/bogus→fcfs) + "
     "PRIORITY queue pops high-first while FCFS keeps arrival order. Deterministic."),
    ("VLM vision-feature cache", "GATED", "VLM vision/image-reuse cache (W682: wrapper + ~6x repeat-image)",
     ["gemma-4-e4b (W682)"], "W682 gated: vision-tower wrapper installs + the existing "
     "image-reuse is ~6x on a repeat image (measured 2.74s→0.45s). Output stays correct."),
    ("grammar / constrained decode", "GATED", "grammar constraints (choice + regex)",
     ["Qwen2.5-3B"], "choice + regex gated."),
    ("structured output (json_schema)", "GATED", "structured output (json_schema enforced)",
     ["Qwen2.5-3B"], "Enforced-schema gated."),
    ("Anthropic prompt caching", "GATED", "unit-tests",
     ["model-free"], "cache_control breakpoints + cached_tokens covered by unit tests "
     "(6+; many W666 fixes). Logic-level; no real model needed."),
    ("inflight prefix sharing", "GATED", "unit-tests",
     ["model-free"], "InflightPrefixTracker covered by 22 unit tests. A LIVE trigger is "
     "timing-dependent on the serialized (max_workers=1) fast path, so logic-gated only."),
    ("chunked prefill", "REMOVED", None,
     [], "Manual version corrupted KV; mlx-lm native prefill_step_size supersedes it."),
    ("Metal kernels", "REMOVED", None,
     [], "W685: all 5 benchmarked slower than mx.fast or broken. Deleted."),
    ("Medusa / spec factory", "BLOCKED", None,
     [], "Needs trained Medusa heads Yunshu doesn't ship. Kept, bannered NOT-WIRED."),
    ("batch-path spec (engine-loop)", "BLOCKED", None,
     [], "Statistics-only; mlx-lm BatchGenerator exposes no multi-token accept."),
]

# ── Interfaces (protocol surface) ──
INTERFACES = [
    ("OpenAI /v1/chat/completions", "GATED", "core endpoints (/v1/models + /v1/completions)"),
    ("OpenAI /v1/completions (legacy)", "GATED", "core endpoints (/v1/models + /v1/completions)"),
    ("OpenAI /v1/embeddings", "GATED", "scoring endpoints HTTP (embeddings/score/rerank/classify)"),
    ("OpenAI /v1/images/generations", "GATED", "image-gen HTTP route (/v1/images/generations b64_json)"),
    ("OpenAI streaming SSE", "GATED", "streaming SSE (chunk contract + assembled==non-stream)"),
    ("Anthropic /v1/messages", "GATED", "Anthropic /v1/messages (envelope + streaming events)"),
    ("Anthropic count_tokens", "GATED", "Anthropic count_tokens (matches usage + monotonic)"),
    ("Responses /v1/responses", "GATED", "Responses API /v1/responses (envelope + streaming lifecycle)"),
    ("Batch /v1/batch", "GATED", "batch inference API (/v1/batch custom_id mapping)"),
    ("Realtime WebSocket", "GATED", "unit-tests"),
    ("MCP (model-context-protocol)", "GATED", "unit-tests"),
]

# ── Parameter coverage (generation knobs) ──
PARAMS = [
    ("temperature / determinism", "GATED", "sampling (temp0 determinism + seed repro/vary)"),
    ("top_p / top_k / min_p", "GATED", "sampling constraints (top_k/top_p/min_p collapse)"),
    ("freq / presence penalties", "GATED", "freq/presence penalties (reduce repetition)"),
    ("logit_bias (+ OOV guard)", "GATED", "logit_bias effect + out-of-vocab guard"),
    ("stop sequences", "GATED", "logprobs + stop sequences (generation contract)"),
    ("seed reproducibility", "GATED", "sampling (temp0 determinism + seed repro/vary)"),
    ("n>1 parallel sampling", "GATED", "n>1 parallel sampling (gateway choice/usage contract)"),
    ("logprobs / top_logprobs", "GATED", "streaming logprobs (per-chunk entries)"),
    ("json_schema / response_format", "GATED", "JSON mode (response_format json_object)"),
    ("tools / tool_choice", "GATED", "tool-calling E2E (assembly + selection + envelope)"),
    ("usage accounting", "GATED", "usage accounting (exact prompt_tokens + stream==non-stream)"),
    ("context-window overflow", "GATED", "context-window overflow (clean 400, not crash)"),
    ("xtc_probability / xtc_threshold", "GATED", "sampling extras (XTC removes tokens + thinking_budget caps)"),
    ("enable_thinking / thinking_budget", "GATED", "sampling extras (XTC removes tokens + thinking_budget caps)"),
]

_BADGE = {
    "GATED": "✅ GATED", "VERIFIED": "🟢 VERIFIED", "OPT-IN": "🟡 OPT-IN",
    "GAP": "❌ GAP", "REMOVED": "🗑️ REMOVED", "BLOCKED": "🚫 BLOCKED",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaps", action="store_true", help="show only gaps/blocked")
    args = ap.parse_args()

    live = set(_live_gate_sections())
    out: list[str] = []
    out.append("# Yunshu Coverage Matrix — the honest map\n")
    out.append(f"_Derived live from regression.py ({len(live)} sections). "
               "Each GATED cell cites the section that verifies it every run._\n")

    # Drift check: any technique/interface/param claiming a gate that no longer exists?
    drift = []
    for name, status, gate, *_ in TECHNIQUES:
        if status == "GATED" and gate and gate not in live:
            drift.append(f"technique '{name}' → missing section '{gate}'")
    for name, status, gate in INTERFACES + PARAMS:
        if status == "GATED" and gate and gate not in live:
            drift.append(f"'{name}' → missing section '{gate}'")
    if drift:
        out.append("> ⚠️ **DRIFT**: gates referenced below no longer exist in regression.py:")
        out += [f">   - {d}" for d in drift]
        out.append("")

    # Techniques
    out.append("## Techniques\n")
    out.append("| Technique | Status | Gate (regression section) | Verified on |")
    out.append("|---|---|---|---|")
    for name, status, gate, models, note in TECHNIQUES:
        if args.gaps and status not in ("GAP", "BLOCKED"):
            continue
        g = gate if gate else ("—" if status in ("REMOVED", "BLOCKED") else "**none → add one**")
        m = ", ".join(models) if models else "—"
        out.append(f"| {name} | {_BADGE[status]} | {g} | {m} |")
        if note:
            out.append(f"| ↳ _{note}_ | | | |")

    # Interfaces
    out.append("\n## Interfaces\n")
    out.append("| Interface | Status | Gate |")
    out.append("|---|---|---|")
    for name, status, gate in INTERFACES:
        if args.gaps and status not in ("GAP", "BLOCKED"):
            continue
        out.append(f"| {name} | {_BADGE[status]} | {gate or '**none → add one**'} |")

    # Params
    out.append("\n## Parameters\n")
    out.append("| Parameter | Status | Gate |")
    out.append("|---|---|---|")
    for name, status, gate in PARAMS:
        if args.gaps and status not in ("GAP", "BLOCKED"):
            continue
        out.append(f"| {name} | {_BADGE[status]} | {gate or '**none → add one**'} |")

    # Models × modality (breadth)
    out.append("\n## Models present (breadth)\n")
    for modality, ms in MODELS.items():
        out.append(f"- **{modality}**: {', '.join(ms)}")
    out.append("\n> ✅ Model breadth NOW PROVEN: `sweep_models.py` (full tier) runs core "
               "correctness (greedy-determinism, non-degenerate, multi-turn template) across "
               "all 6 LLMs (Qwen2.5-3B ×2, Qwen3.5-0.8/2/9B, gemma-4), subprocess-isolated "
               "(36GB-safe). 6/6 pass. `sweep_perf.py` records per-model TTFT/decode/RSS.\n")

    # Summary
    def _tally(rows, idx):
        from collections import Counter
        return Counter(r[idx] for r in rows)
    tt = _tally(TECHNIQUES, 1)
    it = _tally(INTERFACES, 1)
    pt = _tally(PARAMS, 1)
    out.append("## Summary\n")
    for label, t in (("Techniques", tt), ("Interfaces", it), ("Parameters", pt)):
        parts = ", ".join(f"{_BADGE[k]}×{v}" for k, v in sorted(t.items()))
        out.append(f"- **{label}**: {parts}")
    gaps = [t[0] for t in TECHNIQUES if t[1] == "GAP"] + \
           [i[0] for i in INTERFACES if i[1] == "GAP"] + \
           [p[0] for p in PARAMS if p[1] == "GAP"]
    out.append(f"\n### Gaps to close ({len(gaps)}), priority order:")
    for i, g in enumerate(gaps, 1):
        out.append(f"{i}. {g}")

    text = "\n".join(out)
    print(text)
    rep = _ROOT / "docs" / "reports" / "COVERAGE_MATRIX.md"
    rep.write_text(text + "\n", encoding="utf-8")
    print(f"\n[written] {rep.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
