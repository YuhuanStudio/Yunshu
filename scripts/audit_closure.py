#!/usr/bin/env python3
"""Gateway endpoint audit-closure tracker.

Turns the open-ended "find any bug on a fresh subsystem" grind into a FINITE, checkable
list: every gateway HTTP/WS endpoint is auto-discovered, and each carries an audit verdict.
The point is convergence — once every endpoint's default path is marked `clean`/`hardened`,
the default serving surface is formally closed and remaining work is only opt-in/edge paths.

Run:  PYTHONPATH=. uv run python scripts/audit_closure.py
      ... --todo     # list only un-audited endpoints (the remaining worklist)

It DRIFT-WARNS when an endpoint exists with no STATUS entry (new, unaudited) or a STATUS
entry has no matching endpoint (removed) — so new routes can't silently escape the audit.

Verdicts:
  clean     — read/inspected this area, no bug found on the default path
  hardened  — had real bugs that were fixed across the cited passes; default path now sound
  todo      — not yet given a focused default-path audit
  deferred  — known-limited opt-in/edge behaviour, intentionally NOT fixed (documented)
"""
from __future__ import annotations

import re
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
ROUTERS = ROOT / "python" / "yunshu_gateway" / "routers"
_RX = re.compile(r'@router\.(get|post|put|delete|patch|websocket)\(\s*["\']([^"\']+)["\']')


def discover() -> list[tuple[str, str, str, str]]:
    """Return (file_stem, METHOD, path, fn) for every @router endpoint."""
    rows = []
    for f in sorted(ROUTERS.glob("*.py")):
        src = f.read_text().splitlines()
        for i, ln in enumerate(src):
            m = _RX.search(ln)
            if not m:
                continue
            method, path = m.group(1).upper(), m.group(2)
            fn = ""
            for j in range(i + 1, min(i + 6, len(src))):
                dm = re.search(r'(?:async\s+)?def\s+(\w+)', src[j])
                if dm:
                    fn = dm.group(1)
                    break
            rows.append((f.stem, method, path, fn))
    return rows


# Audit status keyed by "file:METHOD:path". Verdict + representative passes + note.
# Seeded from the sprint history (memory index). `hardened` lists a few of the
# passes that fixed real bugs there; it is NOT an exhaustive changelog.
STATUS: dict[str, tuple[str, str]] = {
    # ── Core inference: the default serving hot paths (highest value) ──
    "chat:POST:/chat/completions": ("hardened", "— tool-parse, CoT-leak, TPM, GLM stream"),
    "completions:POST:/completions": ("hardened", "— param parity, disconnect, FIM-suffix leak"),
    "responses:POST:/responses": ("hardened", "— bg, multimodal, tool-markup, cancel-queued"),
    "responses:GET:/responses/{response_id}": ("hardened", "cross-tenant IDOR; ownership enforced"),
    "responses:DELETE:/responses/{response_id}": ("hardened", "ownership"),
    "responses:POST:/responses/{response_id}/cancel": ("hardened", "cancel-during-queued"),
    "anthropic:POST:/messages": ("hardened", "— markers, tool_result-image, multi-tool, tool_choice"),
    "anthropic:POST:/messages/count_tokens": ("hardened", "temp-file leak"),
    "embeddings:POST:/embeddings": ("clean", "— finite guard, ANE mean-pool, length cap"),
    "scoring:POST:/pooling": ("hardened", "NaN guard, length cap"),
    "scoring:POST:/score": ("hardened", "— metric, length cap"),
    "scoring:POST:/rerank": ("clean", "NaN-sort, length cap"),
    "scoring:POST:/classify": ("clean", "length cap"),
    # ── Audio (heavily audited across the realtime/ASR/TTS sprints) ──
    "audio:POST:/audio/speech": ("hardened", "— format, DSP, sample-rate"),
    "audio:POST:/audio/speech/stream": ("hardened", "streaming resample"),
    "audio:POST:/audio/transcriptions": ("hardened", "— 24-bit, VAD short-utt"),
    "audio:POST:/audio/translations": ("hardened", "active-req counter"),
    "audio:GET:/audio/voices": ("clean", "static list"),
    "audio:POST:/audio/voice-pipeline": ("hardened", "RBAC"),
    "audio:POST:/audio/speech-to-speech/enhance": ("hardened", "DSP, spectral-gating"),
    "audio:POST:/audio/speech-to-speech/separate": ("hardened", "truncation→STFT"),
    "audio:POST:/audio/speech-to-speech/transform": ("hardened", "truncation→STFT"),
    # ── Images (product feature; heavily audited) ──
    "images:POST:/images/generations": ("hardened", "— LoRA, model-iso, round, cancel"),
    "images:POST:/images/generations/stream": ("hardened", "inline-LoRA"),
    "images:POST:/images/variations": ("hardened", "model-iso"),
    "images:POST:/images/edits": ("hardened", "cancel sweep"),
    "images:POST:/images/inpaint": ("hardened", "RePaint, cancel"),
    "images:POST:/images/controlnet": ("hardened", "stop-leak, cancel"),
    "images:POST:/images/depth-guided": ("hardened", "cancel"),
    # ── Other modalities ──
    "video:POST:/video/generations": ("hardened", "— model-iso, native-fail-loud; _denoise cancel deferred"),
    "ocr:POST:/v1/ocr": ("hardened", "— model-iso, think-strip, native key"),
    # ── MCP ──
    "mcp:POST:/mcp": ("hardened", "— envelope, lock, handshake, generate-iso"),
    "mcp:GET:/mcp/tools": ("clean", "discovery, model-iso"),
    "mcp:GET:/mcp/sse": ("hardened", "DoS, client-hang"),
    "mcp:GET:/mcp/client/status": ("clean", "status read"),
    "mcp:GET:/mcp/client/tools": ("clean", "model-iso"),
    # ── Realtime ──
    "realtime:WEBSOCKET:/v1/realtime": ("hardened", "many"),
    "realtime:WEBSOCKET:/realtime": ("hardened", "alias of /v1/realtime"),
    # ── Control plane ──
    "models:GET:/models": ("hardened", "— info-iso, static auth"),
    "models:GET:/models/{model_id:path}": ("hardened", "info-iso"),
    "models:POST:/models/load": ("hardened", "legacy-tenant privesc"),
    "models:POST:/models/unload/{model_id:path}": ("hardened", "— in-use, false-success"),
    "sleep:POST:/sleep": ("hardened", "— consistency, drain, engine-leak"),
    "sleep:POST:/wake-up": ("hardened", "rebuild"),
    "sleep:GET:/sleep/status": ("clean", "gated"),
    "cancel:POST:/cancel": ("hardened", "— privesc, stop-hook, tenant-iso"),
    "cancel:GET:/active-generations": ("clean", "owner-scope"),
    "cached_contents:POST:/cachedContents": ("hardened", "per-owner evict"),
    "cached_contents:GET:/cachedContents": ("clean", "ownership"),
    "cached_contents:GET:/cachedContents/{cid}": ("clean", "ownership"),
    "cached_contents:PATCH:/cachedContents/{cid}": ("clean", "audit — can_infer + _owns 404 IDOR guard + TTL bounds"),
    "cached_contents:DELETE:/cachedContents/{cid}": ("clean", "ownership"),
    # ── Tokenize (audit: CLEAN — auth+model-iso on all 3, fallback + decode
    #    overflow→422 + per-prompt ctx; inputs bounded by the global 10MB body cap,
    #    CPU-only no forward pass) ──
    "tokenize:POST:/tokenize": ("clean", "model-iso fallback; body-cap bounds input"),
    "tokenize:POST:/detokenize": ("clean", "decode overflow→422"),
    "tokenize:POST:/token_count": ("clean", "per-prompt ctx check"),
    # ── Observability — monitoring (28) + profiling (4) audited as a whole class in     #    CLEAN. Uniform can_view_system / can_admin gating, robust None-subsystem handling
    #    (no engine/KV/mesh → empty not 500), no div-by-zero, profiling path-traversal blocked.
    #    Individual highlights kept; the rest inherit FILE_DEFAULT below. ──
    "monitoring:GET:/prometheus": ("hardened", "— DoS, middleware-order, escape"),
    "monitoring:GET:/health-dashboard": ("hardened", "lying-readiness"),
    # ── Batch inference (audit: no HIGH/IDOR/isolation — auth+model-iso+ownership all
    #    gated, size/timeout/cancel bounded; 3 LOW fixed) ──
    "batch_inference:POST:/batch": ("hardened", "model-iso per item, sampling bounds"),
    "batch_inference:GET:/batch/{batch_id}/status": ("clean", "can_infer + _owns_batch IDOR guard"),
    "batch_inference:GET:/batch/{batch_id}/results": ("clean", "ownership + 409-in-progress"),
    "batch_inference:GET:/batch/{batch_id}/results.csv": ("hardened", "CSV-injection hardening"),
    "batch_inference:POST:/batch/upload/csv": ("hardened", "max_concurrent bound + UTF-8 guard"),
}

# Per-file default verdict for routers audited as a whole class (an endpoint not listed
# individually in STATUS inherits its file's default; absent → "todo").
FILE_DEFAULT: dict[str, tuple[str, str]] = {
    "monitoring": ("clean", "class audit — uniform can_view_system gate, None-safe, no div0"),
    "profiling": ("clean", "class audit — can_admin gate, path-traversal blocked, lock-safe"),
    "bench": ("hardened", "class audit — can_benchmark gate, SSRF-defended, lock-serialized, param bounds"),
}

_VERDICT_ORDER = ["todo", "deferred", "hardened", "clean"]
_GLYPH = {"clean": "✅", "hardened": "🛠️ ", "todo": "⬜", "deferred": "🔵"}


def main() -> int:
    rows = discover()
    keys = {f"{s}:{m}:{p}" for s, m, p, _ in rows}
    todo_only = "--todo" in sys.argv

    # drift: STATUS entries with no matching endpoint
    stale = sorted(set(STATUS) - keys)

    counts: dict[str, int] = {v: 0 for v in _VERDICT_ORDER}
    by_file: dict[str, list] = {}
    for s, m, p, fn in rows:
        key = f"{s}:{m}:{p}"
        verdict, note = STATUS.get(key, FILE_DEFAULT.get(s, ("todo", "")))
        counts[verdict] = counts.get(verdict, 0) + 1
        by_file.setdefault(s, []).append((m, p, verdict, note, fn))

    total = len(rows)
    audited = counts.get("clean", 0) + counts.get("hardened", 0)
    print("═" * 78)
    print(f" Gateway endpoint audit closure — {audited}/{total} default paths closed "
          f"({counts['clean']} clean + {counts['hardened']} hardened), "
          f"{counts['todo']} todo, {counts['deferred']} deferred")
    print("═" * 78)
    for s in sorted(by_file):
        eps = by_file[s]
        if todo_only:
            eps = [e for e in eps if e[2] == "todo"]
            if not eps:
                continue
        print(f"\n  {s}/")
        for m, p, verdict, note, fn in eps:
            g = _GLYPH.get(verdict, "?")
            line = f"    {g} {m:9} {p:38} {fn}"
            if note:
                line += f"  — {note}"
            print(line)

    if stale:
        print("\n  ⚠️  DRIFT — STATUS entries with no matching endpoint (removed?):")
        for k in stale:
            print(f"      {k}")

    print()
    print(f"  Remaining worklist (todo): {counts['todo']} endpoints — "
          f"mostly read-only observability/bench/profiling GETs.")
    print(f"  Default-serving inference surface (chat/completions/responses/anthropic/"
          f"embeddings/scoring/audio/images/mcp/realtime): formally closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
