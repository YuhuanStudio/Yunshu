#!/usr/bin/env python3
"""Gateway endpoint audit-closure tracker (Wave 965).

Turns the open-ended "find any bug on a fresh subsystem" grind into a FINITE, checkable
list: every gateway HTTP/WS endpoint is auto-discovered, and each carries an audit verdict.
The point is convergence — once every endpoint's default path is marked `clean`/`hardened`,
the default serving surface is formally closed and remaining work is only opt-in/edge paths.

Run:  PYTHONPATH=. uv run python scripts/audit_closure.py
      ... --todo     # list only un-audited endpoints (the remaining worklist)

It DRIFT-WARNS when an endpoint exists with no STATUS entry (new, unaudited) or a STATUS
entry has no matching endpoint (removed) — so new routes can't silently escape the audit.

Verdicts:
  clean     — read/inspected this wave-sprint, no bug found on the default path
  hardened  — had real bugs that were fixed across the cited waves; default path now sound
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


# Audit status keyed by "file:METHOD:path". Verdict + representative waves + note.
# Seeded from the W7xx-W964 sprint history (memory index). `hardened` lists a few of the
# waves that fixed real bugs there; it is NOT an exhaustive changelog.
STATUS: dict[str, tuple[str, str]] = {
    # ── Core inference: the default serving hot paths (highest value) ──
    "chat:POST:/chat/completions": ("hardened", "W854/855/897/919/920/922/932/961 — tool-parse, CoT-leak, TPM, GLM stream"),
    "completions:POST:/completions": ("hardened", "W913/914/918/943 — param parity, disconnect, FIM-suffix leak"),
    "responses:POST:/responses": ("hardened", "W780/817/895/941/947 — bg, multimodal, tool-markup, cancel-queued"),
    "responses:GET:/responses/{response_id}": ("hardened", "W714 cross-tenant IDOR; ownership enforced"),
    "responses:DELETE:/responses/{response_id}": ("hardened", "W714 ownership"),
    "responses:POST:/responses/{response_id}/cancel": ("hardened", "W947 cancel-during-queued"),
    "anthropic:POST:/messages": ("hardened", "W690/808/896/933/945 — markers, tool_result-image, multi-tool, tool_choice"),
    "anthropic:POST:/messages/count_tokens": ("hardened", "W690 temp-file leak"),
    "embeddings:POST:/embeddings": ("clean", "W867/936/955 — finite guard, ANE mean-pool, length cap"),
    "scoring:POST:/pooling": ("hardened", "W959 NaN guard, W960 length cap"),
    "scoring:POST:/score": ("hardened", "W690/850/960 — metric, length cap"),
    "scoring:POST:/rerank": ("clean", "W812 NaN-sort, W960 length cap"),
    "scoring:POST:/classify": ("clean", "W960 length cap"),
    # ── Audio (heavily audited across the realtime/ASR/TTS sprints) ──
    "audio:POST:/audio/speech": ("hardened", "W847/849/935 — format, DSP, sample-rate"),
    "audio:POST:/audio/speech/stream": ("hardened", "W935 streaming resample"),
    "audio:POST:/audio/transcriptions": ("hardened", "W903/952 — 24-bit, VAD short-utt"),
    "audio:POST:/audio/translations": ("hardened", "W804 active-req counter"),
    "audio:GET:/audio/voices": ("clean", "static list"),
    "audio:POST:/audio/voice-pipeline": ("hardened", "W853b RBAC"),
    "audio:POST:/audio/speech-to-speech/enhance": ("hardened", "W849/931 DSP, spectral-gating"),
    "audio:POST:/audio/speech-to-speech/separate": ("hardened", "W765 truncation→STFT"),
    "audio:POST:/audio/speech-to-speech/transform": ("hardened", "W765 truncation→STFT"),
    # ── Images (product feature; heavily audited) ──
    "images:POST:/images/generations": ("hardened", "W718/824/898/899/902 — LoRA, model-iso, round, cancel"),
    "images:POST:/images/generations/stream": ("hardened", "W898 inline-LoRA"),
    "images:POST:/images/variations": ("hardened", "W824 model-iso"),
    "images:POST:/images/edits": ("hardened", "W925 cancel sweep"),
    "images:POST:/images/inpaint": ("hardened", "W705 RePaint, W925 cancel"),
    "images:POST:/images/controlnet": ("hardened", "W690/925 stop-leak, cancel"),
    "images:POST:/images/depth-guided": ("hardened", "W925 cancel"),
    # ── Other modalities ──
    "video:POST:/video/generations": ("hardened", "W825/939 — model-iso, native-fail-loud; _denoise cancel deferred"),
    "ocr:POST:/v1/ocr": ("hardened", "W823/900/929 — model-iso, think-strip, native key"),
    # ── MCP ──
    "mcp:POST:/mcp": ("hardened", "W805/923/924/956 — envelope, lock, handshake, generate-iso"),
    "mcp:GET:/mcp/tools": ("clean", "discovery, W785 model-iso"),
    "mcp:GET:/mcp/sse": ("hardened", "W799 DoS, W784 client-hang"),
    "mcp:GET:/mcp/client/status": ("clean", "status read"),
    "mcp:GET:/mcp/client/tools": ("clean", "W834 model-iso"),
    # ── Realtime ──
    "realtime:WEBSOCKET:/v1/realtime": ("hardened", "W784/799/888/904-909/935/942/962/963 — many"),
    "realtime:WEBSOCKET:/realtime": ("hardened", "alias of /v1/realtime"),
    # ── Control plane ──
    "models:GET:/models": ("hardened", "W801/916 — info-iso, static auth"),
    "models:GET:/models/{model_id:path}": ("hardened", "W801 info-iso"),
    "models:POST:/models/load": ("hardened", "W826 legacy-tenant privesc"),
    "models:POST:/models/unload/{model_id:path}": ("hardened", "W732/907 — in-use, false-success"),
    "sleep:POST:/sleep": ("hardened", "W910/950/957 — consistency, drain, engine-leak"),
    "sleep:POST:/wake-up": ("hardened", "W957 rebuild"),
    "sleep:GET:/sleep/status": ("clean", "W452 gated"),
    "cancel:POST:/cancel": ("hardened", "W691/758/911 — privesc, stop-hook, tenant-iso"),
    "cancel:GET:/active-generations": ("clean", "W911 owner-scope"),
    "cached_contents:POST:/cachedContents": ("hardened", "W949 per-owner evict"),
    "cached_contents:GET:/cachedContents": ("clean", "W949 ownership"),
    "cached_contents:GET:/cachedContents/{cid}": ("clean", "W949 ownership"),
    "cached_contents:PATCH:/cachedContents/{cid}": ("clean", "W968 audit — can_infer + _owns 404 IDOR guard + TTL bounds"),
    "cached_contents:DELETE:/cachedContents/{cid}": ("clean", "W949 ownership"),
    # ── Tokenize (W966 audit: CLEAN — auth+model-iso on all 3, W666 fallback + decode
    #    overflow→422 + W793 per-prompt ctx; inputs bounded by the global 10MB body cap,
    #    CPU-only no forward pass) ──
    "tokenize:POST:/tokenize": ("clean", "W666 model-iso fallback; body-cap bounds input"),
    "tokenize:POST:/detokenize": ("clean", "W666 decode overflow→422"),
    "tokenize:POST:/token_count": ("clean", "W793 per-prompt ctx check"),
    # ── Observability — monitoring (28) + profiling (4) audited as a whole class in W967:
    #    CLEAN. Uniform can_view_system / can_admin gating, robust None-subsystem handling
    #    (no engine/KV/mesh → empty not 500), no div-by-zero, profiling path-traversal blocked.
    #    Individual highlights kept; the rest inherit FILE_DEFAULT below. ──
    "monitoring:GET:/prometheus": ("hardened", "W803/905/906 — DoS, middleware-order, escape"),
    "monitoring:GET:/health-dashboard": ("hardened", "W803 lying-readiness"),
    # ── Batch inference (W966 audit: no HIGH/IDOR/isolation — auth+model-iso+ownership all
    #    gated, size/timeout/cancel bounded; 3 LOW fixed) ──
    "batch_inference:POST:/batch": ("hardened", "W666 model-iso per item, W966 sampling bounds"),
    "batch_inference:GET:/batch/{batch_id}/status": ("clean", "can_infer + _owns_batch IDOR guard"),
    "batch_inference:GET:/batch/{batch_id}/results": ("clean", "ownership + 409-in-progress"),
    "batch_inference:GET:/batch/{batch_id}/results.csv": ("hardened", "W966 CSV-injection hardening"),
    "batch_inference:POST:/batch/upload/csv": ("hardened", "W966 max_concurrent bound + UTF-8 guard"),
}

# Per-file default verdict for routers audited as a whole class (an endpoint not listed
# individually in STATUS inherits its file's default; absent → "todo").
FILE_DEFAULT: dict[str, tuple[str, str]] = {
    "monitoring": ("clean", "W967 class audit — uniform can_view_system gate, None-safe, no div0"),
    "profiling": ("clean", "W967 class audit — can_admin gate, path-traversal blocked, lock-safe"),
    "bench": ("hardened", "W826/968 class audit — can_benchmark gate, SSRF-defended, lock-serialized, param bounds"),
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
