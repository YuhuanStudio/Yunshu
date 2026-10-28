#!/usr/bin/env python3
"""CPU-only experimental flag census: exact registry decide text and measurement ownership.

Split-invariant 27B prefill is a stable qualified kernel policy, not a new
experimental flag. Console DOCS_ASK limits are stable request bounds, not
experimental inference paths; retrieval never loads an embedding model. Existing stock/lane, tile/stock and chunked/step options
retain their opt-outs; NEXT_TURN_PREFILL=off disables rehearsal only.
Native ANE phase 1 is an explicit research API, with no new setting/flag or serving hook.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yunshu_engine import settings

EVIDENCE = {
    "YUNSHU_SDPA_CHUNK": (
        "sdpa_chunk",
        "pending",
        [
            "tests/unit/test_sdpa_chunk.py (chunked == stock bit for bit on CPU, gate and install logic)",
            "omlx sdpa256_attention (their fused route; Yunshu keeps the stock fallback per chunk)",
        ],
        "No Metal receipt: exactness on Apple GPUs rests on batched matmul without split-K, which CPU cannot show; needs the M1-M4 bitwise A/B and peak-memory measurement.",
    ),
    "YUNSHU_QWEN4_LANE": (
        "lane",
        "partial",
        [
            "settings.py decide: dense projection microbench, Flash-Next oQ3e (probe_singleton.py)",
            "scripts/research/metalrt_step_profile.py flashverify --lane",
            "scripts/research/check_virtual_group.py",
        ],
        "Per-projection microbench shows the lane matmul flat in rows where upstream's per-row singleton qmv scales with them; the verify-forward timing and the engine identity (spec on == off) receipts are the open items.",
    ),
}


def plan():
    flags = {
        name: spec
        for name, spec in settings.REGISTRY.items()
        if spec.stability == "experimental"
    }
    if set(flags) != set(EVIDENCE):
        raise ValueError(
            f"census needs update: new={set(flags) - set(EVIDENCE)}, retired={set(EVIDENCE) - set(flags)}"
        )
    rows = []
    for name, spec in flags.items():
        area, status, evidence, finding = EVIDENCE[name]
        job = [
            "/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python",
            "scripts/research/audit_flag_matrix.py",
            "--out",
            "/Volumes/P5Plus/yunshu-build/codex/audit-flags/timing.json",
            "--require-smoke",
            "/Volumes/P5Plus/yunshu-build/codex/audit-flags/smoke.json",
        ]
        if area == "gpu_sampler":
            job = [
                job[0],
                "scripts/research/ab_gpu_sampler.py",
                "--checkpoint",
                "<text-only mlx-lm checkpoint>",
                "--out",
                "/Volumes/P5Plus/yunshu-build/codex/audit-flags/gpu_sampler.json",
            ]
        if area == "apc_seam":
            job = [
                job[0],
                "scripts/research/ab_seam_checkpoint.py",
                "--checkpoint",
                "<Qwen3.8-27B checkpoint>",
                "--out",
                "/Volumes/P5Plus/yunshu-build/codex/audit-flags/apc_seam.json",
            ]
        if area == "k02":
            job = ["sh", "/Volumes/P5Plus/yunshu-build/codex/k02_matrix.sh"]
        rows.append(
            {
                "flag": name,
                "default": spec.default,
                "added": spec.added,
                "decide": spec.decide,
                "evidence_status": status,
                "evidence": evidence,
                "finding": finding,
                "measurement_area": area,
                "measurement_argv": job,
                "priority": -1,
                "timing_quiet": True,
                "decision": "retain pending qualified measurement; no default changed",
            }
        )
    return {
        "experimental_count": len(rows),
        "max_experimental": settings.MAX_EXPERIMENTAL,
        "flags": rows,
        "consolidation": "All six audit-owned flag settings use one smoke and one timing wrapper; K02 uses its existing queued owner job. No legacy audit job is cancelled or reordered.",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data = plan()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {"experimental_count": data["experimental_count"], "out": str(args.out)}
        )
    )


if __name__ == "__main__":
    main()
