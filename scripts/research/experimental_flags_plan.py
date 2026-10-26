#!/usr/bin/env python3
"""CPU-only experimental flag census: exact registry decide text and measurement ownership.

Split-invariant 27B prefill is a stable qualified kernel policy, not a new
experimental flag. Existing stock/lane, tile/stock and chunked/step options
retain their opt-outs; NEXT_TURN_PREFILL=off disables rehearsal only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from yunshu_engine import settings

EVIDENCE = {
    "YUNSHU_TOOL_GRAMMAR": (
        "tools",
        "claim_and_partial",
        [
            "settings.py decide: 12/12 replay identity and 0 malformed both ways",
            "/Volumes/P5Plus/yunshu-build/tcgrammar/bodies/",
            "audit-toolab-27b-* pending",
        ],
        "The registry records a partial replay result, but no complete audited three-client greedy/sampled on/off receipt was found. Native/prev jobs are compatibility comparisons, not a complete grammar flag matrix.",
    ),
    "YUNSHU_ROUND_DRIVER": (
        "driver",
        "partial",
        [
            "docs/reports/PERF_TREND.md:2026-10-01 and 2026-10-02 round driver",
            "scripts/research/validate_round_driver.sh",
            "OWNERS.md: i6-lead round_driver/*",
        ],
        "Historical single-request 87.3 vs 93.0 loses, while concurrency has value. Today's pre-contention numbers cannot promote a default. New stage rejects on slow single-request/parity; a positive stage runs the full validator three times. Image/int8/MoE completeness must also be verified before path deletion.",
    ),
    "YUNSHU_MTP_ROW_EXACT": (
        "row_exact",
        "partial",
        [
            "docs/research/runs/2026-09-28-matrix/invariant-packed-mtp-sweep.jsonl",
            "docs/research/runs/2026-09-28-matrix/mtp-depth-sweep-exact-fixed.jsonl",
            "audit-sweep-inv-* / audit-sweep-rowexact-* pending",
        ],
        "Old invariant receipts have parity and speed advantage, but current-source long-context paired decision is incomplete. Sweep now emits full-token digests and a final complete record; new job repeats 1K/8K/32K/131K code+prose with AR references.",
    ),
    "YUNSHU_GPU_SAMPLER": (
        "gpu_sampler",
        "pending",
        [
            "scripts/research/ab_gpu_sampler.py",
            "tests/unit/test_gpu_sampler.py (distribution equivalence on a toy vocabulary)",
            "scripts/realmodel/smoke_gpu_sampler.py (one-prompt smoke, not a decision)",
        ],
        "No real-checkpoint receipt: the per-token host sync of the NumPy sampler was measured on CPU only (3.7-21.7 ms per token at 152K-248K vocabulary, host side). Needs the Mac A/B on a text-only checkpoint (decode tok/s, first-token TV against the numpy-vs-numpy noise floor, seeded reruns) before the default can change.",
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
