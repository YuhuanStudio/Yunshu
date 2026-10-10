#!/usr/bin/env python3
"""CPU-only experimental flag census: exact registry decide text and measurement ownership."""

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
    "YUNSHU_SPEC_NODES": (
        "spec_nodes",
        "partial",
        [
            "/Volumes/P5Plus/yunshu-build/decode16/depth/",
            "/Volumes/P5Plus/yunshu-build/decode16/budget/",
        ],
        "Pins the DFlash fast-tree nodes per round for the depth/cost experiment against TensorFold. The cost-aware budget fix (count-based, rank-monotone landing estimates; copy rounds excluded) reached full depth on code and lifted cold decode 3-7% with identical output, so the pin has no remaining value; delete it after the final yv.",
    ),
    "YUNSHU_ENGINE_LOOP": (
        "k02",
        "pending_owner",
        [
            "/Volumes/P5Plus/yunshu-build/codex/k02_matrix.sh",
            "/Volumes/P5Plus/yunshu-build/codex/k02_progress.md",
            "gpuq 1002-215141-00-k02-matrix-quiet",
        ],
        "codex-k02 owns the shared-runner/fast decision and prepared close-flags branch. Reuse its existing p-1 quiet Qwen2.5 4bit/bf16 token and concurrency matrix; do not duplicate or mutate its files.",
    ),
    "YUNSHU_OVERLAP": (
        "k02",
        "pending_owner",
        [
            "/Volumes/P5Plus/yunshu-build/codex/k02_matrix.sh",
            "/Volumes/P5Plus/yunshu-build/codex/k02_progress.md",
            "gpuq 1002-215141-00-k02-matrix-quiet",
        ],
        "Coupled to ENGINE_LOOP: a qualified shared-runner win removes the loop and both overlap arms. Current queued owner matrix is the unification comparison; no separate overlap timing claim is made.",
    ),
    "YUNSHU_SPEC_UNVERIFIED": (
        "external",
        "cpu_correctness_only",
        [
            "tests/unit/test_external_lm_reachability.py",
            "tests/unit/test_external_draft_greedy.py",
            "commits 4de2b6d2 / ca853629",
        ],
        "Ordinary Qwen2 target/draft is reachable without EAGLE training. Three CPU counterexamples exposed ratio acceptance on greedy requests; fixed to target argmax. Unsupported request options fall back. No real-model speed/parity receipt yet.",
    ),
    "YUNSHU_DRAFT_MODEL": (
        "external",
        "cpu_correctness_only",
        [
            "/Volumes/P5Plus/models/Qwen2.5-3B-Instruct-bf16",
            "/Volumes/P5Plus/models/Qwen2.5-3B-Instruct-4bit",
            "scripts/research/audit_external_draft.py --dry-run verified",
        ],
        "Same-tokenizer ordinary target/draft pair is available. CPU preflight validates configs/vocabularies/imports; successful model smoke must precede three alternating traced-parity and uninstrumented-timing repetitions.",
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
