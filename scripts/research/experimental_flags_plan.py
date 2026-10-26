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
    "YUNSHU_APC_SEAM_CHECKPOINT": (
        "apc_seam",
        "pending",
        [
            "scripts/research/ab_seam_checkpoint.py",
            "tests/unit/test_apc_seam_checkpoint.py (plan, alignment, media-safety, tiny-hybrid restore == cold)",
            "TensorFold #1262 / Rapid-MLX 421bf658 claim 125.9 s -> 9.0 s first token at 35k tokens on Qwen3.8-27B (their engines, not Yunshu)",
        ],
        "No Yunshu receipt: the seam cut is exact on the tiny hybrid model on CPU, but the TTFT gain, the first-request copy cost and the memory growth on Qwen3.8-27B need the Mac A/B (16k / 35k / 64k documents, off vs on).",
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
