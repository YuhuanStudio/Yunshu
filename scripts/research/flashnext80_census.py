"""CPU-only per-tensor census of a qwen4_exp MLX pack (safetensors headers only).

Usage: flashnext80_census.py MODEL_DIR [--json OUT]
Buckets bytes (decimal GB) by component; reports quantization bits/group of routed experts.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

COMPONENTS = (
    ("ple", re.compile(r"ngram_embedding")),
    ("mtp", re.compile(r"(^|\.)mtp[._]")),
    ("vision", re.compile(r"(vision|visual)")),
    ("routed_experts", re.compile(r"(switch_mlp|experts)\.")),
    ("shared_expert", re.compile(r"shared_expert")),
    ("linear_attn", re.compile(r"linear_attn")),
    ("self_attn", re.compile(r"self_attn")),
    ("lm_head", re.compile(r"lm_head")),
    ("embed", re.compile(r"embed_tokens")),
)


def classify(name: str) -> str:
    for comp, rx in COMPONENTS:
        if rx.search(name):
            return comp
    return "other"


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def tensor_bytes(entry: dict) -> int:
    a, b = entry["data_offsets"]
    return b - a


def census(model_dir: Path) -> dict:
    idx = model_dir / "model.safetensors.index.json"
    files = (
        sorted(set(json.loads(idx.read_text())["weight_map"].values()))
        if idx.exists()
        else [p.name for p in sorted(model_dir.glob("*.safetensors"))]
    )
    buckets: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    for fn in files:
        for name, entry in read_header(model_dir / fn).items():
            if name == "__metadata__":
                continue
            c = classify(name)
            buckets[c] += tensor_bytes(entry)
            counts[c] += 1
    total = sum(buckets.values())
    cfg = json.loads((model_dir / "config.json").read_text())
    q = cfg.get("quantization") or cfg.get("text_config", {}).get("quantization") or {}
    expert_bits: dict[str, int] = defaultdict(int)
    for k, v in q.items():
        if isinstance(v, dict) and re.search(r"(switch_mlp|experts)", k):
            expert_bits[f"{v.get('bits')}b/g{v.get('group_size')}"] += 1
    return {
        "files": len(files),
        "total_bytes": total,
        "total_gb": total / 1e9,
        "components_gb": {
            k: v / 1e9 for k, v in sorted(buckets.items(), key=lambda x: -x[1])
        },
        "tensor_counts": dict(counts),
        "resident_without_ple_gb": (total - buckets.get("ple", 0)) / 1e9,
        "expert_projection_quant": dict(expert_bits),
        "mtp_tensors": counts.get("mtp", 0),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = census(Path(a.model_dir))
    print(json.dumps(res, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
