"""Where unit tests find real small checkpoints.

Order: YUNSHU_TEST_MODELS, then M3_MODELS (set by the gpuq M3 lane: the M3's retained model
copies), then the M5 model disk, then ~/.yunshu/models. The first root that holds the named
checkpoint wins; if none does the first candidate is returned (callers skip on a missing path).
"""

from __future__ import annotations

import os
from pathlib import Path


def model_roots() -> list[Path]:
    roots = [
        os.environ.get("YUNSHU_TEST_MODELS"),
        os.environ.get("M3_MODELS"),
        "/Volumes/P5Plus/models",
        "~/.yunshu/models",
    ]
    return [Path(r).expanduser() for r in roots if r]


def model_dir(name: str) -> Path:
    roots = model_roots()
    for root in roots:
        if (root / name).exists():
            return root / name
    return roots[0] / name
