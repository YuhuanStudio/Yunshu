"""The ``kld`` stage: teacher-forced KL divergence of base and candidate against a reference.

Pure helpers (cache keys, cache completeness, loading, the verdict) live here; stages.py turns them into cells.
Cache: /Volumes/Micron/yunshu-kld-cache/<key>/<window id>.npz, key = sha of (engine source key, model identity,
the arm's environment, corpus sha, capture mode, tool version): the base arm runs once per main SHA.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

TOOL_VERSION = "1"
CACHE_ROOT = Path("/Volumes/Micron/yunshu-kld-cache")
NOISE_FILE = Path(__file__).with_name("noise.json")
DEFAULT_EPS = 0.002
DEFAULT_TOP1_NOISE = 0.003
POSITION_SLACK = 128  # the server retokenizes the window text (and adds its own prefix tokens): a few dozen more


def cache_key(
    arm: str,
    source_key: str,
    model_identity: str,
    env: dict,
    corpus_sha: str,
    mode: str = "prefill",
) -> str:
    payload = json.dumps(
        [
            TOOL_VERSION,
            arm,
            source_key,
            model_identity,
            sorted(env.items()),
            corpus_sha,
            mode,
        ],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def cache_dir(key: str, root: Path = CACHE_ROOT) -> Path:
    return root / key


def missing_windows(directory: Path, manifest: dict) -> list[str]:
    return [
        w["id"]
        for w in manifest["windows"]
        if not (directory / f"{w['id']}.npz").is_file()
    ]


def load_arm(directory: Path, manifest: dict) -> dict:
    """Every manifest window of an arm; fails closed on a missing or malformed window (numpy: run under the
    engine interpreter, see analyze.py; yv itself runs on the system python)."""
    import numpy as np

    from . import metrics

    gone = missing_windows(directory, manifest)
    if gone:
        raise RuntimeError(f"{directory.name}: incomplete, missing windows {gone[:5]}")
    out = {}
    for w in manifest["windows"]:
        with np.load(directory / f"{w['id']}.npz") as z:
            window = metrics.Window(z["ids"], z["lp"], z["next_lp"], z["tokens"])
        window.check()
        if abs(window.ids.shape[0] - w["tokens"]) > POSITION_SLACK:
            raise RuntimeError(
                f"{w['id']}: {window.ids.shape[0]} positions, manifest says {w['tokens']}"
            )
        out[w["id"]] = window
    return out


def noise() -> dict:
    """eps and top-1 noise of the verdict: measured values from noise.json when present (their source is
    recorded there), the defaults otherwise."""
    if NOISE_FILE.is_file():
        doc = json.loads(NOISE_FILE.read_text())
        return dict(
            eps=float(doc["eps"]),
            top1_noise=float(doc["top1_noise"]),
            source=doc.get("source", ""),
        )
    return dict(
        eps=DEFAULT_EPS,
        top1_noise=DEFAULT_TOP1_NOISE,
        source="default (no measured noise.json)",
    )


def evaluate(ref: dict, base: dict, cand: dict, eps: float, top1_noise: float) -> dict:
    """Metrics of base and cand against ref, cand against base, and the verdict."""
    from . import metrics

    ref_vs_base = metrics.summarize(ref, base)
    ref_vs_cand = metrics.summarize(ref, cand)
    ok, reasons = metrics.verdict(ref_vs_base, ref_vs_cand, eps, top1_noise)
    return dict(
        passed=ok,
        reasons=reasons,
        eps=eps,
        top1_noise=top1_noise,
        ref_vs_base=ref_vs_base,
        ref_vs_cand=ref_vs_cand,
        base_vs_cand=metrics.summarize(base, cand),
    )
