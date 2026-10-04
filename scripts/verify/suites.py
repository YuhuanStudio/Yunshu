"""Suites: the ladder a candidate climbs. `--suite decode` or `--suite smoke,identity,speed`."""

from __future__ import annotations

STAGES = ("preflight", "smoke", "identity", "apc", "quality", "speed", "memory")

# Every key is a default; the CLI can override ctx / reps / mmlu_n / mem_sizes.
SUITES = {
    # decode-path change (kernels, spec decode, sampler): identity incl. spec on == off, speed
    "decode": {
        "stages": ["preflight", "smoke", "identity", "apc", "speed"],
        "ctx": [1024, 8192],
        "spec_off": True,
        "reps": 3,
    },
    # prefill / prefix-cache change: long contexts, TTFT, cache hit == miss, quality
    "prefill": {
        "stages": ["preflight", "smoke", "identity", "apc", "quality", "speed"],
        "ctx": [1024, 8192, 32768],
        "spec_off": False,
        "reps": 3,
        "mmlu_n": 200,
    },
    "scheduler": {
        "stages": ["preflight", "smoke", "identity", "apc", "quality", "speed"],
        "ctx": [1024, 8192],
        "spec_off": False,
        "reps": 3,
        "mmlu_n": 200,
    },
    "memory": {
        "stages": ["preflight", "smoke", "identity", "memory"],
        "ctx": [1024],
        "spec_off": False,
        "reps": 2,
    },
    "full": {
        "stages": list(STAGES),
        "ctx": [1024, 8192, 32768],
        "spec_off": True,
        "reps": 3,
        "mmlu_n": 200,
    },
    # small-model dry runs of the tool itself (and any CPU-light change): minutes, not hours
    "tiny": {
        "stages": list(STAGES),
        "ctx": [1024],
        "spec_off": False,
        "reps": 3,
        "mmlu_n": 12,
        "mem_sizes": [4096, 8192],
        "mem_reps": 1,
    },
}

DEFAULTS = {
    "ctx": [1024, 8192],
    "kinds": ["prose", "code"],
    "spec_off": False,
    "reps": 3,
    "mmlu_n": 200,
    "mem_sizes": [8192, 32768, 98304],
    "mem_reps": 2,
    "speed_tol_pct": 2.0,
    "mem_tol_pct": 3.0,
    "quality_allowed": 1,
}


def parse_suite(spec: str) -> dict:
    """`decode` -> that suite; `smoke,identity` -> an ad-hoc ladder (kept in canonical order)."""
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty suite")
    if len(parts) == 1 and parts[0] in SUITES:
        cfg = dict(DEFAULTS)
        cfg.update(SUITES[parts[0]])
        cfg["name"] = parts[0]
        return cfg
    unknown = [p for p in parts if p not in STAGES]
    if unknown:
        raise ValueError(
            f"unknown suite/stage {unknown}; suites {sorted(SUITES)}, stages {list(STAGES)}"
        )
    cfg = dict(DEFAULTS)
    cfg["stages"] = [s for s in STAGES if s in parts]
    cfg["name"] = ",".join(cfg["stages"])
    return cfg
