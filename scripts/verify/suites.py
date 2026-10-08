"""Suites: the ladder a candidate climbs. `--suite decode` or `--suite smoke,identity,speed`."""

from __future__ import annotations

STAGES = (
    "preflight",
    "smoke",
    "identity",
    "apc",
    "quality",
    "speed",
    "memory",
    "longqa",
    "conc",
    "modelprobe",
    "rerank",
)

# `full` and `tiny` climb the original seven; the long stages (needle retrieval, concurrent
# sub-agents) need the 32K-128K prompt files and belong to the `long` suite.
LADDER = tuple(s for s in STAGES if s not in ("longqa", "conc", "modelprobe", "rerank"))

# Every key is a default; the CLI can override ctx / reps / mmlu_n / mem_sizes.
SUITES = {
    "rerank": {"stages": ["preflight", "rerank"]},
    # decode-path change (kernels, spec decode, sampler): identity incl. spec on == off, speed
    "decode": {
        "stages": ["preflight", "smoke", "identity", "apc", "speed"],
        "ctx": [1024, 8192],
        # spec on == off is opt-in (--spec-off): on main the server-level 'off' path differs
        # from MTP and DFlash in every 27B 1K cell (docs/research/notes/BACKLOG.md)
        "spec_off": False,
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
        "stages": list(LADDER),
        "ctx": [1024, 8192, 32768],
        "spec_off": False,
        "reps": 3,
        "mmlu_n": 200,
    },
    # iteration: fast, base cached across candidates; run `full` before merge
    "quick": {
        "stages": ["preflight", "smoke", "identity", "apc", "quality", "speed"],
        "ctx": [1024, 8192],
        "spec_off": False,
        "reps": 3,
        "mmlu_n": 200,
    },
    # long requests (agentic 32K-128K sessions): identity incl. spec on == off over 2048-token
    # replies, cold/warm/follow-up TTFT + 2048-token decode, 32K/128K memory, needle retrieval,
    # 2 concurrent sub-agents. One server cell per (ctx, kind) so each job stays under 20 min.
    "long": {
        "stages": ["preflight", "identity", "apc", "speed", "memory", "longqa", "conc"],
        "ctx": [32768, 65536, 131072],
        "spec_off": True,
        "reps": 2,
        "decode_tokens": 2048,
        "turn2_tokens": 256,  # the follow-up measures TTFT; keeps 131K cells under 20 min
        "split_cells": True,
        "long_ask": True,
        "keep_going": True,  # a regression in one stage must not hide the others
        "mem_sizes": [32768, 131072],
        "mem_reps": 2,
        "speed_tol_pct": 3.0,
    },
    # release trend: one rep of the headline long cells, no noise estimate (compare tags pairwise)
    "longtrend": {
        "stages": ["preflight", "identity", "apc", "memory", "longqa"],
        "ctx": [32768, 131072],
        "kinds": ["prose"],
        "spec_off": True,
        "reps": 1,
        "decode_tokens": 2048,
        "turn2_tokens": 256,
        "split_cells": True,
        "long_ask": True,
        "keep_going": True,
        "mem_sizes": [131072],
        "mem_reps": 1,
        "apc_require_hit": False,
    },
    # small-model dry runs of the tool itself (and any CPU-light change): minutes, not hours
    "tiny": {
        "stages": list(LADDER),
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
    "quality_max_tokens": 2048,
    "quality_thinking": False,
    "reuse_base_speed": False,
    "decode_tokens": 256,
    "turn2_tokens": 0,
    "conc_tol_pct": 10.0,
    "split_cells": False,
    "long_ask": False,
    "mem_sizes": [8192, 32768, 98304],
    "mem_reps": 2,
    "speed_tol_pct": 2.0,
    "speed_confirm_reps": 2,
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
