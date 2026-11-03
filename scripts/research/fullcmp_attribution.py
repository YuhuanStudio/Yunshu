#!/usr/bin/env python3
"""Attribution fields for the full comparison (FULLCMP_SPEC.md, "Attribution data").

Two layers, recorded in the same run:

  layer 1  non-perturbing, on in every timed rep. Per-request stage timestamps and counters.
           Stored on each decode row under ``attr`` (value) and ``attr_unavailable``
           (field -> reason). Session header carries ``engine_log_path`` and ``server_flags``.
  layer 2  perturbing, one extra "profile rep" per cell group (label ``...-prof``), never used
           for table numbers. Stored as ``part: "profile"`` rows with ``stages`` (Yunshu stage
           timers with syncs, host-sync counter, op census) and ``competitor_stats`` (the
           engine's own debug/stats lines, as-is).

Fail-closed rule: a field is satisfied only by a value under ``attr`` (or the profile row) or by
an explicit ``attr_unavailable`` reason. A missing field is MISSING and the ``attribution_fields``
audit guard fails the cell. Nothing is estimated.

Per-engine status comes from ``ENGINE_OVERRIDES``; ``unavailable`` entries there are the
documented reasons a competitor cannot expose a field, and are recorded per request.
"""

from __future__ import annotations

from dataclasses import dataclass

REQUIRED = "required"
OPTIONAL = "optional"

# ---------------------------------------------------------------------------
# Layer 1: per-request fields (decode rows). Name -> (status, description).
# ---------------------------------------------------------------------------
LAYER1_TIMESTAMPS = {
    "t_receive": (REQUIRED, "client: request sent (epoch s)"),
    "t_template": (REQUIRED, "server: chat template applied (epoch s)"),
    "t_tokenize": (REQUIRED, "server: prompt tokenized (epoch s)"),
    "t_admit": (REQUIRED, "server: admitted to the generation queue (epoch s)"),
    "t_cache_lookup": (REQUIRED, "server: prefix-cache lookup done (epoch s)"),
    "t_restore": (OPTIONAL, "server: cache restore done, if any (epoch s)"),
    "t_prefill_chunks": (
        REQUIRED,
        "server: list of [start_s, end_s, tokens] per prefill chunk",
    ),
    "t_first_token": (REQUIRED, "client: first content token received (epoch s)"),
    "t_rounds": (
        REQUIRED,
        "server: list of per-round end times (epoch s); [] for non-spec engines with per-step rows",
    ),
    "t_last_token": (REQUIRED, "client: last token received (epoch s)"),
    "t_response_end": (REQUIRED, "client: response complete (epoch s)"),
}
LAYER1_COUNTERS = {
    "rounds": (REQUIRED, "int: decode rounds (steps for non-spec engines)"),
    "tokens_per_round": (REQUIRED, "list[int]: accepted tokens per round"),
    "rows_per_round": (REQUIRED, "list[int]: batch rows per round"),
    "drafted": (OPTIONAL, "int: drafted tokens (spec engines)"),
    "accepted": (OPTIONAL, "int: accepted draft tokens (spec engines)"),
    "copy_rounds": (OPTIONAL, "int: copy-lookup rounds (Yunshu n-gram/copy path)"),
    "tree_rounds": (OPTIONAL, "int: tree-verify rounds (DFlash tree)"),
    "cache_hit_tokens": (
        REQUIRED,
        "int: prompt tokens served from cache (server-reported 'cached')",
    ),
    "chunk_sizes": (REQUIRED, "list[int]: prefill chunk sizes"),
    "host_gaps_ms": (
        REQUIRED,
        "list[float]: gaps between consecutive GPU submissions (from timestamps)",
    ),
    "engine_log_path": (
        REQUIRED,
        "str: the engine's own log file for this request's server, verbose level recorded",
    ),
}

# ---------------------------------------------------------------------------
# Layer 2: profile-rep fields (part "profile" rows). Yunshu-only stage timers and counters.
# ---------------------------------------------------------------------------
LAYER2_YUNSHU = {
    "stages.decode_round.drafter": (REQUIRED, "ms per round, synced"),
    "stages.decode_round.plan": (REQUIRED, "ms per round, synced"),
    "stages.decode_round.copy_lookup": (REQUIRED, "ms per round, synced"),
    "stages.decode_round.verify_weights": (REQUIRED, "ms per round, synced"),
    "stages.decode_round.verify_attention": (REQUIRED, "ms per round, synced"),
    "stages.decode_round.commit": (REQUIRED, "ms per round, synced"),
    "stages.prefill.chunk_compute": (REQUIRED, "ms, synced"),
    "stages.prefill.eval_wait": (REQUIRED, "ms, synced"),
    "stages.prefill.apc_store": (REQUIRED, "ms, synced"),
    "stages.prefill.drafter_context": (REQUIRED, "ms, synced"),
    "stages.ttft.template": (REQUIRED, "ms"),
    "stages.ttft.lookup": (REQUIRED, "ms"),
    "stages.ttft.restore": (REQUIRED, "ms"),
    "stages.ttft.prefill": (REQUIRED, "ms"),
    "stages.ttft.drafter_rebuild": (REQUIRED, "ms"),
    "stages.ttft.first_round": (REQUIRED, "ms"),
    "host_sync_count": (REQUIRED, "int: host syncs per step (counter)"),
    "op_census": (REQUIRED, "dict: op name -> count per step"),
}
LAYER2_COMPETITOR_NOTE = "competitor's own debug/stats output, recorded as-is under competitor_stats (raw lines)"


@dataclass(frozen=True)
class EngineAttr:
    """Per-engine status for the fields that can differ across engines."""

    engine: str
    # field -> reason, for layer-1 server-side fields this engine does not expose.
    unavailable: dict
    # layer-2 status for this engine: "stages" (Yunshu-style stage timers) or "stats_as_is".
    layer2: str


_NO_SERVER_STAGES = "not exposed by the engine's HTTP API or log at this verbosity (client-side fields only)"
_NO_RESTORE = "engine has no separate cache-restore step"
_NO_SPEC = "engine does not run speculative decoding in this mode"
_NO_CENSUS = "engine has no op-census hook; profile rep records competitor_stats only"

_COMPETITOR_SERVER_STAGES = {
    f: _NO_SERVER_STAGES
    for f in (
        "t_template",
        "t_tokenize",
        "t_admit",
        "t_cache_lookup",
        "t_prefill_chunks",
        "t_rounds",
        "rounds",
        "tokens_per_round",
        "rows_per_round",
        "chunk_sizes",
        "host_gaps_ms",
    )
}

# Yunshu exposes server stages as x_yunshu.latency.milestones_ms (ms since gateway arrival) and
# x_yunshu.speculative (round totals, per-depth, copy rounds). The rest is not exposed: merged
# spans, per-round lists and GPU submission times. tfbench maps the exposed milestones to
# t_template, t_admit and t_cache_lookup (client t_receive + milestone ms); per-row reasons cover
# the rest. Nothing is estimated.
_YUNSHU_UNAVAILABLE = {
    "t_tokenize": "template and tokenize are one milestone span (template_start/template_end)",
    "t_restore": "cache lookup and restore are one milestone span (apc_start/apc_end)",
    "t_rounds": "per-round end times not exposed; x_yunshu.speculative has only round totals",
    "t_prefill_chunks": "per-chunk timestamps not exposed; milestones give prefill_start/prefill_end only",
    "tokens_per_round": "per-round accepted counts not exposed; only totals (accepted / rounds)",
    "rows_per_round": "per-round batch row counts not exposed (single-request lane)",
    "chunk_sizes": "prefill chunk sizes not exposed by the engine",
    "host_gaps_ms": "GPU submission timestamps not exposed by the engine",
}

ENGINE_OVERRIDES = {
    "yunshu": EngineAttr("yunshu", dict(_YUNSHU_UNAVAILABLE), "stages"),
    "tf-1.2.1": EngineAttr("tf-1.2.1", dict(_COMPETITOR_SERVER_STAGES), "stats_as_is"),
    "tf-1.1.8": EngineAttr("tf-1.1.8", dict(_COMPETITOR_SERVER_STAGES), "stats_as_is"),
    "splash-1.5.0": EngineAttr(
        "splash-1.5.0", dict(_COMPETITOR_SERVER_STAGES), "stats_as_is"
    ),
    "omlx-0.7.3": EngineAttr(
        "omlx-0.7.3", dict(_COMPETITOR_SERVER_STAGES), "stats_as_is"
    ),
    "omlx-0.7.3rc1": EngineAttr(
        "omlx-0.7.3rc1", dict(_COMPETITOR_SERVER_STAGES), "stats_as_is"
    ),
    "mlx-serve": EngineAttr(
        "mlx-serve", dict(_COMPETITOR_SERVER_STAGES), "stats_as_is"
    ),
    "mlx-lm-server": EngineAttr(
        "mlx-lm-server",
        dict(_COMPETITOR_SERVER_STAGES, t_restore=_NO_RESTORE),
        "stats_as_is",
    ),
    "mlx-vlm-server": EngineAttr(
        "mlx-vlm-server",
        dict(_COMPETITOR_SERVER_STAGES, t_restore=_NO_RESTORE),
        "stats_as_is",
    ),
    "kapitsa": EngineAttr("kapitsa", dict(_COMPETITOR_SERVER_STAGES), "stats_as_is"),
    "llama-cpp": EngineAttr(
        "llama-cpp",
        dict(
            _COMPETITOR_SERVER_STAGES,
            t_restore=_NO_RESTORE,
            drafted=_NO_SPEC,
            accepted=_NO_SPEC,
        ),
        "stats_as_is",
    ),
}
# Yunshu's own counters that only exist in spec mode are optional there; the reason is the mode.
LAYER2_UNAVAILABLE_DEFAULT = _NO_CENSUS


def engine_attr(engine: str) -> EngineAttr:
    """Status for an engine id; an unknown engine is an error, never a silent default."""
    if engine not in ENGINE_OVERRIDES:
        raise KeyError(f"no attribution schema for engine {engine!r}")
    return ENGINE_OVERRIDES[engine]


def required_layer1(engine: str) -> list[str]:
    """Layer-1 fields that must be present or explicitly unavailable for this engine."""
    eng = engine_attr(engine)
    out = []
    for name, (status, _) in {**LAYER1_TIMESTAMPS, **LAYER1_COUNTERS}.items():
        if status == REQUIRED:
            out.append(name)
    # Engine-specific unavailability still needs an explicit reason row; it is not "required"
    # to be present as a value, so it is dropped from the MISSING set here.
    return [f for f in out if f not in eng.unavailable]


def required_layer2(engine: str) -> list[str]:
    """Layer-2 fields for the profile rep. Competitors: stats_as_is, only the rows exist."""
    eng = engine_attr(engine)
    if eng.layer2 == "stages":
        return [
            name for name, (status, _) in LAYER2_YUNSHU.items() if status == REQUIRED
        ]
    return ["competitor_stats"]


def check_request(engine: str, row: dict) -> dict:
    """Coverage of one decode row: field -> 'present' | 'unavailable' | 'MISSING'.

    ``row["attr"]`` holds values; ``row["attr_unavailable"]`` holds field -> reason. A reason
    supplied in the row is the engine's explanation for this request; the static
    ``ENGINE_OVERRIDES`` reason covers fields the engine never exposes.
    """
    eng = engine_attr(engine)
    values = row.get("attr") or {}
    reasons = dict(row.get("attr_unavailable") or {})
    cov = {}
    for field in required_layer1(engine) + [
        f for f in eng.unavailable if f in {**LAYER1_TIMESTAMPS, **LAYER1_COUNTERS}
    ]:
        if field in values and values[field] is not None:
            cov[field] = "present"
        elif field in eng.unavailable:
            cov[field] = f"unavailable ({eng.unavailable[field]})"
        elif reasons.get(field):
            cov[field] = f"unavailable ({reasons[field]})"
        else:
            cov[field] = "MISSING"
    return cov


def check_profile(engine: str, row: dict | None) -> dict:
    """Coverage of the profile row for one cell group: field -> status."""
    eng = engine_attr(engine)
    if row is None:
        return {f: "MISSING" for f in required_layer2(engine)}
    cov = {}
    if eng.layer2 == "stages":
        stages = row.get("stages") or {}
        for field in required_layer2(engine):
            if field.startswith("stages."):
                _, stage_name = field.split(".", 1)
                node = stages
                ok = True
                for part in stage_name.split("."):
                    if not isinstance(node, dict) or part not in node:
                        ok = False
                        break
                    node = node[part]
                cov[field] = "present" if ok and node is not None else "MISSING"
            else:
                cov[field] = "present" if row.get(field) is not None else "MISSING"
    else:
        cov["competitor_stats"] = (
            "present" if row.get("competitor_stats") else "MISSING"
        )
    return cov


def missing(cov: dict) -> list[str]:
    return sorted(k for k, v in cov.items() if v == "MISSING")


def check_cell(engine: str, rows: list[dict], profile: dict | None) -> dict:
    """Cell-level result: merged coverage, the MISSING list, and a pass flag.

    ``rows`` are the decode rows of one timed rep. Every row must satisfy the layer-1 rule;
    the profile row, when given, must satisfy layer 2. ``ok`` is False if anything is MISSING
    or if there are no rows at all (no evidence is not success).
    """
    if not rows:
        return {
            "ok": False,
            "missing": ["<no decode rows>"],
            "layer1": {},
            "layer2": {},
        }
    merged: dict = {}
    for row in rows:
        for k, v in check_request(engine, row).items():
            if merged.get(k) != "MISSING":
                merged[k] = v
    l1_missing = missing(merged)
    l2 = check_profile(engine, profile)
    l2_missing = missing(l2)
    return {
        "ok": not l1_missing and not l2_missing,
        "missing": l1_missing + [f"profile:{f}" for f in l2_missing],
        "layer1": merged,
        "layer2": l2,
    }
