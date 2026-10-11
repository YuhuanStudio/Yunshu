"""Structural whitelist for metadata read from local serve logs."""

from __future__ import annotations


def extras(raw):
    from .serve_log import _num, label

    spec = raw.get("speculative") or {}
    if not isinstance(spec, dict):
        spec = {}
    depths = spec.get("per_depth") or []
    depth_rows = []
    if isinstance(depths, list):
        for row in depths[:128]:
            if isinstance(row, dict):
                depth_rows.append(
                    {
                        k: _num(row.get(k))
                        for k in ("position", "drafted", "accepted", "acceptance_rate")
                    }
                )
    structured = raw.get("structured_output") or {}
    if not isinstance(structured, dict):
        structured = {}
    latency = raw.get("latency") or {}
    if not isinstance(latency, dict):
        latency = {}
    stages = (
        "gateway_receive",
        "model_lease_start",
        "model_lease",
        "gateway_admit",
        "engine_submit",
        "engine_admit",
        "template_start",
        "template_end",
        "apc_start",
        "apc_end",
        "prefill_start",
        "prefill_end",
        "first_decode",
        "sse_first_flush",
    )
    durations = (
        "model_lease",
        "gateway_admit",
        "engine_queue",
        "template_tokenize",
        "apc_lookup_restore",
        "prefill",
        "first_decode",
        "sse_first_flush",
    )

    def numbers(key, fields):
        values = latency.get(key)
        return (
            {k: _num(values.get(k)) for k in fields if k in values}
            if isinstance(values, dict)
            else {}
        )

    energy = raw.get("energy")
    if isinstance(energy, dict):
        safe_energy = {
            "schema": label(energy.get("schema")),
            "method": label(energy.get("method")),
            **{
                k: energy.get(k) is True
                for k in (
                    "includes_other_processes",
                    "overlapping_requests_share_host_energy",
                )
            },
        }
        for phase in ("prefill", "decode"):
            receipt = energy.get(phase)
            if isinstance(receipt, dict):
                safe_energy[phase] = {
                    "state": label(receipt.get("state")),
                    "reason": receipt.get("reason")
                    if receipt.get("reason")
                    in (
                        "phase timing unavailable",
                        "phase not fully covered by valid samples",
                    )
                    else None,
                    **{
                        k: _num(receipt.get(k))
                        for k in (
                            "joules",
                            "joules_per_token",
                            "gpu_watts_mean",
                            "coverage_ratio",
                            "extrapolated_s",
                        )
                    },
                }
        energy = safe_energy
    else:
        energy = None
    cache = raw.get("cache") if isinstance(raw.get("cache"), dict) else {}
    reasons = raw.get("reasons") if isinstance(raw.get("reasons"), dict) else {}
    copy = spec.get("copy") if isinstance(spec.get("copy"), dict) else {}
    out = {
        "cache": {
            "tier": label(cache.get("tier")),
            "cached_tokens": _num(cache.get("cached_tokens")),
            "reload_ms": _num(cache.get("reload_ms")),
            **({"device": label(cache.get("device"))} if "device" in cache else {}),
        },
        "reasons": {k: label(reasons.get(k)) for k in ("cache", "spec")},
        "speculative": {
            **{k: label(spec.get(k)) for k in ("mode", "position_basis")},
            **{
                k: _num(spec.get(k))
                for k in ("drafted", "accepted", "acceptance_rate", "rounds")
            },
            "per_depth": depth_rows,
            **(
                {"copy": {k: _num(copy.get(k)) for k in ("rounds", "tokens")}}
                if copy
                else {}
            ),
        },
        "structured_output": {
            **{k: structured.get(k) is True for k in ("requested", "enforced")},
            **{
                k: label(structured.get(k))
                for k in ("engine", "grammar_backend", "reason")
            },
        },
        "latency": {
            "milestones_ms": numbers("milestones_ms", stages),
            "durations_ms": numbers("durations_ms", durations),
            "clock": label(latency.get("clock")),
            "flush_boundary": label(latency.get("flush_boundary")),
        },
        "energy": energy,
    }
    for key in ("cache", "reasons", "speculative", "structured_output", "latency"):
        if not isinstance(raw.get(key), dict):
            out[key] = None
    return out
