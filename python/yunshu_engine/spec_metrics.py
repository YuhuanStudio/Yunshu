# Patches upstream mlx-vlm speculative.common._record_speculative_round (MIT, v0.7.6; vendor.json).
"""Host-only speculative counters. No token ids, arrays, synchronization or locks.

The single MLX thread binds the row while advancing its generator. Tree positions
mean depth (siblings share a denominator); chain positions are zero-based.
"""

from __future__ import annotations

from typing import Any

_active: Any = None
_totals: dict[str, Any] = {}
_original: Any = None


def bind(stats: Any) -> None:
    global _active
    _active = stats


def record_depth(stats: Any, accepted: int, drafted: int, parents=None) -> None:
    if parents is None:
        counts = [1] * drafted
    else:
        depths = [0]
        counts = []
        for parent in parents[1:]:
            depth = depths[parent] + 1
            depths.append(depth)
            while len(counts) < depth:
                counts.append(0)
            counts[depth - 1] += 1
    while len(stats.spec_depth_drafted) < len(counts):
        stats.spec_depth_drafted.append(0)
        stats.spec_depth_accepted.append(0)
    for pos, count in enumerate(counts):
        stats.spec_depth_drafted[pos] += count
        stats.spec_depth_accepted[pos] += int(pos < accepted)


def record(drafter, accepted, drafted, *, parents=None):
    if _original is not None:
        _original(drafter, accepted, drafted)
    st = _active
    if st is None:
        return
    observe(st, accepted, drafted, parents=parents)


def observe(st, accepted, drafted, *, parents=None):
    if not hasattr(st, "spec_depth_drafted"):
        st.spec_depth_drafted, st.spec_depth_accepted = [], []
    record_depth(st, int(accepted), int(drafted), parents)
    mode = getattr(st, "spec_mode", None) or "unknown"
    from types import SimpleNamespace

    total = _totals.get(mode)
    if total is None:
        total = _totals[mode] = SimpleNamespace(
            spec_depth_drafted=[],
            spec_depth_accepted=[],
            rounds=0,
            drafted=0,
            accepted=0,
        )
    record_depth(total, int(accepted), int(drafted), parents)
    total.rounds += 1
    total.drafted += int(drafted)
    total.accepted += int(accepted)


def depth_rows(stats):
    return [
        {
            "position": pos,
            "drafted": n,
            "accepted": a,
            "acceptance_rate": a / n if n else None,
        }
        for pos, (n, a) in enumerate(
            zip(
                getattr(stats, "spec_depth_drafted", []),
                getattr(stats, "spec_depth_accepted", []),
                strict=False,
            )
        )
    ]


def snapshot() -> dict:
    return {
        "object": "yunshu.speculative",
        "scope": "process",
        "position_basis": "depth",
        "data": [
            {
                "mode": mode,
                "num_drafts": t.rounds,
                "num_draft_tokens": t.drafted,
                "num_accepted_tokens": t.accepted,
                "per_depth": depth_rows(t),
            }
            for mode, t in list(_totals.items())
        ],
    }


def install():
    """Hook upstream chain loops without changing generation or its existing counters."""
    global _original
    if _original is not None:
        return
    import sys

    from mlx_vlm.speculative import common

    _original = common._record_speculative_round
    for name, module in list(sys.modules.items()):
        if (
            name.startswith("mlx_vlm.speculative")
            and getattr(module, "_record_speculative_round", None) is _original
        ):
            module._record_speculative_round = record
    common._record_speculative_round = record
