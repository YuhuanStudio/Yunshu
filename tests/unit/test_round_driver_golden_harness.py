"""Fail-closed golden validation without MLX or a checkpoint."""

import copy

import pytest
from scripts.research.round_driver_golden import (
    GLOBAL_STATES,
    compare,
    digest,
    validate,
)


def receipt():
    payload = dict(
        layer="generic",
        device="cpu",
        tokens=4,
        globals=dict.fromkeys(GLOBAL_STATES, False),
        cells=[
            dict(
                mode=m,
                schedule=s,
                rows=[[dict(token=3, logprob=(-0.25).hex()) for _ in range(4)]],
            )
            for m in ("greedy", "seeded")
            for s in ("solo", "burst", "stagger", "cancel", "media", "repeat")
        ],
    )
    return dict(complete=True, payload=payload, sha256=digest(payload))


def test_exact_receipt_comparison():
    compare(receipt(), receipt())


@pytest.mark.parametrize(
    "damage", ["complete", "checksum", "matrix", "globals", "truncated", "nan"]
)
def test_missing_or_corrupted_evidence_fails_closed(damage):
    r = receipt()
    if damage == "complete":
        r["complete"] = False
    elif damage == "checksum":
        r["sha256"] = "bad"
    elif damage == "matrix":
        r["payload"]["cells"].pop()
    elif damage == "globals":
        r["payload"]["globals"].pop("ragged_kv._STATE")
    elif damage == "truncated":
        r["payload"]["cells"][0]["rows"][0].pop()
    else:
        r["payload"]["cells"][0]["rows"][0][0]["logprob"] = "nan"
    if damage != "checksum":
        r["sha256"] = digest(r["payload"])
    with pytest.raises(ValueError):
        validate(r)


def test_one_logprob_bit_changes_the_golden():
    a = receipt()
    b = copy.deepcopy(a)
    b["payload"]["cells"][0]["rows"][0][0]["logprob"] = (-0.25000000000000006).hex()
    b["sha256"] = digest(b["payload"])
    with pytest.raises(ValueError, match="golden mismatch"):
        compare(a, b)
