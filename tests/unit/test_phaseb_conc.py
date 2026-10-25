"""CPU checks for the Phase B HTTP measurement and its identity verdict."""

import pytest
from scripts.research import phaseb_conc as p


def test_plan_and_cli_are_cpu_only():
    a = p.parser().parse_args(
        ["--base-src", "/a", "--cand-src", "/b", "--out", "/unused"]
    )
    assert list(map(len, p.validate(a))) == [2, 4, 2]
    a.tokens = 1
    with pytest.raises(ValueError):
        p.validate(a)


def test_identity_fails_closed():
    with pytest.raises(RuntimeError):
        p.check_identity({"A": {"digest": "same"}}, {"A": {"digest": "changed"}})
    with pytest.raises(RuntimeError):
        p.check_identity({"A": {"digest": "same"}}, {})
    p.check_identity({"A": {"digest": "same"}}, {"A": {"digest": "same"}})


def test_metrics_include_running_row_during_prefill():
    a = dict(times=list(range(21)), ttft=0, usage={"completion_tokens": 21}, digest="a")
    b = dict(
        t0=5, times=[9, 10, 11], ttft=4, usage={"completion_tokens": 3}, digest="b"
    )
    assert p.interval_rates(a, b) == dict(
        prefill_tps=1.0, overlap_tps=1.0, after_tps=1.0
    )
    summary = p.summarize({"A": a, "B": b})
    assert summary["per_row"]["A"]["tps"] == 1.0
    assert summary["aggregate_tps"] == 1.2


def test_speed_verdict_requires_interleaved_pairs_and_wins():
    rows = []
    for rep in range(3):
        for arm, tps in [("base", 20.0), ("cand", 30.0)]:
            rows.append(
                dict(
                    ctx_chars=128000,
                    scenario="burst2",
                    shape="seeded",
                    arm=arm,
                    rep=rep,
                    concurrent=dict(
                        aggregate_tps=2 * tps,
                        per_row={"A": {"tps": tps}, "B": {"tps": tps}},
                    ),
                )
            )
    assert p.speed_verdict(rows)[0] == []
    assert p.speed_verdict(rows[:-1])[0]
    for r in rows:
        if r["arm"] == "cand":
            r["concurrent"]["aggregate_tps"] = 39.0
    assert p.speed_verdict(rows)[0]


def test_yv_registers_phaseb_without_expanding_default_ladder():
    from scripts.verify.stages import STAGE_FUNCS
    from scripts.verify.suites import LADDER, parse_suite

    assert "phaseb-identity" in STAGE_FUNCS and "phaseb-speed" in STAGE_FUNCS
    assert parse_suite("phaseb-identity")["stages"] == ["phaseb-identity"]
    assert not {"phaseb-identity", "phaseb-speed"} & set(LADDER)


def test_measurement_evidence_fails_closed():
    assert not p.validate_records([])[0]
    assert not p.validate_records([{"complete": True, "success": True}])[0]
    r = dict(
        arm="cand",
        scenario="solo",
        shape="greedy",
        solo={"per_row": {"A": {"digest": "id"}}},
        concurrent={"per_row": {"A": {"digest": "id"}}},
    )
    rows = [r, {"complete": True, "success": True}]
    assert p.validate_records(
        rows, arms=("cand",), scenarios=("solo",), shapes=("greedy",)
    )[0]
    assert not p.validate_records(
        [r, *rows], arms=("cand",), scenarios=("solo",), shapes=("greedy",)
    )[0]


def test_yv_pilot_uses_one_arm_cell_and_short_output():
    from scripts.verify.suites import parse_suite

    cfg = parse_suite("phaseb-pilot")
    assert cfg["phaseb_arms"] == ["cand"]
    assert cfg["phaseb_ctx_chars"] == [16000]
    assert cfg["phaseb_scenarios"] == ["burst2"]
    assert cfg["phaseb_tokens"] == 128
