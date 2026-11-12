"""CPU checks for the Phase B HTTP measurement and its identity verdict."""

import pytest
from scripts.research import phaseb_conc as p
from scripts.research import prompt_len

SMALL_TOK = "/Volumes/P5Plus/models/Qwen3.5-0.8B-MLX-bf16"


class _WordTok:
    """Word-level stand-in for the HF tokenizer: one token per word, a one-token template overhead."""

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids):
        return " ".join(ids)

    def apply_chat_template(
        self, msgs, tokenize=True, add_generation_prompt=True, **kw
    ):
        ids = ["<tpl>"]
        for m in msgs:
            ids += m["content"].split()
        return ids


@pytest.fixture
def no_disk_tok(monkeypatch):
    # run_cell loads its tokenizer for the length fit; the CPU suite must never read a model directory
    monkeypatch.setattr(prompt_len, "load_tokenizer", lambda path=None: _WordTok())
    # the corpus lives on P5Plus, which the ci-local sandbox cannot read
    from scripts.research import conc_probe

    monkeypatch.setattr(
        conc_probe, "filler", lambda chars, salt: prompt_len.filler_source(chars)
    )


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
                    ctx_tokens=32768,
                    scenario="burst2",
                    shape="seeded",
                    arm=arm,
                    rep=rep,
                    concurrent=dict(
                        aggregate_tps=2 * tps,
                        per_row={
                            "A": {"tps": tps, "start": 0.0},
                            "B": {"tps": tps, "start": 0.1},
                        },
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
    assert cfg["phaseb_ctx_tokens"] == [4096]
    assert cfg["phaseb_scenarios"] == ["burst2"]
    assert cfg["phaseb_tokens"] == 128


def test_prefill_rate_counts_stalls_at_both_edges():
    # A is stalled for most of B's ten-second prefill. Arrival-only rates
    # incorrectly report 10 tok/s for the two adjacent tokens.
    a = dict(times=[0, 5, 5.1, 15])
    b = dict(t0=1, times=[11, 12])
    assert p.interval_rates(a, b)["prefill_tps"] == pytest.approx(0.2)


def test_saved_identity_evidence_rechecks_solo_hash():
    r = dict(
        arm="cand",
        scenario="solo",
        shape="seeded",
        solo={"per_row": {"A": {"digest": "solo"}}},
        concurrent={"per_row": {"A": {"digest": "changed"}}},
    )
    assert not p.validate_records(
        [r, {"complete": True, "success": True}],
        arms=("cand",),
        scenarios=("solo",),
        shapes=("seeded",),
    )[0]


def test_stagger_waits_for_running_row_first_token(monkeypatch, no_disk_tok):
    import threading
    from types import SimpleNamespace

    from scripts.research import sampledspec_conc as sampled

    newcomer = threading.Event()
    calls = []

    def stream(url, body, zero, *, on_first_token=None):
        calls.append(on_first_token)
        if len(calls) <= 2:  # matched solo references
            pass
        elif on_first_token is not None:
            assert not newcomer.is_set()
            on_first_token()
            assert newcomer.wait(timeout=2)
        else:
            newcomer.set()
        return dict(
            times=[1, 2, 3],
            t0=0,
            ttft=1,
            usage={"completion_tokens": 3, "prompt_tokens": 64},
            digest="same",
        )

    monkeypatch.setattr(sampled, "stream", stream)
    a = SimpleNamespace(
        tokens=128, join=0, ctx_tokens=64, mode="identity", tokenizer=SMALL_TOK
    )
    result = p.run_cell(
        SimpleNamespace(model="test", url="unused"), a, "stagger", "seeded"
    )
    assert result["identity"] and newcomer.is_set()
    assert calls[:2] == [None, None]
    assert callable(calls[2]) and calls[3] is None


def test_validate_ignores_queue_device_stamps():
    stamp = {"device": "m5", "execution_device": "m5", "remote_host": None}
    side = {"per_row": {"A": {"digest": "id"}, **stamp}, **stamp}
    r = dict(arm="cand", scenario="solo", shape="greedy", solo=side, concurrent=side)
    assert p.validate_records(
        [r, {"complete": True, "success": True}],
        arms=("cand",),
        scenarios=("solo",),
        shapes=("greedy",),
    )[0]


def test_base_arm_mismatch_is_recorded_not_fatal(monkeypatch, no_disk_tok):
    from types import SimpleNamespace

    from scripts.research import sampledspec_conc as sampled

    n = iter(range(100))
    monkeypatch.setattr(
        sampled,
        "stream",
        lambda *a, **k: dict(
            times=[1, 2, 3],
            t0=0,
            ttft=1,
            usage={"completion_tokens": 3, "prompt_tokens": 64},
            digest=f"d{next(n)}",
        ),
    )
    a = SimpleNamespace(
        tokens=128, join=0, ctx_tokens=64, mode="identity", tokenizer=SMALL_TOK
    )
    server = SimpleNamespace(model="test", url="unused")
    r = p.run_cell(server, a, "solo", "greedy", strict=False)
    assert r["identity"] is False
    with pytest.raises(RuntimeError):
        p.run_cell(server, a, "solo", "greedy")


def test_validate_skips_digest_equality_for_base_arm():
    side = lambda d: {"per_row": {"A": {"digest": d}}}  # noqa: E731
    r = dict(
        arm="base",
        scenario="solo",
        shape="greedy",
        solo=side("a"),
        concurrent=side("b"),
    )
    assert p.validate_records(
        [r, {"complete": True, "success": True}],
        arms=("base",),
        scenarios=("solo",),
        shapes=("greedy",),
    )[0]


def test_speed_verdict_reports_uncovered_stagger_instead_of_aborting():
    def cell(arm, rep, stagger):
        return dict(
            arm=arm,
            rep=rep,
            ctx_tokens=1,
            scenario="stagger",
            shape="greedy",
            solo={"per_row": {}},
            concurrent={
                "aggregate_tps": 10.0 + (arm == "cand"),
                "per_row": {"A": {"tps": 5.0, "start": 0.0}},
            },
            stagger=stagger,
        )

    ok = dict(prefill_tps=1.0, overlap_tps=1.0, after_tps=1.0)
    rows = [cell(a, r, ok) for a in ("base", "cand") for r in range(3)]
    assert p.speed_verdict(rows)[0] == []
    rows[0]["stagger"] = dict(ok, after_tps=None)
    assert any("stagger did not cover" in x for x in p.speed_verdict(rows)[0])


def test_default_draft_fits_the_served_model():
    from types import SimpleNamespace

    tb = SimpleNamespace(
        default_drafter=lambda model: "mtp"
    )  # Flash-Next: not the 27B drafter
    assert p.resolve_draft(SimpleNamespace(draft=None, model="m"), tb) == "mtp"
    assert p.resolve_draft(SimpleNamespace(draft="off", model="m"), tb) == "off"


def _burst_cell(arm, rep, first, tps, aggregate):
    # arrival rank 1 is the row that reached the server first; labels swap between arms
    per_row = {name: dict(tps=tps[i], start=0.1 * i) for i, name in enumerate(first)}
    return dict(
        arm=arm,
        rep=rep,
        ctx_tokens=8192,
        scenario="burst2",
        shape="greedy",
        solo={"per_row": {}},
        concurrent=dict(aggregate_tps=aggregate, per_row=per_row),
    )


def test_speed_verdict_pairs_rows_by_arrival_order_not_label():
    rows = []
    for rep in range(3):
        # base: A arrives first (fast); cand: B arrives first (fast), A second
        rows.append(_burst_cell("base", rep, ("A", "B"), (50.0, 10.0), 60.0))
        rows.append(_burst_cell("cand", rep, ("B", "A"), (50.0, 10.0), 61.0))
    assert p.speed_verdict(rows)[0] == []
    # a real first-arrival regression hits every cand rep, not one (median)
    for r in rows:
        if r["arm"] == "cand":
            r["concurrent"]["per_row"]["B"]["tps"] = 30.0
    assert any("per-row regression" in x for x in p.speed_verdict(rows)[0])


def test_speed_verdict_ignores_label_swap_without_slowdown():
    # no real slowdown: the fast first arrival is B in base and A in cand, so labels swap
    rows = []
    for rep in range(3):
        rows.append(_burst_cell("base", rep, ("B", "A"), (50.0, 10.0), 60.0))
        rows.append(_burst_cell("cand", rep, ("A", "B"), (50.0, 10.0), 61.0))
    assert p.speed_verdict(rows)[0] == []


def test_speed_verdict_fails_closed_without_request_start():
    rows = [
        _burst_cell(arm, rep, ("A", "B"), (20.0, 10.0), 30.0 + (arm == "cand"))
        for arm in ("base", "cand")
        for rep in range(3)
    ]
    for r in rows:
        for v in r["concurrent"]["per_row"].values():
            del v["start"]
    assert any("missing request start" in x for x in p.speed_verdict(rows)[0])
