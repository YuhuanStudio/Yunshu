"""scripts/research/fullcmp_manifest.py: the designed-grid manifest is unique, costed from measured constants, and honest."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "fullcmp_manifest", ROOT / "scripts/research/fullcmp_manifest.py"
)
fm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fm)


def _rows():
    return fm.build_rows()


def test_grid_shape_per_engine():
    rows, _ = _rows()
    n_tasks = len(fm.agent_tasks())
    for eng in fm.ENGINE_IDS:
        rs = [r for r in rows if r["engine"] == eng]
        decode = [r for r in rs if r["table"] == "decode"]
        # core cross 20 + one-factor sweeps 31 + secondary models 12
        assert len(decode) == 63, eng
        assert len([r for r in rs if r["table"] == "behaviour"]) == len(
            fm.BEHAVIOUR_CHECKS
        )
        assert len([r for r in rs if r["table"] == "agent"]) == len(fm.AGENTS) * n_tasks
    assert len([r for r in rows if r["table"] == "soak"]) == 2


def test_core_cross_is_the_lead_grid():
    rows, _ = _rows()
    core = [r for r in rows if r["batch"] == "B1" and r["engine"] == "yunshu"]
    assert len(core) == 20
    combos = {(r["ctx"], r["conc"], r["prompt_kind"]) for r in core}
    assert combos == {
        (c, n, k) for c in fm.CORE_CTX for n in (1, 8) for k in ("code", "prose")
    }
    assert all(r["out_len"] == 512 and r["model"] == fm.PRIMARY for r in core)
    assert all(r["phases"] == ["cold", "warm", "fu"] for r in core)


def test_ids_are_unique():
    rows, _ = _rows()
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_unsupported_costs_nothing_and_every_row_has_a_reason():
    rows, _ = _rows()
    for r in rows:
        assert r["status"] in (fm.R, fm.U, fm.X), r["id"]
        assert r["reason"], r["id"]
        if r["status"] == fm.X:
            assert r["est_gpu_min"] == 0.0, r["id"]
        else:
            assert r["est_gpu_min"] > 0, r["id"]


def test_certain_unsupported_rows_only():
    rows, _ = _rows()
    x = [r for r in rows if r["status"] == fm.X]
    assert x, "ollama rows must exist as unsupported, not dropped"
    assert all(
        r["engine"] == "ollama"
        or (r["engine"] == "mlx-lm-server" and r.get("media") != "none")
        for r in x
    )
    ollama = [r for r in rows if r["engine"] == "ollama"]
    assert ollama and all(r["status"] == fm.X and "n/a" in r["reason"] for r in ollama)
    text_only = [
        r
        for r in rows
        if r["engine"] == "mlx-lm-server" and r.get("media", "none") != "none"
    ]
    assert len(text_only) == 2 and all(r["status"] == fm.X for r in text_only)


def test_measured_engines_have_runnable_cells_and_others_do_not_claim_them():
    rows, _ = _rows()
    yun = [r for r in rows if r["engine"] == "yunshu" and r["status"] == fm.R]
    assert {
        (r["ctx"], r["conc"], r["prompt_kind"]) for r in yun if r["batch"] == "B1"
    } == {
        (c, 1, k) for c in ("1K", "8K", "32K", "64K", "128K") for k in ("code", "prose")
    }
    assert not [r for r in rows if r["engine"] == "kapitsa" and r["status"] == fm.R]
    assert not [r for r in rows if r["batch"] == "B5" and r["status"] == fm.R]


def test_constants_cite_sources_and_flag_assumptions():
    for eng, perf in fm.ENGINE_PERF.items():
        assert perf["source"].strip(), eng
        assert perf["ready_s"] > 0, eng
        if eng == "ollama":  # unsupported: no cost, no curve
            continue
        if not perf["prefill"]:
            assert any("no measurement" in a for a in perf["assumed"]), eng
    assert fm.ENGINE_PERF["omlx-stable"]["assumed"], (
        "the stable build is a proxy and must be flagged"
    )


def test_loglin_hits_measured_anchors_exactly():
    assert fm.prefill_tps("omlx-rc", 1024) == 772.0
    assert fm.decode_tps("tf-1.2.0", 32768) == 112.0
    assert fm.prefill_tps("yunshu", 32768) == 946.0


def test_slowdown_is_monotone_and_measured_at_two_and_four():
    assert fm.slowdown("yunshu", 1) == 1.0
    assert fm.slowdown("yunshu", 2) == 5.9
    assert fm.slowdown("yunshu", 4) == 8.3
    assert (
        1.0
        < fm.slowdown("yunshu", 2)
        < fm.slowdown("yunshu", 4)
        < fm.slowdown("yunshu", 8)
        < fm.slowdown("yunshu", 16)
    )


def test_cost_monotone_in_the_obvious_directions():
    base = dict(
        eng="yunshu",
        model=fm.PRIMARY,
        ctx_tokens=8192,
        conc=1,
        out=512,
        think=False,
        img_tokens=0,
        phases=("cold", "warm", "fu"),
    )
    cost = lambda **kw: fm.wall_s(**{**base, **kw})  # noqa: E731
    assert cost(ctx_tokens=131072) > cost(ctx_tokens=8192) > cost(ctx_tokens=1024)
    assert cost(conc=8) > cost(conc=1)
    assert cost(out=2048) > cost(out=512)
    assert cost(think=True) > cost(think=False)
    assert cost(phases=("cold", "warm", "fu_idle")) > cost(
        phases=("cold", "warm", "fu")
    )


def test_batches_cover_every_row_and_sum_to_the_grand_total():
    rows, _ = _rows()
    assert {r["batch"] for r in rows} <= set(fm.BATCH_NAMES)
    total = sum(r["est_gpu_min"] for r in rows if r["status"] != fm.X) / 60
    by_batch = (
        sum(
            sum(
                r["est_gpu_min"]
                for r in rows
                if r["batch"] == b and r["status"] != fm.X
            )
            for b in fm.BATCH_NAMES
        )
        / 60
    )
    assert abs(total - by_batch) < 1e-6


def test_cut_plan_drops_each_row_once_and_never_goes_negative():
    rows, _ = _rows()
    live_h = sum(r["est_gpu_min"] for r in rows if r["status"] != fm.X) / 60
    plan = fm.cut_plan(rows)
    assert plan
    cut_h = sum(h for _, _, h, _ in plan)
    assert cut_h <= live_h + 1e-6
    assert abs(plan[-1][3] - (live_h - cut_h)) < 1e-6
    assert all(left >= -1e-9 for *_, left in plan)


def test_output_is_deterministic():
    a = [json.dumps(r, sort_keys=True) for r in _rows()[0]]
    b = [json.dumps(r, sort_keys=True) for r in _rows()[0]]
    assert a == b


def test_summary_prints_every_section():
    rows, _ = _rows()
    buf = io.StringIO()
    fm.summarize(rows, out=buf)
    text = buf.getvalue()
    for needle in ("Per engine", "Per batch", "GRAND TOTAL", "runnable", "unverified"):
        assert needle in text
    for b in fm.BATCH_NAMES:
        assert b in text
