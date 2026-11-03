"""scripts/research/fullcmp_manifest.py: the user-approved manifest is unique, costed, honest, and under 48 h."""

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


def test_engine_list_is_the_approved_twelve_plus_ollama():
    assert fm.ENGINE_IDS == [
        "yunshu",
        "tf-1.2.1",
        "tf-1.1.8",
        "splash-1.5.0",
        "omlx-0.7.3",
        "omlx-0.7.3rc1",
        "mlx-serve-26.11.1",
        "mlx-lm-server",
        "mlx-vlm-server",
        "llama-cpp-b12061",
        "lmstudio",
        "kapitsa-0.2.26",
        "ollama",
    ]
    assert "tf-1.2.0" not in fm.ENGINE_IDS
    assert len([e for e in fm.ENGINE_IDS if e != "ollama"]) == 12


def test_no_dropped_scope_and_no_secondary_models():
    rows, _ = _rows()
    ids = " ".join(r["id"] for r in rows)
    for banned in ("256K", "c16", "o8192", "flashnext", "q27-8bit", "q27-bf16", "soak", "B3", "B6"):
        assert banned not in ids, banned
    assert {r["batch"] for r in rows} == {"B1", "B2", "B4", "B5"}
    assert {r["agent"] for r in rows if r["batch"] == "B5"} == {"claude"}
    assert {r["ctx_tokens"] for r in rows if r["table"] == "decode"} <= set(fm.CTX_TOKENS.values())


def test_b1_is_the_core_cross_at_conc_one_for_every_engine():
    rows, _ = _rows()
    for eng in fm.ENGINE_IDS:
        b1 = [r for r in rows if r["batch"] == "B1" and r["engine"] == eng]
        assert len(b1) == 10, eng
        assert {(r["ctx"], r["prompt_kind"]) for r in b1} == {
            (c, k) for c in fm.CORE_CTX for k in fm.CORE_KINDS
        }
        assert all(r["conc"] == 1 and r["out_len"] == 512 for r in b1)
        assert all(r["phases"] == ["cold", "warm", "fu"] for r in b1)


def test_b2_has_every_requested_sweep_point():
    rows, _ = _rows()
    yun = [r for r in rows if r["batch"] == "B2" and r["engine"] == "yunshu"]
    conc = {(r["ctx"], r["conc"]) for r in yun if r["sweep"] == "conc"}
    assert conc == {(c, n) for c in ("8K", "32K") for n in (2, 4, 8)}
    assert {r["ctx"] for r in yun if r["sweep"] == "ctx"} == {"4K", "16K"}
    assert {(r["ctx"], r["out_len"]) for r in yun if r["sweep"] == "out"} == {
        (c, o) for c in ("1K", "32K") for o in (128, 2048)
    }
    assert {r["prompt_kind"] for r in yun if r["sweep"] == "kind"} == {"tool-chat", "cjk", "json-schema"}
    assert [r["thinking"] for r in yun if r["sweep"] == "thinking"] == ["on"]
    assert {r["ctx"] for r in yun if r["sweep"] == "sampling"} == {"1K", "32K"}
    assert {r["ctx"] for r in yun if r["sweep"] == "idle5"} == {"8K", "32K", "128K"}
    assert [r["ctx"] for r in yun if r["sweep"] == "restart"] == ["32K"]
    mixed = [r for r in yun if r["sweep"] == "mixed"]
    assert len(mixed) == 1 and mixed[0]["ctx"] == "128K" and mixed[0]["conc"] == 8
    assert {r["media"] for r in yun if r["sweep"] == "image"} == {"img1", "img4"}
    assert all(r["ctx"] == "1K" for r in yun if r["sweep"] == "image")


def test_ids_are_unique():
    rows, _ = _rows()
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_every_row_has_a_status_and_reason_and_zero_cost_when_unsupported():
    rows, _ = _rows()
    for r in rows:
        assert r["status"] in (fm.UNV, fm.UNS, fm.CUT), r["id"]
        assert r["reason"], r["id"]
        if r["status"] == fm.UNS:
            assert r["est_gpu_min"] == 0.0, r["id"]
        else:
            assert r["est_gpu_min"] > 0, r["id"]


def test_unsupported_rows_come_from_census_flags_only():
    rows, _ = _rows()
    x = [r for r in rows if r["status"] == fm.UNS]
    assert all(
        r["engine"] == "ollama"
        or (r["engine"] == "mlx-lm-server" and (r.get("media") != "none" or r.get("prompt_kind") == "json-schema" or r.get("turns") == "restart" or r.get("agent") == "claude" or r.get("check") in ("anthropic_api", "responses_api", "json_validity")))
        for r in x
    )
    ollama = [r for r in rows if r["engine"] == "ollama"]
    assert ollama and all(r["status"] == fm.UNS and "GGUF" in r["reason"] for r in ollama)
    lm = [r for r in rows if r["engine"] == "mlx-lm-server" and r["status"] == fm.UNS]
    assert lm, "mlx-lm-server must show its census gaps as unsupported rows"
    assert not [r for r in rows if r["engine"] == "yunshu" and r["status"] == fm.UNS]


def test_serial_engine_is_costed_not_dropped():
    rows, _ = _rows()
    tf118 = [r for r in rows if r["engine"] == "tf-1.1.8" and r["sweep"] == "conc" and r["conc"] == 8]
    assert len(tf118) == 2 and all(r["status"] == fm.UNV and r["est_gpu_min"] > 0 for r in tf118)


def test_b4_and_b5_shape():
    rows, _ = _rows()
    b4 = [r for r in rows if r["batch"] == "B4"]
    assert len(b4) == 7 * len(fm.ENGINE_IDS)
    n_tasks = len(fm.agent_tasks())
    b5 = [r for r in rows if r["batch"] == "B5"]
    assert {r["engine"] for r in b5} == {"yunshu", fm.PLACEHOLDER}
    assert len(b5) == 2 * n_tasks  # one row per task; REPS is inside the cost
    assert all(r["reps"] == fm.REPS for r in b5)
    assert all(r["est_gpu_min"] >= fm.AGENT_RUN_MIN * fm.REPS for r in b5 if r["status"] == fm.UNV)  # plus a boot share


def test_agent_task_count_comes_from_the_agentbench_task_set():
    tasks = fm.agent_tasks()
    assert len(tasks) == len(set(tasks))
    assert sum(t.startswith("polyglot-") for t in tasks) == 12
    assert "fix-cart-discount" in tasks and "polyglot-grep" in tasks
    assert len(tasks) == 20


def test_yunshu_concurrency_is_measured_not_the_snapshot_artifact():
    assert fm.slowdown("yunshu", 8192, 1) == 1.0
    assert fm.slowdown("yunshu", 8192, 2) == 2.33
    assert fm.slowdown("yunshu", 32768, 2) == 2.90
    assert fm.slowdown("yunshu", 32768, 4) == 4.32
    assert fm.slowdown("yunshu", 32768, 8) == 4.36  # ctx-independent c8 point
    assert fm.slowdown("yunshu", 1024, 4) == 2.66  # nearest measured ctx is 8K
    assert fm.slowdown("yunshu", 131072, 2) == 3.2  # nearest measured ctx is 64K
    assert fm.ENGINE_PERF["yunshu"]["slow"] != {2: {None: 5.9}}
    assert fm.ENGINE_PERF["yunshu"]["source"].startswith("PROVISIONAL")


def test_fallback_and_serial_slowdown_are_explicit():
    assert fm.slowdown("tf-1.1.8", 8192, 4) == 4.0  # serial lone loop
    assert abs(fm.slowdown("kapitsa-0.2.26", 8192, 8) - (1 + 0.15 * 7)) < 1e-9
    assert fm.slowdown("mlx-lm-server", 32768, 2) == 1.19
    assert abs(fm.slowdown("mlx-lm-server", 32768, 8) - (1 + 0.15 * 7)) < 1e-9


def test_constants_cite_sources_and_flag_assumptions():
    for eng, perf in fm.ENGINE_PERF.items():
        assert perf["source"].strip(), eng
        if eng == "ollama":
            continue
        assert perf["ready_s"] > 0, eng
        if not perf["prefill"]:
            assert any("no measurement" in a for a in perf["assumed"]), eng
    for eng in ("tf-1.2.1", "splash-1.5.0", "mlx-serve-26.11.1", "omlx-0.7.3"):
        assert fm.ENGINE_PERF[eng]["assumed"], eng


def test_loglin_hits_measured_anchors_exactly():
    assert fm.prefill_tps("omlx-0.7.3rc1", 1024) == 772.0
    assert fm.decode_tps("tf-1.2.1", 32768) == 112.0
    assert fm.prefill_tps("yunshu", 32768) == 946.0


def test_cost_monotone_in_the_obvious_directions():
    base = dict(
        eng="yunshu",
        ctx_tokens=8192,
        conc=1,
        out=512,
        think=False,
        img_tokens=0,
        phases=("cold", "warm", "fu"),
    )
    cost = lambda **kw: fm.wall_s(**{**base, **kw})  # noqa: E731
    assert cost(ctx_tokens=131072) > cost(ctx_tokens=8192) > cost(ctx_tokens=1024)
    assert cost(conc=8) > cost(conc=4) > cost(conc=2) > cost(conc=1)
    assert cost(out=2048) > cost(out=512)
    assert cost(think=True) > cost(think=False)
    assert cost(phases=("cold", "fu_idle")) > cost(phases=("cold",))


def test_planned_total_after_cuts_is_reported_honestly_and_batches_add_up():
    rows, _ = _rows()
    buf = io.StringIO()
    fm.apply_cuts(rows, out=buf)
    total = fm.planned_h(rows)
    if total > fm.TARGET_GPU_H:  # over target after every cut: the remainder must be reported
        assert "STILL OVER by" in buf.getvalue()
    else:
        assert "STILL OVER" not in buf.getvalue()
    by_batch = sum(fm._h(rows, b) for b in fm.BATCH_NAMES)
    assert abs(by_batch - fm.planned_h(rows)) < 1e-6


def test_cuts_apply_in_order_only_while_over_target(monkeypatch):
    rows, _ = _rows()
    buf = io.StringIO()
    fm.apply_cuts(rows, out=buf)
    total = fm.planned_h(rows)
    text = buf.getvalue()
    assert text.startswith("planned total before cuts")
    if total <= fm.TARGET_GPU_H:
        assert "no cut applied" in text
    else:
        assert "cut (1)" in text
    # every cut row keeps its id, cost and reason (never silently dropped)
    cut = [r for r in rows if r["status"] == fm.CUT]
    assert all(r["reason"].startswith("cut to fit") and r["est_gpu_min"] > 0 for r in cut)


def test_output_is_deterministic():
    a = [json.dumps(r, sort_keys=True) for r in _rows()[0]]
    b = [json.dumps(r, sort_keys=True) for r in _rows()[0]]
    assert a == b


def test_summary_prints_every_section():
    rows, _ = _rows()
    buf = io.StringIO()
    fm.summarize(rows, out=buf)
    text = buf.getvalue()
    for needle in ("GPU hours by engine and batch", "Per batch", "GRAND TOTAL", "unverified"):
        assert needle in text
    for b in fm.BATCH_NAMES:
        assert b in text
    for eng in fm.ENGINE_IDS:
        assert f"| {eng} |" in text
