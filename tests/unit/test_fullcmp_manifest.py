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
    # 12 measured engines (yunshu-pb2 added 2026-11-04: the same curve as yunshu, release date = its
    # commit 4f99270cae40, 2026-11-03T08:20:53+08:00) + lmstudio (not measured) + ollama (unsupported)
    assert fm.ENGINE_IDS == [
        "yunshu-next",
        "yunshu-pb2",
        "yunshu",
        "tf-1.2.1",
        "tf-1.1.8",
        "splash-1.5.0",
        "omlx-0.7.3",
        "omlx-0.7.3rc1",
        "omlx-0.7.4.dev1",
        "mlx-serve-26.11.1",
        "mlx-lm-server",
        "mlx-vlm-server",
        "llama-cpp-b12061",
        "lmstudio",
        "kapitsa-0.2.26",
        "kapitsa-0.2.31",
        "ollama",
    ]
    assert "tf-1.2.0" not in fm.ENGINE_IDS
    # 12 measured + the extra oMLX 0.7.4.dev1 + the pinned Kapitsa 0.2.31 re-run + lmstudio
    assert len([e for e in fm.ENGINE_IDS if e != "ollama"]) == 16


def test_no_dropped_scope_and_no_secondary_models():
    rows, _ = _rows()
    ids = " ".join(r["id"] for r in rows)
    for banned in (
        "256K",
        "c16",
        "o8192",
        "flashnext",
        "q27-8bit",
        "q27-bf16",
        "soak",
        "B3",
        "B6",
    ):
        assert banned not in ids, banned
    assert {r["batch"] for r in rows} == {"B1", "B2", "B4", "B5"}
    assert {r["agent"] for r in rows if r["batch"] == "B5"} == {"claude"}
    assert {r["ctx_tokens"] for r in rows if r["table"] == "decode"} <= set(
        fm.CTX_TOKENS.values()
    )


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
    assert {r["prompt_kind"] for r in yun if r["sweep"] == "kind"} == {
        "tool-chat",
        "cjk",
        "json-schema",
    }
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
        # measured refusal (wire-1108 B2): TensorFold 1.2.1 refuses image parts as launched
        or (
            r["engine"] in ("tf-1.2.1", "tf-1.1.8", "llama-cpp-b12061")
            and r.get("media") != "none"
        )
        # restart cells of engines with no disk tier as launched (fullcmp_manifest.NO_DISK_TIER, evidence cited)
        or (r["engine"] in fm.NO_DISK_TIER and r.get("turns") == "restart")
        or (
            r["engine"] == "mlx-lm-server"
            and (
                r.get("thinking")
                == "on"  # census reasoning_split false: reasoning_content empty
                or r.get("media") != "none"
                or r.get("prompt_kind") == "json-schema"
                or r.get("turns") == "restart"
                or r.get("agent") == "claude"
                or r.get("check") in ("anthropic_api", "responses_api", "json_validity")
            )
        )
        for r in x
    )
    ollama = [r for r in rows if r["engine"] == "ollama"]
    assert ollama and all(
        r["status"] == fm.UNS and "GGUF" in r["reason"] for r in ollama
    )
    lm = [r for r in rows if r["engine"] == "mlx-lm-server" and r["status"] == fm.UNS]
    assert lm, "mlx-lm-server must show its census gaps as unsupported rows"
    assert not [r for r in rows if r["engine"] == "yunshu" and r["status"] == fm.UNS]


def test_serial_engine_is_costed_not_dropped():
    rows, _ = _rows()
    tf118 = [
        r
        for r in rows
        if r["engine"] == "tf-1.1.8" and r["sweep"] == "conc" and r["conc"] == 8
    ]
    assert len(tf118) == 2 and all(
        r["status"] == fm.UNV and r["est_gpu_min"] > 0 for r in tf118
    )


def test_b4_and_b5_shape():
    rows, _ = _rows()
    b4 = [r for r in rows if r["batch"] == "B4"]
    assert len(b4) == 7 * len(fm.ENGINE_IDS)
    n_tasks = len(fm.agent_tasks())
    b5 = [r for r in rows if r["batch"] == "B5"]
    assert {r["engine"] for r in b5} == {"yunshu", fm.PLACEHOLDER}
    assert len(b5) == 2 * n_tasks  # one row per task; REPS is inside the cost
    assert all(r["reps"] == fm.REPS for r in b5)
    assert all(
        r["est_gpu_min"] >= fm.AGENT_RUN_MIN * fm.REPS
        for r in b5
        if r["status"] == fm.UNV
    )  # plus a boot share


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
    assert (
        "fullcmp_cell_timing.json" in fm.ENGINE_PERF["yunshu"]["source"]
    )  # measured, not snapshot


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
    # Yunshu anchors are the measured B1 planning values (fullcmp_cell_timing.json), not snapshot014
    assert (
        fm.prefill_tps("yunshu", 32768)
        == fm.TIMING["yunshu"]["planning"]["prefill_tps"]["32768"]["value"]
    )
    assert (
        fm.prefill_tps("yunshu", 131072)
        == fm.TIMING["yunshu"]["planning"]["prefill_tps"]["131072"]["value"]
    )


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
    if (
        total > fm.TARGET_GPU_H
    ):  # over target after every cut: the remainder must be reported
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
    if "cut (" not in text:
        assert "no cut applied" in text
        assert total <= fm.TARGET_GPU_H
    else:
        # cuts applied in order; within target, or the summary says it is still over (the ranked proposal)
        assert total <= fm.TARGET_GPU_H or "STILL OVER" in text
    # every cut row keeps its id, cost and reason (never silently dropped)
    cut = [r for r in rows if r["status"] == fm.CUT]
    assert all(
        r["reason"].startswith("cut to fit") and r["est_gpu_min"] > 0 for r in cut
    )


def test_output_is_deterministic():
    a = [json.dumps(r, sort_keys=True) for r in _rows()[0]]
    b = [json.dumps(r, sort_keys=True) for r in _rows()[0]]
    assert a == b


def test_summary_prints_every_section():
    rows, _ = _rows()
    buf = io.StringIO()
    fm.summarize(rows, out=buf)
    text = buf.getvalue()
    for needle in (
        "GPU hours by engine and batch",
        "Per batch",
        "GRAND TOTAL",
        "unverified",
    ):
        assert needle in text
    for b in fm.BATCH_NAMES:
        assert b in text
    for eng in fm.ENGINE_IDS:
        assert f"| {eng} |" in text


# --- measured cell timing (fullcmp_cell_timing.json, job 1103-220358) -------------------------


def _leaves_with_value(node, path=""):
    if isinstance(node, dict):
        if "value" in node:
            yield path, node
        for k, v in node.items():
            yield from _leaves_with_value(v, f"{path}/{k}")


def test_every_timing_number_carries_its_kind_and_source():
    for path, leaf in _leaves_with_value(fm.TIMING):
        assert leaf.get("kind"), path
        assert str(leaf.get("source", "")).strip(), path


def test_measured_yunshu_cells_reproduce_the_wall_fit():
    # wall = ready + cold + warm + turn2 + overhead, within 1 s, for every complete cell
    cells = fm.TIMING["yunshu"]["measured"]
    n = 0
    for ctx, kinds in cells.items():
        for kind, e in kinds.items():
            if not e["complete"]:
                continue
            n += 1
            fit = (
                e["ready_s"]["value"]
                + e["cold_total_s"]["value"]
                + e["warm_total_s"]["value"]
                + e["turn2_total_s"]["value"]
                + fm.OVERHEAD_S
            )
            assert abs(fit - e["job_wall_s"]["value"]) < 1.0, (ctx, kind)
    assert n == 7  # 64K prose is the truncated cell, not measured


def test_overhead_is_the_measured_idle_plus_shutdown():
    assert abs(fm.OVERHEAD_S - 30.6) < 0.1


def test_a_cell_pays_its_own_server_start_and_memory_idle():
    rows, _meta = fm.build_rows(engines=["yunshu"])
    r = next(
        x
        for x in rows
        if x["id"] == "B1.core.yunshu.1K.c1.code.o512.single.th-off.none.greedy"
    )
    wall = fm.wall_s("yunshu", 1024, 1, 512, False, 0, ("cold", "warm", "fu"))
    want = fm.REPS * (wall + fm.ENGINE_PERF["yunshu"]["ready_s"] + fm.OVERHEAD_S) / 60.0
    assert abs(r["est_gpu_min"] - want) < 0.01


def test_warm_phase_pays_the_reply_decode():
    warm = fm.wall_s("yunshu", 32768, 1, 512, False, 0, ("warm",))
    dec = fm.decode_tps("yunshu", 32768)
    assert warm > 512 / dec


def test_yunshu_and_pb2_have_the_same_post_cut_row_set_per_batch():
    # the comparison is the point of pb2: after the cut step both engines keep and cut the same cells
    rows, _ = fm.build_rows()
    fm.apply_cuts(rows, out=io.StringIO())
    for batch in ("B1", "B2", "B4"):

        def post(engine):
            return sorted(
                (r["id"].replace(f".{engine}.", ".E."), r["status"])
                for r in rows
                if r["batch"] == batch and r["engine"] == engine
            )

        assert post("yunshu") == post("yunshu-pb2"), batch


def test_yunshu_pb2_rows_mirror_yunshu_main_cell_for_cell():
    # yunshu-pb2 (phaseb2-main 4f99270) is the same engine config: its rows mirror yunshu's per batch
    rows, _ = fm.build_rows()
    for batch in (
        "B1",
        "B2",
        "B4",
    ):  # B5 (agents) runs only yunshu and the placeholder, by design
        a = sorted(
            r["id"].replace(".yunshu-pb2.", ".yunshu.")
            for r in rows
            if r["batch"] == batch and r["engine"] == "yunshu-pb2"
        )
        b = sorted(
            r["id"] for r in rows if r["batch"] == batch and r["engine"] == "yunshu"
        )
        assert a == b, batch
    pb2 = [
        r for r in rows if r["engine"] == "yunshu-pb2" and r["status"] == "unverified"
    ]
    assert pb2, "yunshu-pb2 rows are in the real manifest"
