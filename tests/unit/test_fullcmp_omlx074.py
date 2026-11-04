"""oMLX 0.7.4.dev1 as an extra engine (2026-11-04): same recipe as 0.7.3, INT8-activation prefill saved OFF.

0.7.4.dev1 turns INT8-activation (A8) prefill ON by default on M5 (model_settings.py:216-222, :509 of the wheel) and it
changes numerics (:337-343), so the measured configuration is lossless: a8_prefill_enabled=false in the model settings.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "research"))
import bench_engines as be  # noqa: E402
import fullcmp_audit_manifest as fam  # noqa: E402
import fullcmp_engaged as fe  # noqa: E402
import fullcmp_manifest as fm  # noqa: E402
import fullcmp_provenance as fp  # noqa: E402
import fullcmp_run as fr  # noqa: E402

E074, E073 = "omlx-0.7.4.dev1", "omlx-0.7.3"
FIX = Path(__file__).parent / "fixtures"


def _launch(engine, parallel=1):
    return be.build_launch(engine, 18990, "/tmp/h", {}, parallel=parallel)


def _settings(launch):
    (path,) = [p for p in launch.files if p.endswith("model_settings.json")]
    return json.loads(launch.files[path])


def test_the_launch_runs_the_0_7_4_binary_with_the_0_7_3_flags():
    a, b = _launch(E074), _launch(E073)
    assert a.cmd[0] == be.OMLX_BIN_074DEV1 and a.cmd[0].endswith("venvs/omlx-0.7.4.dev1/bin/omlx")
    assert a.cmd[1:] == b.cmd[1:]  # every flag unchanged: --paged-ssd-cache-dir, --max-concurrent-requests, ...
    assert "--paged-ssd-cache-dir" in a.cmd and "--max-concurrent-requests" in a.cmd


def test_a8_prefill_is_saved_off_in_a_version_2_settings_file_and_0_7_3_is_unchanged():
    s74, s73 = _settings(_launch(E074)), _settings(_launch(E073))
    m74 = s74["models"]["Qwen3.8-27B-oQ4e-mtp"]
    assert s74["version"] == 2 and m74["a8_prefill_enabled"] is False
    assert m74["dflash_enabled"] is True and m74["dflash_draft_model"] == be.DRAFTER and m74["mtp_enabled"] is False
    assert s73 == {"version": 1, "models": {"Qwen3.8-27B-oQ4e-mtp": {
        "dflash_enabled": True, "dflash_draft_model": be.DRAFTER, "mtp_enabled": False}}}
    assert be.omlx_model_settings("m")["version"] == 1 and "a8_prefill_enabled" not in be.omlx_model_settings("m")["models"]["m"]


def test_the_recorded_flags_carry_the_a8_setting_and_the_settings_hash():
    launch = _launch(E074)
    (path,) = [p for p in launch.files if p.endswith("model_settings.json")]
    assert launch.flags["a8_prefill_enabled"] is False
    assert launch.flags["model_settings_sha256"] == hashlib.sha256(launch.files[path].encode()).hexdigest()
    assert "a8_prefill_enabled" not in _launch(E073).flags


def test_concurrency_flag_and_registry_entries_mirror_omlx_073():
    assert be.PARALLEL_FLAG[E074] == "--max-concurrent-requests"
    assert _launch(E074, parallel=16).cmd[_launch(E074, parallel=16).cmd.index("--max-concurrent-requests") + 1] == "16"
    assert be.tfbench_engine(E074) == E074 and be.ENGINES[E074].expected_mode == "dflash"
    assert be.ENGINES[E074].checkpoint == be.ENGINES[E073].checkpoint and be.ENGINES[E074].drafter == be.ENGINES[E073].drafter


def test_the_census_pin_has_the_release_timestamp_sha_and_binary():
    census = fp.load_json(fp.CENSUS_PATH) if Path(fp.CENSUS_PATH).is_file() else None
    if census is None or "latest_dev_date" not in census["omlx"]:
        pytest.skip("census with the 0.7.4.dev1 entry not on this machine")
    om = census["omlx"]
    assert om["latest_dev_date"] == "2026-11-04T10:13:41Z" and om["release_dates"]["0.7.4.dev1"] == "2026-11-04"
    assert om["sha256"]["0.7.4.dev1 wheel cp313"] == "40ba8e231dea8473686e0e8bd6f4f6f133c479da1586384e96817b85a80d2f4f"
    assert "4213c3284abc7b3746a91464a7af7ccc3732b759" in om["release_date_source"]["0.7.4.dev1"]
    for key in ("a8_prefill", "evidence_lines", "launch_flags", "prefill_accelerator"):
        assert om["census_0_7_4_dev1"][key]
    assert "model_settings.py:216-222" in om["census_0_7_4_dev1"]["a8_prefill"]


def test_the_evidence_entry_reads_the_real_omlx_lines_like_0_7_3rc1():
    """The MTP / Chat completion / late-join / calibration lines are byte-identical code in 0.7.4.dev1: the real rc1 mixed
    log gives the same shared verdict under the new engine id."""
    log = (FIX / "omlx_rc1_mixed_1106d.server.txt").read_text(errors="replace")
    reqs = [{"rid": i, "pt": p, "ct": 64} for i, p in enumerate([1024, 1024, 1024, 8204, 8204, 1024, 1024, 1024])]
    conc = [{"part": "mixed", "n": 8, "requests": reqs}]
    a = fe.concurrent_check(fe.load_table(), "omlx-0.7.3rc1", "dflash", log, conc, 0)
    b = fe.concurrent_check(fe.load_table(), E074, "dflash", log, conc, 0)
    assert (b[0]["engaged_mode"], b[0].get("label"), b[1]) == (a[0]["engaged_mode"], a[0].get("label"), a[1])
    assert fe.WARM_EXTRA[E074] == fe.WARM_EXTRA["omlx-0.7.3"]


def test_the_wiring_smoke_manifest_has_core_conc_and_mixed_and_nothing_else_changes():
    rows = fm.wire_omlx074_rows()
    shapes = {(r["batch"], r["sweep"], r["conc"], r["ctx"]) for r in rows}
    assert shapes == {("B1", "core", 1, "1K"), ("B2", "conc", 2, "1K"), ("B2", "mixed", 4, "8K")}
    assert all(r["engine"] == E074 and r["status"] == "unverified" for r in rows)
    for r in rows:
        part, args, _ = fr.harness_part(r)
        assert part in ("decode", "conc", "mixed")
        if part != "decode":
            assert args[args.index("--parallel") + 1] == str(r["conc"])
    # no other smoke or the real plan total includes the extra engine's hours
    assert all(r["engine"] != E074 for r in fm.wire_1108_rows())
    rows_all, _ = fm.build_rows()
    fm.apply_cuts(rows_all, out=io.StringIO())
    assert fm.planned_h([r for r in rows_all if r["engine"] != E074]) == fm.planned_h(rows_all)


def test_pins_resolve_for_the_new_engine_from_the_census():
    if not Path(fp.CENSUS_PATH).is_file():
        pytest.skip("census not on this machine")
    census = fp.load_json(fp.CENSUS_PATH)
    if "latest_dev_date" not in census["omlx"]:
        pytest.skip("census without the 0.7.4.dev1 entry")
    pin = json.loads(fr.PIN_PATH.read_text())
    pins = fam.build_pins(census, pin)
    p = pins[E074]
    assert p["version"] == "0.7.4.dev1" and str(p.get("release_date", "")).startswith("2026-11-04")
