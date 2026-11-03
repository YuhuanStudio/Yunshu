"""Wiring-smoke fixes (2026-11-03): tf-1.1.8 release date in the audit pin, expected-mode probes for
tf-1.1.8 / splash-1.5.0 / mlx-lm-server, and the engaged-mode parsers on the lines those engines print.
CPU only: no server, no GPU, no model."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import bench_engines as be  # noqa: E402
import fullcmp_audit_manifest as fam  # noqa: E402


def test_tf_1_1_8_pin_carries_its_github_release_date():
    pin = fam._pin("tf-1.1.8", "1.1.8", "2026-10-28", "ab", "release archive", lambda p: None)
    assert pin["release_date"] == "2026-10-28"


def test_probe_expected_modes_match_the_engine_registry():
    for engine in ("tf-1.1.8", "splash-1.5.0", "mlx-lm-server"):
        mode, _ = fam._expected(engine)
        assert mode == be.ENGINES[engine].expected_mode, engine
    assert fam._expected("mlx-lm-server") == ("ar", r"Starting httpd at 127\.0\.0\.1")


def test_tf_1_1_8_drafter_line_engages_dflash():
    log = "[tensorfold] serving m at http://127.0.0.1:1/v1 (sampling: greedy; drafts: DFlash2 (/d); context: 1024)\n"
    assert be.detect_mode("tf-1.1.8", log) == "dflash"


def test_tf_1_1_8_without_drafter_line_fails_closed():
    log = "[tensorfold] serving m at http://127.0.0.1:1/v1 (sampling: greedy; drafts: ; context: 1024)\n"
    assert be.detect_mode("tf-1.1.8", log) is None


def test_mlx_lm_stock_server_engages_ar_once_httpd_starts():
    log = "Starting httpd at 127.0.0.1 on port 18999 ...\n"
    assert be.detect_mode("mlx-lm-server", log) == "ar"
    assert be.detect_mode("mlx-lm-server", "loading weights\n") is None


def test_smoke_rows_log_the_usage_sample_and_real_rows_do_not():
    import fullcmp_run as fr

    smoke = {"batch": "SMOKE", "table": "decode", "engine": "tf-1.1.8", "conc": 1,
             "sampling": "greedy", "turns": "single", "phases": ["cold", "warm"],
             "prompt_kind": "code", "ctx": "1K", "ctx_tokens": 1024, "out_len": 64, "id": "x"}
    _, extra, _ = fr.harness_part(smoke)
    assert extra[-1] == "--log-usage-sample"
    real = dict(smoke, batch="B1")
    _, extra_real, _ = fr.harness_part(real)
    assert "--log-usage-sample" not in (extra_real or [])
