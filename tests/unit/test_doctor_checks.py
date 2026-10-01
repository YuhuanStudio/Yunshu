"""F13: doctor dependency / capability / download / disk / cache checks, each with a fix."""

from __future__ import annotations

import importlib
import json

from yunshu_cli import cache as cache_cli

doctor = importlib.import_module("yunshu_cli.doctor")


def _by_name(checks):
    return {c.name: c for c in checks}


def test_version_guard_fails_below_minimum_with_upgrade_command():
    fake = {
        "mlx": "0.32.3",
        "mlx-lm": "0.30.0",
        "mlx-vlm": "0.7.4",
        "llguidance": "1.7.9",
    }
    checks = _by_name(doctor.check_versions(pkg=fake.get))
    assert checks["version mlx-lm"].status == "fail"
    assert "uv pip install" in checks["version mlx-lm"].fix
    assert checks["version llguidance"].status == "fail"
    assert "version mlx" not in checks


def test_version_guard_ok_and_missing_packages_are_not_version_failures():
    fake = {"mlx": "0.40.0", "mlx-lm": "0.31.3", "llguidance": "2.0"}
    (c,) = doctor.check_versions(pkg=fake.get)
    assert c.status == "ok"


def test_extras_missing_audio_and_llguidance_have_fixes():
    checks = _by_name(doctor.check_extras(pkg=lambda n: None))
    assert checks["extra audio"].status == "warn"
    assert "yunshu[audio]" in checks["extra audio"].fix
    assert checks["llguidance"].status == "fail" and checks["llguidance"].fix


def test_api_feature_state_flags_open_network_binding(monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)
    c = _by_name(doctor.check_api_features("0.0.0.0"))["auth"]
    assert c.status == "warn" and "YUNSHU_AUTH_TOKEN" in c.fix
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "t")
    assert _by_name(doctor.check_api_features("0.0.0.0"))["auth"].status == "ok"


def test_half_downloaded_models_are_reported_with_pull_fix(tmp_path):
    half = tmp_path / "org" / "half"
    (half / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (
        half / ".cache" / "huggingface" / "download" / "w.safetensors.incomplete"
    ).write_text("x")
    (half / "config.json").write_text("{}")
    missing = tmp_path / "gone"
    missing.mkdir()
    (missing / "config.json").write_text("{}")
    (missing / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-1.safetensors"}})
    )
    ok = tmp_path / "ok"
    ok.mkdir()
    (ok / "config.json").write_text("{}")
    (ok / "model.safetensors").write_bytes(b"x")
    checks = doctor.check_downloads(tmp_path)
    detail = " | ".join(c.detail for c in checks)
    assert "half" in detail and "gone" in detail and "ok:" not in detail
    assert all(c.status == "warn" and "yunshu pull" in c.fix for c in checks)


def test_disk_budget_warns_when_cache_cap_exceeds_free_space(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cache_cli, "cache_targets", lambda: [("apc", tmp_path / "apc", 64 << 30)]
    )
    (c,) = doctor.check_disk_budget(tmp_path, free=lambda p: 10 << 30)
    assert c.status == "warn" and "YUNSHU_VLM_APC_DISK_GB" in c.fix
    (c,) = doctor.check_disk_budget(tmp_path, free=lambda p: 500 << 30)
    assert c.status == "ok"


def test_cache_integrity_check_points_at_gc(tmp_path, monkeypatch):
    bad = tmp_path / "ns" / ("shard_" + "a" * 32 + ".safetensors")
    bad.parent.mkdir()
    bad.write_bytes(b"\x01")
    monkeypatch.setattr(cache_cli, "cache_targets", lambda: [("apc", tmp_path, None)])
    (c,) = doctor.check_cache_integrity()
    assert c.status == "warn" and "yunshu cache gc --apply" in c.fix
    bad.unlink()
    (c,) = doctor.check_cache_integrity()
    assert c.status == "ok"
