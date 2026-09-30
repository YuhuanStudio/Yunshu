"""The APC SSD tier is on by default: where it lives, how to switch it off, what
namespaces its states, and the doctor line."""

from __future__ import annotations

import importlib
from pathlib import Path

from yunshu_engine import paths

doctor = importlib.import_module("yunshu_cli.doctor")


def test_default_dir_is_under_the_yunshu_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("YUNSHU_VLM_APC_DISK", raising=False)
    monkeypatch.delenv("YUNSHU_VLM_APC_DISK_DIR", raising=False)
    assert paths.apc_dir() == tmp_path / ".yunshu" / "cache" / "apc"


def test_dir_override_and_opt_out(monkeypatch, tmp_path):
    monkeypatch.setenv("YUNSHU_VLM_APC_DISK_DIR", str(tmp_path / "x"))
    assert paths.apc_dir() == tmp_path / "x"
    monkeypatch.setenv("YUNSHU_VLM_APC_DISK", "0")
    assert paths.apc_dir() is None


def test_namespace_follows_the_weights(tmp_path):
    from yunshu_engine.vlm_engine import VLMEngine

    (tmp_path / "config.json").write_text("{}")
    w = tmp_path / "model.safetensors"
    w.write_bytes(b"a" * 10)
    eng = VLMEngine.__new__(VLMEngine)
    eng._model_path = str(tmp_path)
    first = eng._apc_disk_namespace()
    assert first == eng._apc_disk_namespace()
    w.write_bytes(b"b" * 11)  # the checkpoint was replaced in place
    assert eng._apc_disk_namespace() != first


def test_doctor_reports_the_tier(monkeypatch, tmp_path):
    monkeypatch.setenv("YUNSHU_VLM_APC_DISK_DIR", str(tmp_path / "apc"))
    c = doctor.check_prefix_disk()
    assert c.status == "ok" and str(tmp_path / "apc") in c.detail
    monkeypatch.setenv("YUNSHU_VLM_APC_DISK", "0")
    c = doctor.check_prefix_disk()
    assert c.status == "ok" and "off" in c.detail
    assert isinstance(tmp_path, Path)
