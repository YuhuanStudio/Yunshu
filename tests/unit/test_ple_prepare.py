"""`yunshu prepare` and the auto-use of a prepared PLE repack: disk check, skip, fingerprint, storage choice."""

import json
import logging
import os
import struct
import subprocess
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from yunshu_engine import ple_prepare, ple_ssd

PLE = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.0.weight"
TABLE = 5000


@pytest.fixture
def pack(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    header = {PLE: {"dtype": "U8", "shape": [TABLE], "data_offsets": [0, TABLE]}}
    raw = json.dumps(header).encode()
    (model / "m.safetensors").write_bytes(
        struct.pack("<Q", len(raw)) + raw + b"\0" * TABLE
    )
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {PLE: "m.safetensors"}})
    )
    (model / "config.json").write_text(
        json.dumps({"model_type": "qwen4_exp", "text_config": {"hidden_size": 8}})
    )
    rows = tmp_path / "rows"
    monkeypatch.setenv("YUNSHU_PLE_ROWS_DIR", str(rows))
    monkeypatch.delenv("YUNSHU_PLE_STORAGE", raising=False)
    return model


def make_prepared(model, *, size=None):
    directory = ple_ssd.rows_dir_for(model)
    directory.mkdir(parents=True)
    manifest = {
        "layout": "interleaved_q4",
        "data_file": "ple-q4.rows",
        "row_count": 30,
        "row_stride": 100,
        "source_root": os.path.relpath(model, directory),
    }
    (directory / "ple-rows.json").write_text(json.dumps(manifest))
    (directory / "ple-q4.rows").write_bytes(b"\0" * (3000 if size is None else size))
    return directory


def test_unprepared_is_not_used_and_auto_means_ssd(pack):
    assert ple_ssd.prepared_manifest(pack) is None
    assert ple_ssd.effective_storage(pack) == ("ssd", "auto: not prepared")


def test_prepared_is_found_and_auto_means_ssd_rows(pack):
    make_prepared(pack)
    assert ple_ssd.prepared_manifest(pack) is not None
    assert ple_ssd.effective_storage(pack) == ("ssd-rows", "auto: prepared")
    assert ple_prepare.status(pack)["prepared"] is True


def test_explicit_ssd_overrides_a_prepared_copy(pack, monkeypatch):
    make_prepared(pack)
    monkeypatch.setenv("YUNSHU_PLE_STORAGE", "ssd")
    assert ple_ssd.effective_storage(pack) == ("ssd", "YUNSHU_PLE_STORAGE")


def test_external_ple_points_at_the_prepared_rows(pack, monkeypatch, caplog):
    directory = make_prepared(pack)
    monkeypatch.setattr(
        ple_ssd, "manifest_for", lambda p: pytest.fail("ranges manifest not wanted")
    )
    config = {"model_type": "qwen4_exp", "text_config": {}}
    with caplog.at_level(logging.INFO):
        assert ple_ssd.external_ple(config, pack) is True
    assert config["text_config"]["ple_storage"]["manifest"] == str(
        directory / "ple-rows.json"
    )
    assert "ssd-rows (auto: prepared)" in caplog.text
    assert "yunshu prepare" not in caplog.text


def test_unprepared_load_logs_the_hint_once(pack, monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(ple_ssd, "manifest_for", lambda p: tmp_path / "ple-store.json")
    config = {"model_type": "qwen4_exp", "text_config": {}}
    with caplog.at_level(logging.INFO):
        ple_ssd.external_ple(config, pack)
    assert "storage ssd (auto: not prepared)" in caplog.text
    assert caplog.text.count("yunshu prepare model") == 1


@pytest.mark.parametrize("damage", ["index", "size", "source"])
def test_a_mismatching_repack_is_not_used(pack, damage):
    directory = make_prepared(pack, size=2999 if damage == "size" else None)
    if damage == "source":
        path = directory / "ple-rows.json"
        manifest = json.loads(path.read_text())
        manifest["source_root"] = "../elsewhere"
        path.write_text(json.dumps(manifest))
    if (
        damage == "index"
    ):  # the checkpoint changed: a new fingerprint, a different directory
        index = pack / "model.safetensors.index.json"
        index.write_text(index.read_text() + " ")
    assert ple_ssd.prepared_manifest(pack) is None
    assert ple_ssd.effective_storage(pack)[0] == "ssd"


def test_disk_space_is_checked_before_anything_is_written(pack, monkeypatch):
    free = ple_prepare.needed_bytes(pack) - 1
    monkeypatch.setattr(ple_prepare, "_free_bytes", lambda d: free)
    with pytest.raises(ple_prepare.InsufficientDiskError) as err:
        ple_prepare.start_job(pack)
    assert err.value.free == free and "GiB" in str(err.value)
    assert not ple_ssd.rows_dir_for(pack).exists()
    assert ple_prepare.needed_bytes(pack) >= TABLE + ple_prepare.SAFETY_MIN_BYTES


def cli(*args):
    from yunshu_cli import app

    return CliRunner().invoke(app, ["--json", "prepare", *args])


def test_cli_skips_a_finished_pack(pack, monkeypatch):
    make_prepared(pack)
    monkeypatch.setattr(
        ple_prepare, "start_job", lambda p: pytest.fail("must not repack")
    )
    out = cli(str(pack))
    assert out.exit_code == 0, out.output
    assert json.loads(out.stdout)["skipped"] is True


def test_cli_refuses_when_the_disk_is_short(pack, monkeypatch):
    monkeypatch.setattr(ple_prepare, "_free_bytes", lambda d: 1)
    out = cli(str(pack))
    assert out.exit_code == 1
    assert "not enough free disk space" in out.output


def test_cli_remove_undoes_it(pack):
    directory = make_prepared(pack)
    out = cli(str(pack), "--remove")
    assert out.exit_code == 0 and json.loads(out.stdout)["removed"] is True
    assert not directory.exists()
    assert ple_ssd.effective_storage(pack)[0] == "ssd"


def test_cancel_kills_the_child_and_clears_its_temp_file(pack, monkeypatch):
    real = subprocess.Popen
    monkeypatch.setattr(
        ple_prepare.subprocess,
        "Popen",
        lambda cmd, **kw: real(
            [sys.executable, "-c", "import time; time.sleep(60)"], **kw
        ),
    )
    job = ple_prepare.start_job(pack)
    partial = ple_ssd.rows_dir_for(pack) / ".ple-q4.rows.x.partial"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"\0" * 10)
    assert job.poll()["state"] == "running"
    job.cancel()
    assert job.poll()["state"] == "cancelled" and not partial.exists()


def test_admin_endpoint_status_start_and_remove(pack, monkeypatch):
    from yunshu_gateway.routers import admin_prepare

    monkeypatch.setattr(admin_prepare, "_resolve", lambda model_id: pack)
    monkeypatch.setattr(admin_prepare, "_check_permission", lambda r, p: None)
    monkeypatch.setattr(admin_prepare, "_loaded_paths", lambda: {})
    app = FastAPI()
    app.include_router(admin_prepare.router, prefix="/v1")
    client = TestClient(app)
    got = client.get("/v1/yunshu/models/m/prepare").json()
    assert got["applicable"] and not got["prepared"] and got["storage"] == "ssd"
    monkeypatch.setattr(ple_prepare, "_free_bytes", lambda d: 1)
    short = client.post("/v1/yunshu/models/m/prepare")
    assert short.status_code == 507 and short.json()["detail"]["free_bytes"] == 1
    make_prepared(pack)
    again = client.post("/v1/yunshu/models/m/prepare")
    assert again.status_code == 202 and again.json()["prepared"] is True
    gone = client.delete("/v1/yunshu/models/m/prepare").json()
    assert gone["prepared"] is False and gone["storage"] == "ssd"
