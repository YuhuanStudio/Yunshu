import json

from yunshu_engine import spec_select

TARGET = {
    "model_type": "qwen3_5",
    "text_config": {"num_hidden_layers": 64, "hidden_size": 5120},
}


def _drafter(path, layers=64, hidden=5120):
    path.mkdir(parents=True)
    cfg = {"dflash_config": {}, "num_target_layers": layers, "hidden_size": hidden}
    (path / "config.json").write_text(json.dumps(cfg))
    (path / "model.safetensors").write_bytes(b"x")
    return path


def _choose(tmp_path, monkeypatch, mtp=True):
    monkeypatch.setattr("yunshu_engine.model_discovery.hf_cache_snapshots", lambda: [])
    return spec_select.choose(
        TARGET, spec_family=True, mtp_capable=mtp, models_dir=tmp_path
    )


def test_auto_picks_matching_drafter(tmp_path, monkeypatch):
    d = _drafter(tmp_path / "incoai" / "Qwen3.8-27B-DFlash2")
    c = _choose(tmp_path, monkeypatch)
    assert (c.kind, c.drafter, c.automatic) == ("dflash", str(d), True)


def test_mismatched_drafter_ignored(tmp_path, monkeypatch):
    _drafter(tmp_path / "Other-DFlash", hidden=4096)
    assert _choose(tmp_path, monkeypatch).kind == "mtp"


def test_force_mtp_off_and_path(tmp_path, monkeypatch):
    _drafter(tmp_path / "x-DFlash2")
    monkeypatch.setenv("YUNSHU_VLM_DRAFT", "mtp")
    assert _choose(tmp_path, monkeypatch).kind == "mtp"
    monkeypatch.setenv("YUNSHU_VLM_DRAFT", "off")
    assert _choose(tmp_path, monkeypatch).kind == "none"
    monkeypatch.setenv("YUNSHU_VLM_DRAFT", "/some/dir")
    c = _choose(tmp_path, monkeypatch, mtp=False)
    assert (c.kind, c.drafter, c.automatic) == ("dflash", "/some/dir", False)


def test_non_spec_family(tmp_path):
    c = spec_select.choose(
        {}, spec_family=False, mtp_capable=False, models_dir=tmp_path
    )
    assert c.kind == "none"
