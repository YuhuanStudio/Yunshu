"""An available checkpoint does not prove an explicitly requested Hub revision."""

import pytest

from yunshu_cli import model as cli_model


@pytest.mark.parametrize("revision", ["release-tag", "a" * 40])
@pytest.mark.parametrize("local", [False, True])
def test_explicit_revision_reaches_hub_even_when_model_is_present(
    monkeypatch, tmp_path, revision, local
):
    import huggingface_hub

    base = tmp_path / "models"
    target = base / "org" / "model"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}")
    (target / "model.safetensors").write_bytes(b"existing weights")
    monkeypatch.setattr(
        cli_model, "scan_hf_cache", lambda: [{"name": "org/model", "path": str(target)}]
    )
    monkeypatch.setattr(cli_model, "is_json", lambda: True)
    monkeypatch.setattr(cli_model, "_detect_model_type", lambda _: "LLM")
    emitted, calls = [], []
    monkeypatch.setattr(cli_model, "emit", lambda data, **kwargs: emitted.append(data))
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kwargs: calls.append(kwargs)
    )
    # Cache-only and downloaded-local states both used to bypass revision resolution.
    if not local:
        target.rename(base / "cached")
        cached = base / "cached"
        monkeypatch.setattr(
            cli_model,
            "scan_hf_cache",
            lambda: [{"name": "org/model", "path": str(cached)}],
        )

        def download(**kwargs):
            calls.append(kwargs)
            target.mkdir(parents=True)
            (target / "config.json").write_text("{}")
            (target / "model.safetensors").write_bytes(b"requested weights")

        monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    cli_model.pull("org/model", str(base), revision, False)
    assert calls == [
        {"repo_id": "org/model", "local_dir": str(target), "revision": revision}
    ]
    assert emitted[0]["revision"] == revision


def test_default_pull_reuses_complete_local_model(monkeypatch, tmp_path):
    import huggingface_hub

    target = tmp_path / "org" / "model"
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}")
    (target / "model.safetensors").write_bytes(b"existing weights")
    emitted = []
    monkeypatch.setattr(cli_model, "emit", lambda data, **kwargs: emitted.append(data))

    def unexpected_download(**kwargs):
        raise AssertionError(
            "an ordinary pull should reuse the complete local checkpoint"
        )

    monkeypatch.setattr(huggingface_hub, "snapshot_download", unexpected_download)
    cli_model.pull("org/model", str(tmp_path), None, False)
    assert emitted[0]["status"] == "present"
