"""Fail-closed CPU admission for the diagnostic draft readout replay."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
import probe_draft_readout as probe  # noqa: E402


def test_preserves_pinned_interpreter_and_leaves_other_commands_alone(tmp_path):
    cmd = ["/external/venv/bin/python", "-m", "yunshu_cli", "serve", "--port", "18996"]
    result = probe.server_command(cmd, tmp_path)
    assert result[0] == cmd[0]
    assert result[2:4] == ["serve", str(tmp_path)]
    assert result[4:] == cmd[3:]
    assert probe.server_command(["ps", "-ax"], tmp_path) == ["ps", "-ax"]


def test_manifest_requires_complete_and_valid_content(tmp_path):
    f = tmp_path / "x.safetensors"
    f.write_bytes(b"fixture")
    data = {"complete": True, "frames": [{"file": f.name, "sha256": probe.digest(f)}]}
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps(data))
    assert probe.read_manifest(tmp_path) == data
    f.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        probe.read_manifest(tmp_path)
    data["complete"] = False
    m.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="incomplete"):
        probe.read_manifest(tmp_path)


def receipt():
    return {
        "complete": True,
        "manifest_sha256": "same",
        "device": {"name": "M5"},
        "mlx": "old",
        "frames": [
            {
                "file": "x",
                "input_sha256": "same",
                "shape": [1, 2],
                "logits": [[1.0, 2.0]],
                "argmax": [1],
            }
        ],
    }


def test_comparison_reports_drift_and_argmax():
    a = receipt()
    b = copy.deepcopy(a)
    b["mlx"] = "new"
    b["frames"][0]["logits"] = [[3.0, 2.0]]
    b["frames"][0]["argmax"] = [0]
    result = probe.compare_replays(a, b)
    assert result["complete"] and result["diagnostic_only"]
    assert result["frames"] == [
        {"file": "x", "max_abs_delta": 2.0, "changed_values": 1, "argmax_changed": True}
    ]


@pytest.mark.parametrize(
    "field,value",
    [
        ("complete", False),
        ("manifest_sha256", "other"),
        ("device", "other"),
        ("frames", []),
    ],
)
def test_comparison_rejects_missing_or_mixed_evidence(field, value):
    a, b = receipt(), receipt()
    b[field] = value
    with pytest.raises(ValueError):
        probe.compare_replays(a, b)


def test_nonfinite_logits_fail_closed():
    a, b = receipt(), receipt()
    b["frames"][0]["logits"][0][0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        probe.compare_replays(a, b)


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_result_keeps_a_complete_record_after_gpuq_stamps_it(
    tmp_path, suffix, monkeypatch
):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/dev"))
    from device_evidence import stamp_output
    from gpuq_digest import output_problem

    out = tmp_path / ("receipt" + suffix)
    from types import SimpleNamespace

    # Exercise the real replay publication path without loading MLX or a checkpoint.
    mx_stub = SimpleNamespace(
        metal=SimpleNamespace(device_info=lambda: {"device_name": "M5"})
    )
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx_stub))
    monkeypatch.setitem(sys.modules, "mlx.core", mx_stub)
    monkeypatch.setattr(probe, "read_manifest", lambda _: {"frames": []})
    monkeypatch.setattr(probe.importlib.metadata, "version", lambda _: "fixture")
    (tmp_path / "manifest.json").write_text("{}")
    probe.replay(SimpleNamespace(directory=tmp_path, out=out))
    stamp_output(out, "m5")
    assert output_problem(out, expect_complete=True) is None
    assert json.loads(out.read_text())["mlx"] == "fixture"
