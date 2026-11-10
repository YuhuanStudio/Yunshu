import importlib.metadata
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from probe_legacy_readout import (  # noqa: E402
    SOURCE,
    build_header,
    fixture_inputs,
    geometry,
    reference_inputs,
)


def test_captured_readouts_use_80_whole_groups():
    for n in (5, 8):
        assert geometry(15, n, 5120) == (80, 64)


@pytest.mark.parametrize(
    "args", [(14, 8, 5120), (15, 7, 5120), (15, 8, 4096), (True, 8, 5120)]
)
def test_uncaptured_abis_fail_closed(args):
    with pytest.raises(ValueError):
        geometry(*args)


def test_installed_header_can_be_called_without_constant_address_space_arguments():
    root = Path(str(importlib.metadata.distribution("mlx").locate_file("mlx/include")))
    h = build_header(root)
    assert "METAL_FUNC void qmm_t_impl(" in h
    part = h[h.index("METAL_FUNC void qmm_t_impl(") :]
    signature = part[: part.index("{")]
    assert "const constant int&" not in signature
    assert "false, T, BM, BK, BN" in SOURCE
    assert "threadgroup T Xs" in SOURCE


def test_fixture_has_exact_representable_group_terms():
    import numpy as np

    x, w, s, b = fixture_inputs()
    assert x.shape == (15, 5120) and w.shape == (8, 640)
    assert np.all(x.reshape(15, 80, 64).sum(-1)[:, :63] == 1 + 2**-8)
    assert np.all(x[:, 63 * 64 :] == 0) and np.all(s == 1) and np.all(b == 0)


def reference_fixture(tmp_path):
    import json

    from probe_draft_readout import digest

    f = tmp_path / "frame.safetensors"
    f.write_bytes(b"fixture-checksum-only")
    frame = {
        "file": f.name,
        "sha256": digest(f),
        "shape": [15, 5120],
        "width": 5,
        "bits": 4,
        "group_size": 64,
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"complete": True, "frames": [frame]}))
    return {
        "complete": True,
        "mlx": "0.32.3",
        "manifest_sha256": digest(manifest),
        "frames": [
            {
                "file": f.name,
                "input_sha256": digest(f),
                "shape": [15, 5],
                "logits": [[0.0] * 5] * 15,
            }
        ],
    }


def test_reference_provenance_is_checked_before_gpu_work(tmp_path):
    r = reference_fixture(tmp_path)
    assert len(reference_inputs(tmp_path, r)[1]) == 1
    r["mlx"] = "0.32.4"
    with pytest.raises(ValueError, match="0.32.3"):
        reference_inputs(tmp_path, r)


def test_changed_inputs_or_missing_rows_fail_closed(tmp_path):
    r = reference_fixture(tmp_path)
    r["frames"][0]["input_sha256"] = "wrong"
    with pytest.raises(ValueError, match="inputs"):
        reference_inputs(tmp_path, r)
    r = reference_fixture(tmp_path)
    r["frames"][0]["logits"] = []
    with pytest.raises(ValueError, match="shape"):
        reference_inputs(tmp_path, r)
