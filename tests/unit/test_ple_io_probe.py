"""ple_io_probe: row reads land on the manifest's offsets (CPU only, a tiny fake table)."""

import importlib.util
import json
import os
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[2] / "scripts/research/ple_io_probe.py"
_SPEC = importlib.util.spec_from_file_location("ple_io_probe", _PATH)
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


def _table(tmp_path):
    data = bytes(range(256)) * 64
    (tmp_path / "s.safetensors").write_bytes(data)
    t = lambda off, shape, dt: {  # noqa: E731
        "file": "s.safetensors",
        "offset": off,
        "dtype": dt,
        "shape": shape,
    }
    shard = dict(
        row_start=10,
        row_count=100,
        weight=t(0, [100, 2], "U32"),
        scales=t(1000, [100, 1], "BF16"),
        biases=t(2000, [100, 1], "BF16"),
    )
    manifest = tmp_path / "ple-store.json"
    manifest.write_text(json.dumps(dict(source_root=".", shards=[shard])))
    return manifest, data


def test_plan_reads_matches_row_offsets(tmp_path):
    manifest, data = _table(tmp_path)
    shards = probe.load_layout(manifest)
    reads = probe.plan_reads(shards, np.asarray([10, 15]))
    root = str((tmp_path / "s.safetensors").resolve())
    assert reads == [
        (root, 0, 8),
        (root, 1000, 2),
        (root, 2000, 2),
        (root, 40, 8),
        (root, 1010, 2),
        (root, 2010, 2),
    ]
    for threads in (1, 4):
        assert probe.run_reads(reads, threads, nocache=False) >= 0
    assert os.path.getsize(root) == len(data)


def test_row_outside_the_table_is_an_error(tmp_path):
    manifest, _ = _table(tmp_path)
    try:
        probe.plan_reads(probe.load_layout(manifest), np.asarray([5]))
    except IndexError:
        return
    raise AssertionError("row 5 is before the table")
