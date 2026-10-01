"""F17: SSD cache integrity check and GC (truncation, old format, temp orphans, size cap)."""

from __future__ import annotations

import json
import os
import struct

from typer.testing import CliRunner

from yunshu_cli import cache as cache_cli
from yunshu_kv import cache_gc


def _st(path, payload=b"x" * 64, meta=None, cut=0, extra=b""):
    """Write a valid safetensors file; ``cut`` trims bytes off the end, ``extra`` appends."""
    header = {
        "t": {"dtype": "U8", "shape": [len(payload)], "data_offsets": [0, len(payload)]}
    }
    if meta is not None:
        header["__metadata__"] = meta
    raw = json.dumps(header).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    blob = struct.pack("<Q", len(raw)) + raw + payload + extra
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob[: len(blob) - cut] if cut else blob)
    return path


def _age(path, seconds):
    t = path.stat().st_mtime - seconds
    os.utime(path, (t, t))


def test_intact_files_are_left_alone(tmp_path):
    _st(tmp_path / "ns" / ("shard_" + "a" * 32 + ".safetensors"))
    _st(
        tmp_path / "b" / ("b" + "0" * 63 + ".safetensors"),
        meta={"yunshu_cache_version": "1"},
    )
    rep = cache_gc.scan(tmp_path, apply=True)
    assert rep.files == 2 and rep.findings == []


def test_truncated_file_with_valid_header_is_found(tmp_path):
    p = _st(tmp_path / "ns" / ("shard_" + "a" * 32 + ".safetensors"), cut=10)
    rep = cache_gc.scan(tmp_path)
    assert [(f.reason, f.path) for f in rep.findings] == [("truncated", str(p))]
    assert p.exists()  # dry run
    cache_gc.scan(tmp_path, apply=True)
    assert not p.exists()


def test_trailing_garbage_and_bad_header_are_corrupt(tmp_path):
    a = _st(tmp_path / "ns" / ("exact_" + "1" * 32 + ".safetensors"), extra=b"zz")
    b = tmp_path / "ns" / ("shard_" + "2" * 32 + ".safetensors")
    b.write_bytes(struct.pack("<Q", 10**12) + b"junk")
    c = tmp_path / "ns" / ("shard_" + "3" * 32 + ".safetensors")
    c.write_bytes(b"")
    reasons = {f.path: f.reason for f in cache_gc.scan(tmp_path).findings}
    assert reasons[str(a)] == "corrupt"
    assert reasons[str(b)] in ("corrupt", "truncated")
    assert reasons[str(c)] == "truncated"


def test_old_format_text_entry_is_invalidated(tmp_path):
    old = _st(
        tmp_path / "c" / ("c" + "1" * 63 + ".safetensors"),
        meta={"yunshu_cache_version": "0"},
    )
    unversioned = _st(tmp_path / "d" / ("d" + "1" * 63 + ".safetensors"))
    reasons = {f.path: f.reason for f in cache_gc.scan(tmp_path).findings}
    assert reasons == {str(old): "old-format", str(unversioned): "old-format"}


def test_tmp_orphans_collected_but_in_flight_writes_kept(tmp_path):
    stale = _st(tmp_path / "b" / "abc.safetensors.deadbeef.tmp")
    fresh = _st(tmp_path / "b" / "abd.safetensors.cafebabe.tmp")
    _age(stale, 3600)
    rep = cache_gc.scan(tmp_path, apply=True)
    assert [f.reason for f in rep.findings] == ["tmp-orphan"]
    assert not stale.exists() and fresh.exists()


def test_size_cap_drops_oldest_valid_files_first(tmp_path):
    files = []
    for i in range(4):
        p = _st(tmp_path / "ns" / (f"shard_{i:032x}.safetensors"), payload=b"x" * 1000)
        _age(p, (4 - i) * 100)  # shard 0 is the oldest
        files.append(p)
    size = files[0].stat().st_size
    rep = cache_gc.scan(tmp_path, max_bytes=2 * size + 5, apply=True)
    assert [f.reason for f in rep.findings] == ["over-cap", "over-cap"]
    assert [p.exists() for p in files] == [False, False, True, True]
    assert rep.kept_bytes <= 2 * size + 5


def test_unrecognized_files_are_never_removed(tmp_path):
    other = _st(tmp_path / "model.safetensors", cut=5)
    note = tmp_path / "README.txt"
    note.write_text("mine")
    rep = cache_gc.scan(tmp_path, max_bytes=1, apply=True)
    assert rep.unrecognized == 1 and rep.findings == []
    assert other.exists() and note.exists()


def test_cli_gc_dry_run_then_apply(tmp_path, monkeypatch):
    bad = _st(tmp_path / "ns" / ("shard_" + "a" * 32 + ".safetensors"), cut=3)
    monkeypatch.setattr(cache_cli, "cache_targets", lambda: [("apc", tmp_path, None)])
    runner = CliRunner()
    r = runner.invoke(cache_cli.cache_app, ["gc"])
    assert r.exit_code == 0 and bad.exists() and "--apply" in r.output
    r = runner.invoke(cache_cli.cache_app, ["gc", "--apply"])
    assert r.exit_code == 0 and not bad.exists()
