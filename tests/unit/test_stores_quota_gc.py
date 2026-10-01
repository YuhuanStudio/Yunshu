"""R24: file / conversation store quota, orphan GC and failed-write cleanup."""

from __future__ import annotations

import os
import time

import pytest

from yunshu_gateway import conversations_store as cs
from yunshu_gateway import files_store as fs


def _store(tmp_path, total=0):
    s = fs.FileStore(tmp_path / "files", max_bytes=10_000)
    s.max_total_bytes = total
    return s


def test_failed_meta_write_leaves_no_orphan_blob(tmp_path, monkeypatch):
    s = _store(tmp_path)
    real = fs._atomic_write

    def flaky(path, data):
        if path.parent.name == "meta":
            raise OSError("disk full")
        real(path, data)

    monkeypatch.setattr(fs, "_atomic_write", flaky)
    with pytest.raises(OSError):
        s.put(b"hello", "a.txt")
    assert list((s.root / "blobs").iterdir()) == []


def test_gc_removes_crash_orphans_but_keeps_fresh(tmp_path):
    s = _store(tmp_path)
    live = s.put(b"live", "live.txt")
    old = time.time() - 3600
    stale_tmp = s.root / "blobs" / ".x.abcd.tmp"
    stale_tmp.write_bytes(b"junk")
    orphan_blob = s.root / "blobs" / "file_orphan"
    orphan_blob.write_bytes(b"orphan")
    orphan_meta = s.root / "meta" / "file_nometa.json"
    orphan_meta.write_text('{"id": "file_nometa", "created_at": 1}')
    fresh_blob = s.root / "blobs" / "file_inflight"
    fresh_blob.write_bytes(b"in flight")  # a put between blob and meta writes
    for p in (stale_tmp, orphan_blob, orphan_meta):
        os.utime(p, (old, old))
    removed = s.gc()
    assert removed >= 3
    assert not stale_tmp.exists()
    assert not orphan_blob.exists()
    assert not orphan_meta.exists()
    assert fresh_blob.exists()
    assert s.read(live["id"]) == b"live"


def test_gc_reaps_expired_files(tmp_path):
    s = _store(tmp_path)
    meta = s.put(b"x", "a.txt", expires_after=1)
    m = s._meta(meta["id"])
    d = m.read_text().replace(str(meta["expires_at"]), str(int(time.time()) - 5))
    m.write_text(d)
    s.gc()
    assert not m.exists()
    assert not s._blob(meta["id"]).exists()


def test_total_quota_enforced_and_expired_freed_first(tmp_path):
    s = _store(tmp_path, total=100)
    a = s.put(b"a" * 60, "a.bin")
    with pytest.raises(fs.FileStoreError) as ei:
        s.put(b"b" * 60, "b.bin")
    assert ei.value.status == 413
    assert ei.value.code == "storage_quota_exceeded"
    assert s.read(a["id"]) == b"a" * 60  # existing data untouched
    s.delete(a["id"])
    s.put(b"b" * 60, "b.bin")  # space freed


def test_conversation_failed_save_leaves_no_tmp(tmp_path, monkeypatch):
    s = cs.ConversationStore(tmp_path / "c")
    rec = s.create()

    def boom(_fd):
        raise OSError("io")

    monkeypatch.setattr(cs.os, "fsync", boom)
    with pytest.raises(OSError):
        s.update_metadata(rec["id"], {"a": "b"})
    assert [p.name for p in (tmp_path / "c").iterdir() if p.name.endswith(".tmp")] == []
