"""F15: service log rotation (size, age, retention) and secret redaction."""

from __future__ import annotations

import gzip
import os

from yunshu_engine import log_rotation
from yunshu_engine.log_rotation import Policy, redact, rotate


def _policy(**kw):
    base = {"max_bytes": 1000, "keep": 3, "retention_s": 0.0, "interval_s": 0.0}
    return Policy(**{**base, **kw})


def test_redact_covers_headers_keys_and_env_assignments():
    cases = [
        "Authorization: Bearer abcdef1234567890",
        "YUNSHU_AUTH_TOKEN=hunter2hunter2",
        "x-api-key: sk-abcdefghijklmnopqrstuvwx",
        "token hf_abcdefghijklmnopqrstuvwxyz0123",
        '{"api_key": "supersecretvalue"}',
        "GET /v1/x?api_key=abc123def456 HTTP/1.1",
    ]
    for line in cases:
        out = redact(line)
        assert "[REDACTED]" in out, line
        for secret in (
            "abcdef1234567890",
            "hunter2hunter2",
            "sk-abcdef",
            "hf_abcdef",
            "supersecretvalue",
            "abc123def456",
        ):
            assert secret not in out


def test_redact_keeps_ordinary_numbers():
    line = "usage max_tokens=512 completion_tokens: 42 status=200"
    assert redact(line) == line


def test_rotates_by_size_gzips_redacts_and_truncates(tmp_path):
    log = tmp_path / "yunshu.log"
    log.write_text("Authorization: Bearer topsecrettoken123\n" + "x" * 2000 + "\n")
    out = rotate(log, _policy(), now=1_800_000_000)
    assert out["rotated"] and log.stat().st_size == 0
    (arch,) = log_rotation.archives(log)
    with gzip.open(arch, "rt") as fh:
        text = fh.read()
    assert "topsecrettoken123" not in text and "[REDACTED]" in text


def test_small_fresh_log_is_not_rotated(tmp_path):
    log = tmp_path / "yunshu.log"
    log.write_text("hello\n")
    assert not rotate(log, _policy(interval_s=3600)).get("rotated")
    assert log.read_text() == "hello\n"


def test_time_rotation_after_interval(tmp_path):
    log = tmp_path / "yunshu.log"
    log.write_text("one line\n")
    old = log.stat().st_mtime - 7200
    os.utime(log, (old, old))
    # first call: no archive yet, the file's own age counts
    out = rotate(log, _policy(max_bytes=0, interval_s=3600))
    assert out["rotated"]


def test_keep_and_retention_prune_archives(tmp_path):
    log = tmp_path / "yunshu.log"
    for i in range(5):
        log.write_text("x" * 2000)
        rotate(log, _policy(keep=99), now=1_800_000_000 + i * 10)
    assert len(log_rotation.archives(log)) == 5
    out = rotate(log, _policy(keep=2), now=1_800_000_100)
    assert len(log_rotation.archives(log)) == 2 and len(out["pruned"]) == 3
    out = rotate(log, _policy(keep=99, retention_s=60), now=1_800_000_100 + 3600)
    assert log_rotation.archives(log) == []


def test_settings_are_registered():
    p = Policy.from_settings()
    assert p.max_bytes == 50 * 1024 * 1024 and p.keep == 7
