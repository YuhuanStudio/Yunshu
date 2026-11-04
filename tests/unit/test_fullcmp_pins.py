"""Audit pins: every census engine has one; the pinned digest is the launched file's (not the
release archive's); engines without an established engaged mode fail closed ('unverified').

CPU only, fake census and fake digests. The real census lives in the run dir (outside the repo).
"""

from __future__ import annotations

import pytest

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "research"))

import fullcmp_audit_manifest as fam  # noqa: E402

H = "a" * 64  # archive digests in the fake census
D = "c" * 64  # digest of every launched file in the fake world

CENSUS_TO_MANIFEST = {
    "yunshu": ("yunshu",),
    "tensorfold": ("tf-1.2.1", "tf-1.1.8"),
    "splash": ("splash-1.5.0",),
    "omlx": ("omlx-0.7.3", "omlx-0.7.3rc1"),
    "mlx-serve": ("mlx-serve-26.11.1",),
    "mlx_lm_server": ("mlx-lm-server",),
    "mlx_vlm_server": ("mlx-vlm-server",),
    "llama_cpp": ("llama-cpp-b12061",),
    "kapitsa": ("kapitsa-0.2.26",),
    "lm_studio": ("lmstudio",),
    "ollama": ("ollama",),
}

PIN = {
    "sha": "0f2960047855546a9f13b2631b680352b1c57cfe",
    "version": "0.1.7.dev0",
    "python_executable": "/venv/bin/python",
    "worktree": "/fake/tree",
}


def _census():
    def one(date, sha_key, sha_val=H):
        return {"latest_date": date, "sha256": {sha_key: sha_val}}

    return {
        "_meta": {"models": {}},
        "yunshu": {"latest_date": "2026-11-03 (commit 0f2960047)"},
        "tensorfold": {
            "latest_date": "2026-11-01T23:52:33Z",
            "sha256": {"1.2.1 tarball": H, "1.1.8 tarball": H},
        },
        "splash": one("2026-10-28T20:15:32Z", "1.5.0 tarball"),
        "omlx": {
            "latest_date": "2026-10-31T16:42:55Z",
            "latest_rc_date": "2026-10-27T09:22:10Z",
            "sha256": {"0.7.3 wheel cp313": H, "0.7.3rc1 wheel cp313": H},
        },
        "mlx-serve": one("2026-11-01T19:34:34Z", "26.11.1 binary mlx-serve"),
        "mlx_lm_server": one(
            "2026-10-16T02:49:34Z", "mlx_lm-0.32.2-py3-none-any.whl (PyPI digest)"
        ),
        "mlx_vlm_server": one(
            "2026-10-28T17:48:51Z", "mlx_vlm-0.7.11-py3-none-any.whl (PyPI digest)"
        ),
        "llama_cpp": {
            "latest_date": "v0.8.0 2026-10-28T12:47:51Z; b12061 2026-11-02T23:17:24Z",
            "sha256": {"b12061 macos-arm64 tarball": H},
        },
        "kapitsa": {
            "latest_date": "2026-11-01T22:32:02Z",
            "sha256": {"0.2.26 tarball": H, "0.2.26 binary kapitsa": H},
        },
        "lm_studio": {},
        "ollama": {},
    }


def _head(worktree):
    return PIN["sha"]


def _sha_of(path):
    return D if path else None


def test_every_census_engine_has_a_pin():
    pins = fam.build_pins(_census(), PIN, sha_of=_sha_of, git_head=_head)
    for key, mids in CENSUS_TO_MANIFEST.items():
        for mid in mids:
            assert mid in pins, f"census {key} -> {mid} has no pin"


def test_pinned_binary_is_the_launched_file_not_the_archive():
    # regression (2026-11-03 smoke): the pin held the release archive digest, the harness header
    # holds the launched binary's digest, so every TensorFold / Splash / mlx-serve cell failed
    # pinned_versions with "binary sha256 missing or not the pinned binary"
    pins = fam.build_pins(_census(), PIN, sha_of=_sha_of, git_head=_head)
    for mid in ("tf-1.2.1", "tf-1.1.8", "splash-1.5.0", "mlx-serve-26.11.1"):
        assert pins[mid]["binary_sha256"] == D, mid
        assert pins[mid]["archive_sha256"] == H, mid
        assert pins[mid]["install_path"], mid


def test_install_paths_carry_the_pinned_version():
    pins = fam.build_pins(_census(), PIN, sha_of=_sha_of, git_head=_head)
    assert "tensorfold-1.2.1" in pins["tf-1.2.1"]["install_path"]
    assert "tensorfold-1.1.8" in pins["tf-1.1.8"]["install_path"]
    assert "<ver>" not in pins["tf-1.2.1"]["install_path"]


def test_unestablished_engaged_modes_fail_closed():
    # An engine with no engaged-mode pattern cannot be verified from its log (fail closed: no regex).
    # Engines with established evidence (mlx-vlm-server dflash, llama-cpp mtp, tf-* dflash) carry one.
    pins = fam.build_pins(_census(), PIN, sha_of=_sha_of, git_head=_head)
    assert pins["mlx-serve-26.11.1"]["engaged_regex"] is None
    for mid in ("mlx-vlm-server", "llama-cpp-b12061", "tf-1.2.1"):
        assert pins[mid]["engaged_regex"] is not None, mid


def test_unsupported_engines_have_no_digest_and_say_why():
    pins = fam.build_pins(_census(), PIN, sha_of=_sha_of, git_head=_head)
    for mid in ("lmstudio", "ollama"):
        assert pins[mid]["binary_sha256"] is None
        assert pins[mid]["unsupported_reason"]


def test_missing_binary_leaves_the_digest_empty():
    # the yunshu pin fails closed (raises) on a missing binary, so the other engines are checked
    # with a stub that only hides the tf digest
    def sha_of(p):
        return None if "tensorfold" in p else D  # only the tf digest is hidden (pb2 python must exist)

    pins = fam.build_pins(_census(), PIN, sha_of=sha_of, git_head=_head)
    assert pins["tf-1.2.1"]["binary_sha256"] is None


def test_yunshu_pin_fails_closed_on_a_wrong_tree_or_missing_binary():
    # Lead ruling 2026-11-03: a wrong tree or an absent launched binary must not produce a pin.
    with pytest.raises(ValueError, match="pinned tree"):
        fam.build_pins(_census(), PIN, sha_of=_sha_of, git_head=lambda w: "f" * 40)
    with pytest.raises(ValueError, match="not on disk"):
        fam.build_pins(_census(), PIN, sha_of=lambda p: None, git_head=_head)


def test_yunshu_pin_is_dflash_with_the_engaged_line_and_uv_lock():
    pins = fam.build_pins(_census(), PIN, sha_of=_sha_of, git_head=_head)
    y = pins["yunshu"]
    assert y["binary_sha256"] == D
    assert y["expected_mode"] == "dflash"
    assert y["engaged_regex"] == "draft=dflash"
    assert y["git_sha"] == PIN["sha"]


def test_file_sha256_hashes_the_real_bytes(tmp_path):
    f = tmp_path / "bin"
    f.write_bytes(b"abc")
    assert fam.file_sha256(str(f)) == hashlib.sha256(b"abc").hexdigest()
    assert fam.file_sha256(str(tmp_path / "missing")) is None
