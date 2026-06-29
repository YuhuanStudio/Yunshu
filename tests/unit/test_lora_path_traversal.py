"""resolve_lora_file (reached from an UNTRUSTED image-generation prompt via the
inline <lora:NAME:WEIGHT> tag) had a bare `if os.path.isfile(name): return name` — so any
authenticated tenant could load an ARBITRARY .safetensors on the host (an absolute path
or a .. escape), e.g. another tenant's private adapter, simply by putting the full path in
the prompt. Now a direct path is honored only when it resolves INSIDE search_dir."""

from __future__ import annotations

import os
import tempfile

from yunshu_engine.diffusion_features import resolve_lora_file


def _mkfile(d, name):
    p = os.path.join(d, name)
    open(p, "w").close()
    return p


def test_absolute_path_outside_search_dir_blocked():
    inside_dir = tempfile.mkdtemp()
    outside_dir = tempfile.mkdtemp()
    secret = _mkfile(outside_dir, "secret.safetensors")
    assert resolve_lora_file(secret, inside_dir) is None


def test_parent_dir_escape_blocked():
    inside_dir = tempfile.mkdtemp()
    outside_dir = tempfile.mkdtemp()
    _mkfile(outside_dir, "secret.safetensors")
    escape = os.path.join(
        inside_dir, "..", os.path.basename(outside_dir), "secret.safetensors"
    )
    assert resolve_lora_file(escape, inside_dir) is None


def test_path_inside_search_dir_allowed():
    d = tempfile.mkdtemp()
    p = _mkfile(d, "mylora.safetensors")
    assert resolve_lora_file(p, d) == p


def test_stem_lookup_still_works():
    d = tempfile.mkdtemp()
    _mkfile(d, "mylora.safetensors")
    assert resolve_lora_file("mylora", d) == os.path.join(d, "mylora.safetensors")


def test_nonexistent_returns_none():
    d = tempfile.mkdtemp()
    assert resolve_lora_file("does-not-exist", d) is None
