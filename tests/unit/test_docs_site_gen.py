"""The docs site's generated pages (site/scripts/gen_content.py, site/scripts/dump_openapi.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "site" / "scripts" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mdx_safe_escapes_outside_code_only():
    gc = _load("gen_content")
    assert gc.mdx_safe("a {b} <c> `{keep}`") == "a &#123;b&#125; &lt;c&gt; `{keep}`"


def test_sanitize_keeps_fenced_blocks():
    gc = _load("gen_content")
    text = "x {y}\n```toml\na = {b = 1}\n```\n"
    out = gc.sanitize(text)
    assert "x &#123;y&#125;" in out and "a = {b = 1}" in out


def test_configuration_pages_cover_every_stable_setting(tmp_path, monkeypatch):
    gc = _load("gen_content")
    monkeypatch.setattr(gc, "OUT", tmp_path)
    gc.main()
    from yunshu_engine.settings import REGISTRY

    for suffix in ("", ".zh-TW", ".zh-CN"):
        page = (tmp_path / f"configuration{suffix}.mdx").read_text()
        assert page.startswith("---\ntitle: ")
        for name, s in REGISTRY.items():
            if s.stability == "stable":
                assert f"`{name}`" in page
