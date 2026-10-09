"""Every shipped feature is indexed, documented and visible in all three READMEs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INDEX = json.loads((ROOT / "docs/feature_index.json").read_text())
READMES = [ROOT / name for name in ("README.md", "README.zh-TW.md", "README.zh-CN.md")]
STATUSES = {"stable", "partial", "experimental"}


def test_index_shape():
    groups = INDEX["groups"]
    assert set(groups) == {"inference", "apis", "agents", "multimodal", "ops"}
    ids = [f["id"] for f in INDEX["features"]]
    assert len(ids) == len(set(ids))
    for f in INDEX["features"]:
        assert f["group"] in groups, f
        assert f["status"] in STATUSES, f
        assert f["title"] and f["summary"], f
    for group in groups:
        assert any(f["group"] == group for f in INDEX["features"]), group


@pytest.mark.parametrize("feature", INDEX["features"], ids=lambda f: f["id"])
def test_feature_doc_exists_and_is_linked_from_every_readme(feature):
    doc = ROOT / feature["doc"]
    assert doc.is_file(), f"{feature['id']}: missing {feature['doc']}"
    for readme in READMES:
        assert f"]({feature['doc']})" in readme.read_text(), (
            f"{readme.name} does not link {feature['doc']} for {feature['id']}"
        )


def test_every_feature_group_has_a_readme_table():
    for readme in READMES:
        text = readme.read_text()
        for feature in INDEX["features"]:
            assert "| [" in text and f"]({feature['doc']}) |" in text, readme.name


def test_markdown_images_exist():
    """README and docs images resolve to files (so a screenshot cannot silently go missing)."""
    pattern = re.compile(
        r'(?:src|srcset)="([^"]+\.(?:webp|png|jpg|svg))"|!\[[^\]]*\]\(([^)]+)\)'
    )
    files = [*READMES, ROOT / "docs/CONSOLE.md"]
    seen = 0
    for path in files:
        for m in pattern.finditer(path.read_text()):
            target = m.group(1) or m.group(2)
            if target.startswith(("http://", "https://")):
                continue
            assert (path.parent / target).is_file(), f"{path.name}: {target}"
            seen += 1
    assert seen >= 20


def test_console_images_are_small_and_complete():
    images = sorted((ROOT / "docs/images/console").rglob("*.webp"))
    assert images
    for image in images:
        assert image.stat().st_size < 300_000, image
    for loc in ("en", "zh-TW", "zh-CN"):
        for name in ("overview", "requests", "playground", "keys"):
            for scheme in ("dark", "light"):
                assert (
                    ROOT / f"docs/images/console/{loc}/{name}-{scheme}.webp"
                ).is_file()
        assert (ROOT / f"docs/images/console/{loc}/mobile-overview-dark.webp").is_file()


def test_console_guide_covers_every_page():
    text = (ROOT / "docs/CONSOLE.md").read_text()
    for page in (
        "overview",
        "requests",
        "logs",
        "diagnostics",
        "models",
        "downloads",
        "cache",
        "playground",
        "api",
        "keys",
        "settings",
        "island",
    ):
        assert f"images/console/en/{page}-" in text, page
