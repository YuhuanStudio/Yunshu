"""The console's docs name only routes the gateway really registers, and the generated
configuration pages match the settings registry. CPU only; needs node for the route finder."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, FRONTEND / "scripts" / f"{name}.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_docs_mention_only_registered_routes(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    routes = _load("dump_routes").collect_routes()
    assert "POST /v1/chat/completions" in routes and "WS /v1/realtime" in routes
    file = tmp_path / "routes.json"
    file.write_text(json.dumps(routes))
    run = subprocess.run(
        [node, str(FRONTEND / "scripts" / "docs-routes.mjs"), str(file)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert run.returncode == 0, run.stdout + run.stderr


def test_configuration_pages_match_the_settings_registry():
    sys.path.insert(0, str(ROOT / "python"))
    assert _load("gen_docs_config").main(check=True) == 0, (
        "regenerate with: PYTHONPATH=python python frontend/scripts/gen_docs_config.py"
    )


def test_guide_pointers_name_real_console_pages():
    """docs/guides/*.md that have a counterpart in the console's docs point at its MDX source."""
    import re

    seen = 0
    for md in (ROOT / "docs" / "guides").glob("*.md"):
        for m in re.finditer(
            r"\]\(\.\./\.\./(frontend/docs/[^)]+\.mdx)\)", md.read_text()
        ):
            seen += 1
            page = ROOT / m.group(1)
            assert page.is_file(), (
                f"{md.name} points at {m.group(1)}, which does not exist"
            )
            for suffix in (".zh-TW", ".zh-CN"):
                assert page.with_name(page.stem + suffix + ".mdx").is_file()
    assert seen >= 9
