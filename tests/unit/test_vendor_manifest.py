"""vendor.json stays consistent: every tracked local file exists and is well formed."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT / "vendor.json").read_text())


def test_entries_are_well_formed():
    for kind in ("vendored", "derived", "inspired", "patches"):
        for e in MANIFEST[kind]:
            assert e["kind"] == kind
            assert (ROOT / e["path"]).is_file(), e["path"]
            assert e["repo"] and e["license"] and e["clone"]
            if kind == "patches":
                assert e["module"] and e["symbol"] and e["signature"]
                assert len(e["source_sha256"]) == 64
            else:
                assert e["commit"]
            if kind in ("derived", "inspired"):
                assert e["upstream_paths"]


def test_tracked_files_credit_upstream():
    for kind in ("derived", "inspired", "patches"):
        for e in MANIFEST[kind]:
            head = (ROOT / e["path"]).read_text()[:3000]
            assert "# Upstream (" in head or "# Patches upstream" in head, e["path"]
