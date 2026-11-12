"""vendor.json credits mlx-vlm to its official upstream (Blaizzy/mlx-vlm), never to a fork or mirror."""

import json
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[2] / "vendor.json"
OFFICIAL_MLX_VLM = "https://github.com/Blaizzy/mlx-vlm"


def _urls(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in ("repo", "url") and isinstance(v, str):
                yield v
            yield from _urls(v)
    elif isinstance(node, list):
        for v in node:
            yield from _urls(v)


def test_mlx_vlm_entries_point_at_the_official_repo():
    manifest = json.loads(VENDOR.read_text())
    mlx_vlm = [e for e in _urls(manifest) if "/mlx-vlm" in e]
    assert mlx_vlm, "expected mlx-vlm references in vendor.json"
    assert all(u.startswith(OFFICIAL_MLX_VLM) for u in mlx_vlm), mlx_vlm


def test_no_neywa_mirror_in_upstream_urls():
    manifest = json.loads(VENDOR.read_text())
    assert not [u for u in _urls(manifest) if "Neywa-Labs/mlx-vlm" in u]
