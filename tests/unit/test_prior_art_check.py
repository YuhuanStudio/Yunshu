"""CPU-only checkpoint discovery contracts."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "prior_art_check", Path(__file__).parents[2] / "scripts/dev/prior_art_check.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_discovery_paginates_filters_and_deduplicates():
    calls = []

    def fetch(url):
        calls.append(url)
        if len(calls) == 1:
            return [
                {"id": "official/Laya"},
                {"id": "aac/Laya-MLX", "downloads": 2},
            ], "page2"
        return [
            {"id": "aac/Laya-MLX", "downloads": 3},
            {"id": "mlx-community/Laya"},
            {"id": "sahil/Laya-mxfp4"},
        ], None

    rows = module.discover("official/Laya", fetch)
    assert "search=Laya" in calls[0]
    assert len(calls) == 2
    assert [r["id"] for r in rows] == [
        "aac/Laya-MLX",
        "mlx-community/Laya",
        "sahil/Laya-mxfp4",
    ]


def test_network_failure_is_not_no_prior_art():
    def fetch(url):
        raise OSError("offline")

    with pytest.raises(OSError, match="offline"):
        module.discover("Laya", fetch)


def test_pagination_cycle_fails():
    def fetch(url):
        return [], url

    with pytest.raises(ValueError, match="cycle"):
        module.discover("Laya", fetch)
