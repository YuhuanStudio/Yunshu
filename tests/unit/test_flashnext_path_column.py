"""Every optimization receives an explicit Flash-Next architecture decision."""

import json
from pathlib import Path


def test_flashnext_column_covers_every_feature_with_source_evidence():
    root = Path(__file__).resolve().parents[2]
    matrix = json.loads((root / "python/yunshu_engine/path_matrix.json").read_text())
    column = matrix["architecture_columns"]["flashnext"]
    assert set(column) == set(matrix["features"])
    for feature, cell in column.items():
        assert cell["s"] in {"engaged", "na", "missing", "partial"}, feature
        assert cell.get("why") or cell.get("ev"), feature
        if cell["s"] in {"missing", "partial"}:
            assert cell.get("owner") and cell.get("impact"), feature
        for rel, needle in cell.get("ev", []):
            assert needle in (root / rel).read_text(), feature
