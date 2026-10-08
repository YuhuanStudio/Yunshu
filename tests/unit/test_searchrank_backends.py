"""CPU-only preflight for the neural backend pilot; no model imports."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "searchrank_backends", ROOT / "scripts/research/searchrank_backends.py"
)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def test_parser_metrics_and_fail_closed(tmp_path):
    args = bench.parser().parse_args(
        [
            "--model",
            str(tmp_path),
            "--cache",
            str(tmp_path / "cache"),
            "--out",
            str(tmp_path / "out"),
            "--dry-run",
        ]
    )
    assert args.dry_run
    values = bench.metrics([[1, 0]] * len(bench.PAIRS), [0.1] * len(bench.PAIRS))
    assert values["correct"] == 12 and values["query_p50_ms"] == 100
    with pytest.raises(ValueError):
        bench.metrics([[float("nan"), 1]], [0.1])
    assert not bench.validate([{"complete": True}])[0]
    assert bench.run(args) == 0
