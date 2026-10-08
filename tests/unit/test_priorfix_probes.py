import importlib.util
from pathlib import Path

import pytest


def probe():
    path = Path(__file__).parents[2] / "scripts/research/priorfix_embedding_parity.py"
    spec = importlib.util.spec_from_file_location("priorfix_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_embedding_probe_fails_closed():
    p = probe()
    assert p.compare([[1, 0]], [[1, 0]], 0.999999)["passed"]
    assert not p.compare([[1, 0]], [[0, 1]], 0.999999)["passed"]
    with pytest.raises(ValueError):
        p.compare([[1, 0]], [[float("nan"), 0]], 0.99)
    args = p.parser().parse_args(
        [
            "--model",
            "fixture",
            "--reference",
            "fixture.json",
            "--out",
            "out",
            "--dry-run",
        ]
    )
    assert args.dry_run
