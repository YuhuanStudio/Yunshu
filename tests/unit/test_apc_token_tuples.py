"""APC keys are built from token tuples; the per-token conversion must stay a C-speed `map(int, ...)`
(about 3x faster than the generator form, ~90 us per 8K-token key) and produce byte-identical keys."""

import ast
from pathlib import Path

import numpy as np
import pytest

from yunshu_engine.apc_manager import _sequence_hash

ROOT = Path(__file__).resolve().parents[2] / "python" / "yunshu_engine"


def _genexpr_token_tuples(path: Path) -> list[int]:
    hits = []
    for node in ast.walk(ast.parse(path.read_text())):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "tuple"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.GeneratorExp)
            and isinstance(node.args[0].elt, ast.Call)
            and getattr(node.args[0].elt.func, "id", "") == "int"
            and len(node.args[0].generators) == 1
            and not node.args[0].generators[0].ifs
        ):
            hits.append(node.lineno)
    return hits


@pytest.mark.parametrize("name", ["apc_manager.py", "apc_storage.py"])
def test_no_per_token_generator_in_apc_key_paths(name):
    assert _genexpr_token_tuples(ROOT / name) == [], "use tuple(map(int, ids))"


@pytest.mark.parametrize(
    "ids",
    [
        list(range(300)),
        np.arange(300, dtype=np.int32).tolist(),
        list(np.arange(300, dtype=np.int64)),  # numpy scalars
        tuple(range(300)),
        [True, False, 2, 3],
    ],
)
def test_sequence_hash_is_the_same_for_every_id_container(ids):
    ref = _sequence_hash(tuple(int(t) for t in ids), 7, 64)
    assert _sequence_hash(tuple(map(int, ids)), 7, 64) == ref
