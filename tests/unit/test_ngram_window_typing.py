"""The dynamic drafter must not leak Any through the typed driver boundary."""

import ast
from pathlib import Path
from textwrap import indent

from mypy import api


def test_ngram_window_retains_its_list_result_type(tmp_path):
    path = Path(__file__).parents[2] / "python/yunshu_engine/round_driver/driver.py"
    driver = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "RoundDriver"
    )
    method = next(
        node
        for node in driver.body
        if isinstance(node, ast.FunctionDef) and node.name == "_ngram_window"
    )
    source = """from typing import Any
logger: Any
class _Row:
    ngram: Any
    force: Any
    req: Any
    generated: int
    context: list[int]
    ngram_logged: bool
class RoundDriver:
    batch: Any
    _ngram_untrimmable_logged: bool
""" + indent(ast.unparse(method), "    ")
    output, errors, status = api.run(
        [
            "--strict",
            "--no-incremental",
            "--cache-dir",
            str(tmp_path / "mypy"),
            "--no-error-summary",
            "-c",
            source,
        ]
    )
    assert status == 0, output + errors
