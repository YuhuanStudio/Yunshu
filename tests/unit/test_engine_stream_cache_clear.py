"""A streaming request's normal end clears the MLX buffer pool, and the error handlers clear it too (guard)."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "python/yunshu_engine/engine_stream.py"


def _clear_call_locations():
    tree = ast.parse(SRC.read_text())
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "clear_cache"
    ]
    out = []
    for call in calls:
        node = call
        in_handler = False
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.ExceptHandler):
                in_handler = True
                break
        out.append((call.lineno, in_handler))
    return out


def test_normal_stream_end_clears_the_buffer_pool():
    locations = _clear_call_locations()
    assert any(not in_handler for _, in_handler in locations), (
        "the normal stream end must clear the MLX buffer pool"
    )


def test_error_handlers_clear_the_buffer_pool():
    assert any(in_handler for _, in_handler in _clear_call_locations())
