"""The live probe must reject missing sources/terminal records before loading MLX."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "docs_ask_smoke",
    Path(__file__).resolve().parents[2] / "scripts/research/docs_ask_smoke.py",
)
assert SPEC and SPEC.loader
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


def test_parser_needs_tokens_sources_and_done():
    wire = 'event: sources\ndata: {"sources":[{"url":"/docs#start"}]}\n\nevent: token\ndata: {"text":"answer [1]"}\n\nevent: done\ndata: {}\n\n'
    assert SMOKE.parse_answer(wire)["answer"] == "answer [1]"
    for bad in (
        wire.replace("event: done", "event: ignored"),
        wire.replace('"sources":[{"url":"/docs#start"}]', '"sources":[]'),
        'event: error\ndata: {"code":"engine_error"}\n\n',
    ):
        with pytest.raises(RuntimeError):
            SMOKE.parse_answer(bad)
