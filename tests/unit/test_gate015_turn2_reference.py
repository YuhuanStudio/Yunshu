import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def probe():
    path = Path(__file__).parents[2] / "scripts/research/gate015_turn2_reference.py"
    spec = importlib.util.spec_from_file_location("gate015_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_keeps_the_release_reply_and_followup_prompt():
    module = probe()
    bench = SimpleNamespace(
        req=lambda *args: {"messages": [{"role": "user", "content": args[1]}]}
    )
    row = dict(
        part="decode", phase="cold", ct=2048, finish="length", text="original reply"
    )
    body = module.followup_body(bench, "model", "original prompt", [row])
    assert [m["content"] for m in body["messages"]] == [
        "original prompt",
        "original reply",
        "Continue with the next part, at the same length.",
    ]
    with pytest.raises(ValueError):
        module.followup_body(bench, "model", "prompt", [{**row, "ct": 12}])


@pytest.mark.parametrize(
    "result",
    [{}, {"ct": 255, "finish": "length"}, {"ct": 256, "finish": "length", "cached": 1}],
)
def test_reference_rejects_incomplete_or_cached_results(result):
    with pytest.raises(ValueError):
        probe().validate(result)
