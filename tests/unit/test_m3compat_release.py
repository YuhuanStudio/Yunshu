"""Fail closed when a release HTTP run lacks committed-token evidence."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "m3compat_release", Path(__file__).parents[2] / "scripts/dev/m3compat_release.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_receipts_require_one_complete_nonempty_record_per_request():
    log = 'm3compat_receipt {"ids": [17, 21], "fp_exact_rows": {}}\n'
    assert module.read_receipts(log, 1)["ids"] == [17, 21]
    with pytest.raises(RuntimeError, match="missing or empty"):
        module.read_receipts(log, 2)
    with pytest.raises(RuntimeError, match="missing or empty"):
        module.read_receipts('m3compat_receipt {"ids": []}\n', 1)
    with pytest.raises(RuntimeError, match="missing or empty"):
        module.read_receipts("", 1)


def test_agent_cases_keep_full_tool_request_token_budget(monkeypatch, tmp_path):
    import json
    import sys
    from types import SimpleNamespace

    body = {
        "max_tokens": 32000,
        "messages": [{"role": "user", "content": "Apply the requested patch."}],
        "tools": [{"type": "function", "function": {"name": "apply_patch"}}],
        "tool_choice": "auto",
    }
    for name in ("0002-req.json", "0004-req.json", "0006-req.json", "0003-req.json"):
        (tmp_path / name).write_text(json.dumps(body))
    monkeypatch.setitem(
        sys.modules,
        "tfbench",
        SimpleNamespace(
            BODIES=tmp_path,
            BODIES2=tmp_path,
            load_prompt=lambda _: "context",
            req=lambda *a: {},
        ),
    )
    for _, request in module.cases(False)[6:]:
        assert request["max_tokens"] == 32000
        assert request["tools"] == body["tools"]
        assert request["messages"] == body["messages"]
        assert request["temperature"] == 0
