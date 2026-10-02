"""Fail-closed CPU gates for the queued multi-arm audit harness."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from scripts.research import audit_flag_matrix as audit


def test_models_readiness_request_is_get(monkeypatch):
    seen = []
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=None)
    response.read.return_value = b'{"data": []}'

    def opened(req, timeout):
        seen.append(req.get_method())
        return response

    monkeypatch.setattr(audit.urllib.request, "urlopen", opened)
    assert audit.request("http://localhost:18990", "/v1/models", None) == {"data": []}
    assert seen == ["GET"]


def test_a_dry_run_can_never_authorize_a_27b_sweep():
    receipt = {
        "complete": True,
        "smoke": True,
        "dry_run": True,
        "source_sha": "s",
        "areas": ["external"],
        "arms": [{"rc": 0, "rows": [1]}],
    }
    assert not audit.validate_smoke(receipt, "s", ["external"])
    receipt["dry_run"] = False
    assert audit.validate_smoke(receipt, "s", ["external"])
    assert not audit.validate_smoke(receipt, "new-source", ["external"])
    receipt["arms"] = []
    assert not audit.validate_smoke(receipt, "s", ["external"])


@pytest.mark.parametrize("bad", ["parity", "digest", "complete", "missing_cell"])
def test_sweep_rc_zero_is_insufficient(monkeypatch, tmp_path, bad):
    records = [
        {"task": task, "block": block, "parity": True, "token_digest": "s"}
        for task in ("code", "prose")
        for block in (0, 6)
    ]
    records.append({"complete": True})
    if bad == "parity":
        records[0]["parity"] = False
    elif bad == "digest":
        records[0].pop("token_digest")
    elif bad == "complete":
        records.pop()
    else:
        records.pop(0)

    def run(command, *, cwd, stdout, stderr):
        stdout.write("\n".join(json.dumps(r) for r in records) + "\n")
        stdout.flush()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(audit.subprocess, "run", run)
    args = SimpleNamespace(out=tmp_path / "result.json", smoke=False, dry_run=False)
    with pytest.raises(RuntimeError):
        audit.subprocess_arm(args, "row_exact", 0, 0, 1024)


def test_tool_quality_counts_schema_errors_and_drops():
    body = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]
    }
    good = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"function": {"name": "read", "arguments": '{"path":"a.py"}'}}
                    ]
                }
            }
        ]
    }
    assert audit.quality(good, body)["malformed"] == 0
    good["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
        '{"path":42}'
    )
    assert audit.quality(good, body)["malformed"] == 1
    dropped = {"choices": [{"message": {"content": "<tool_call>read</tool_call>"}}]}
    assert audit.quality(dropped, body)["dropped"] is True


def test_summary_cannot_hide_missing_pair_or_token_difference():
    trace = {"prompt_digest": "prompt", "digest": "token"}
    rows = [
        {
            "area": "tree",
            "arm": 0,
            "rep": 0,
            "rc": 0,
            "rows": [{"case": "c", "wall_s": 2, "traces": [trace]}],
        }
    ]
    assert audit.summarize(rows)["tree"][0]["eligible"] is False
    rows.append(
        {
            "area": "tree",
            "arm": 1,
            "rep": 0,
            "rc": 0,
            "rows": [
                {"case": "c", "wall_s": 1, "traces": [{**trace, "digest": "changed"}]}
            ],
        }
    )
    summary = audit.summarize(rows)["tree"][0]
    assert summary["wall_speedup"] == 2 and summary["token_parity"] is False


def test_flag_census_covers_all_current_experiments():
    from scripts.research.experimental_flags_plan import plan
    from yunshu_engine import settings

    census = plan()
    assert {row["flag"] for row in census["flags"]} == {
        name
        for name, spec in settings.REGISTRY.items()
        if spec.stability == "experimental"
    }
    assert all(
        row["decide"] and row["evidence"] and row["measurement_argv"]
        for row in census["flags"]
    )
