"""Traffic inventory must include Responses instructions, not infer an empty system."""

import hashlib
import importlib.util
import json
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[2] / "scripts/research/agentic/characterize_auxiliary.py"
)
spec = importlib.util.spec_from_file_location("auxiliary_inventory", SCRIPT)
assert spec is not None and spec.loader is not None
inventory_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inventory_module)


def test_responses_instructions_and_developer_input_are_fingerprinted(tmp_path):
    directory = tmp_path / "cap-codex"
    directory.mkdir()
    body = dict(
        model="captured-model",
        instructions="Actual Responses instructions",
        input=[
            dict(
                role="developer",
                content=[dict(type="input_text", text="Developer context")],
            ),
            dict(role="user", content="Question"),
        ],
        tools=[dict(type="function")],
    )
    (directory / "0001-req.json").write_text(json.dumps(body))
    body["input"].append(dict(role="assistant", content="Reply"))
    (directory / "0002-req.json").write_text(json.dumps(body))
    records = inventory_module.inventory(tmp_path)
    assert len(records) == 1
    record = records[0]
    assert record["captures"] == 2
    assert record["tools"] == 1
    assert record["max_tokens"] is None
    parts = record["system_parts"]
    assert [part["source"] for part in parts] == ["instructions", "input[0].developer"]
    assert (
        parts[0]["sha256"]
        == hashlib.sha256(b"Actual Responses instructions").hexdigest()
    )
    assert record["system_chars"] > len(body["instructions"])


def test_captured_title_inventory_hash_matches_gateway(tmp_path):
    from yunshu_gateway.admission import _OPENCODE_TITLE_SHA256

    directory = tmp_path / "cap-opencode"
    directory.mkdir()
    fixture = Path(__file__).parents[1] / "fixtures/auxiliary/opencode_title.json"
    (directory / "0001-req.json").write_bytes(fixture.read_bytes())
    (record,) = inventory_module.inventory(tmp_path)
    assert record["system_sha256"] == _OPENCODE_TITLE_SHA256
    assert record["system_chars"] == 2096
    assert record["roles"] == ["system", "user", "user"]
