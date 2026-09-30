"""What `yunshu launch` hands to Claude Code, Codex and opencode, derived from /v1/models."""

from __future__ import annotations

import json
import tomllib

from yunshu_cli.integrations import CodexIntegration, OpenCodeIntegration
from yunshu_cli.integrations.agent_config import (
    ModelInfo,
    claude_code_env,
    codex_catalog,
    codex_provider_toml,
    opencode_provider,
)

ITEM = {
    "id": "qwen3.8-27b",
    "display_name": "Qwen3.8 27B",
    "max_input_tokens": 262144,
    "max_tokens": 131072,
    "context_length": 262144,
    "architecture": {
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
    },
    "yunshu": {
        "context": {"length": 262144},
        "reasoning": {
            "supported": True,
            "effort_levels": ["xhigh", "medium", "low"],
            "default_effort": "xhigh",
        },
        "server_tools": {"web_search": {"available": True, "provider": "searxng"}},
    },
}


def test_model_info_from_models_item():
    m = ModelInfo.from_models_item(ITEM)
    assert m.id == "qwen3.8-27b" and m.context == 262144 and m.max_output == 131072
    assert m.vision and m.reasoning and m.effort_levels == ["xhigh", "medium", "low"]
    assert m.web_search and m.web_search_provider == "searxng"
    bare = ModelInfo.from_models_item({"id": "x"})
    assert bare.context == 131072 and not bare.reasoning and not bare.web_search


def test_claude_code_env():
    env = claude_code_env(
        ModelInfo.from_models_item(ITEM),
        "http://127.0.0.1:8000/",
        "tok",
        effort="medium",
    )
    assert (
        env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8000"
    )  # no trailing slash, no /v1
    assert env["ANTHROPIC_AUTH_TOKEN"] == "tok" and "ANTHROPIC_API_KEY" not in env
    for k in (
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_SMALL_FAST_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
    ):
        assert env[k] == "qwen3.8-27b"
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "262144"
    assert int(env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"]) < 262144
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "32000"
    assert env["CLAUDE_CODE_EFFORT_LEVEL"] == "medium"
    assert env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"


def test_codex_catalog_entry():
    cat = codex_catalog(ModelInfo.from_models_item(ITEM))
    (e,) = cat["models"]
    assert e["slug"] == "qwen3.8-27b" and e["context_window"] == 262144
    assert [x["effort"] for x in e["supported_reasoning_levels"]] == [
        "xhigh",
        "medium",
        "low",
    ]
    assert e["default_reasoning_level"] == "xhigh"
    assert e["input_modalities"] == ["text", "image"] and e["base_instructions"]
    assert e["visibility"] == "list" and e["shell_type"] == "shell_command"
    # a non-reasoning model offers no effort levels, so Codex sends none
    plain = codex_catalog(ModelInfo(id="m"))["models"][0]
    assert (
        plain["supported_reasoning_levels"] == []
        and "default_reasoning_level" not in plain
    )


def test_codex_toml_parses_and_carries_window():
    info = ModelInfo.from_models_item(ITEM)
    t = tomllib.loads(codex_provider_toml(info, "http://127.0.0.1:8000", "/x/cat.json"))
    assert t["model"] == "qwen3.8-27b" and t["model_provider"] == "yunshu"
    assert (
        t["model_context_window"] == 262144 and t["model_catalog_json"] == "/x/cat.json"
    )
    assert t["model_auto_compact_token_limit"] < 262144
    assert t["web_search"] == "live"
    p = t["model_providers"]["yunshu"]
    assert p["base_url"] == "http://127.0.0.1:8000/v1" and p["wire_api"] == "responses"
    off = tomllib.loads(codex_provider_toml(ModelInfo(id="m"), "http://h:1", "/c.json"))
    assert off["web_search"] == "disabled"


def test_codex_configure_merges_existing_config(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        'model = "old"\napproval_policy = "never"\n\n[model_providers.yunshu]\nname = "stale"\n\n'
        '[projects."/p"]\ntrust_level = "trusted"\n'
    )
    monkeypatch.setattr(CodexIntegration, "CONFIG_PATH", cfg)
    monkeypatch.setattr(CodexIntegration, "CATALOG_PATH", tmp_path / "cat.json")
    CodexIntegration().configure(
        8000, "", "qwen3.8-27b", "127.0.0.1", info=ModelInfo.from_models_item(ITEM)
    )
    t = tomllib.loads(cfg.read_text())
    assert t["model"] == "qwen3.8-27b" and t["approval_policy"] == "never"
    assert t["projects"]["/p"]["trust_level"] == "trusted"
    assert t["model_providers"]["yunshu"]["name"] == "Yunshu"
    assert (
        json.loads((tmp_path / "cat.json").read_text())["models"][0]["slug"]
        == "qwen3.8-27b"
    )
    assert list(tmp_path.glob("config.*.bak"))  # the old file was backed up


def test_opencode_provider_limits():
    p = opencode_provider(
        ModelInfo.from_models_item(ITEM), "http://127.0.0.1:8000", "k"
    )
    m = p["models"]["qwen3.8-27b"]
    assert m["limit"] == {"context": 262144, "output": 131072}
    assert m["reasoning"] and m["modalities"]["input"] == ["text", "image"]
    assert p["options"] == {"baseURL": "http://127.0.0.1:8000/v1", "apiKey": "k"}


def test_opencode_configure_writes_provider(tmp_path, monkeypatch):
    cfg = tmp_path / "opencode.json"
    cfg.write_text(json.dumps({"theme": "x", "provider": {"other": {}}}))
    monkeypatch.setattr(OpenCodeIntegration, "CONFIG_PATH", cfg)
    OpenCodeIntegration().configure(
        8000, "", "qwen3.8-27b", "127.0.0.1", info=ModelInfo.from_models_item(ITEM)
    )
    d = json.loads(cfg.read_text())
    assert (
        d["theme"] == "x"
        and "other" in d["provider"]
        and d["model"] == "yunshu/qwen3.8-27b"
    )
    assert (
        d["provider"]["yunshu"]["models"]["qwen3.8-27b"]["limit"]["context"] == 262144
    )


def test_launch_dry_run_prints_the_config(monkeypatch):
    """`yunshu launch <tool> --dry-run` (options after the tool name) reads /v1/models and prints."""
    import httpx
    from typer.testing import CliRunner

    from yunshu_cli.integrations import launch_app

    class Resp:
        def __init__(self, data, status=200):
            self._d, self.status_code = data, status

        def json(self):
            return self._d

        def raise_for_status(self):
            pass

    def fake_get(url, **kw):
        if url.endswith("/health"):
            return Resp({"status": "ok"})
        return Resp({"data": [ITEM]})

    monkeypatch.setattr(httpx, "get", fake_get)
    runner = CliRunner()
    for tool, needle in (
        ("claude", "CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144"),
        ("codex", "model_context_window = 262144"),
        ("opencode", '"context": 262144'),
    ):
        r = runner.invoke(
            launch_app, [tool, "--dry-run", "-u", "http://127.0.0.1:8000"]
        )
        assert r.exit_code == 0, (tool, r.output)
        assert needle in r.output, (tool, r.output)
    r = runner.invoke(
        launch_app, ["claude", "--dry-run", "--effort", "medium", "-u", "http://h:1"]
    )
    assert "CLAUDE_CODE_EFFORT_LEVEL=medium" in r.output
