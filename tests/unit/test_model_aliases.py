"""YUNSHU_MODEL_ALIASES: agent-style model names (claude-sonnet-4-5, opus, gpt-5) map onto a served model."""

from __future__ import annotations

import json

import pytest

from yunshu_engine.model_manager import ModelManager


@pytest.fixture
def manager():
    m = ModelManager.__new__(ModelManager)
    m._entries = {"qwen3.8-27b": object(), "qwen3.5-9b": object()}
    return m


def test_no_aliases_keeps_unknown_names_unresolved(manager, monkeypatch):
    monkeypatch.delenv("YUNSHU_MODEL_ALIASES", raising=False)
    assert manager.resolve_model_id("claude-sonnet-4-5") is None
    assert manager.resolve_model_id("qwen3.8-27b") == "qwen3.8-27b"


def test_prefix_exact_and_wildcard_aliases(manager, monkeypatch):
    monkeypatch.setenv(
        "YUNSHU_MODEL_ALIASES",
        json.dumps(
            {
                "claude-haiku*": "qwen3.5-9b",
                "claude-*": "qwen3.8-27b",
                "opus": "qwen3.8-27b",
                "*": "qwen3.5-9b",
            }
        ),
    )
    assert (
        manager.resolve_model_id("claude-haiku-4-5") == "qwen3.5-9b"
    )  # first match wins
    assert manager.resolve_model_id("claude-sonnet-4-5") == "qwen3.8-27b"
    assert manager.resolve_model_id("Opus") == "qwen3.8-27b"
    assert manager.resolve_model_id("gpt-5") == "qwen3.5-9b"  # the "*" fallback
    assert (
        manager.resolve_model_id("qwen3.8-27b") == "qwen3.8-27b"
    )  # real names win over aliases


def test_alias_to_an_unserved_model_resolves_to_nothing(manager, monkeypatch):
    monkeypatch.setenv(
        "YUNSHU_MODEL_ALIASES", json.dumps({"claude-*": "not-installed"})
    )
    assert manager.resolve_model_id("claude-sonnet-4-5") is None
