"""the MCP `generate` tool must resolve ONLY the requested model and
never silently fall back to the default engine. The old
`except (KeyError, Exception): engine = get_engine()` served the default model on
any resolution failure — returning a different model's output with no error AND
bypassing the per-key model-isolation gate (which checks the REQUESTED model)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from yunshu_gateway.routers.mcp import _tool_generate


@pytest.mark.asyncio
async def test_unknown_model_errors_without_default_fallback():
    """A named-but-unresolvable model returns INVALID_PARAMS and must NOT touch the
    default engine."""
    default_called = {"hit": False}

    def _get_engine():
        default_called["hit"] = True
        return object()  # a sentinel "default engine" that must never be used

    async def _get_engine_for_model(model):
        raise KeyError(model)

    with (
        patch("yunshu_gateway.engine.get_engine_for_model", _get_engine_for_model),
        patch("yunshu_gateway.engine.get_engine", _get_engine),
    ):
        result = await _tool_generate(
            {"model": "model-b", "messages": [{"role": "user", "content": "hi"}]}, 1
        )

    assert "error" in result, result
    assert (
        result["error"]["code"] == -32602
    )  # INVALID_PARAMS, not a default-served answer
    assert default_called["hit"] is False  # no silent cross-model fallback


@pytest.mark.asyncio
async def test_load_failure_does_not_fall_back_to_default():
    """A non-KeyError load failure (e.g. OOM) must surface as an isError result, not
    a different model's output."""
    default_called = {"hit": False}

    def _get_engine():
        default_called["hit"] = True
        return object()

    async def _get_engine_for_model(model):
        raise RuntimeError("OOM loading model-b")

    with (
        patch("yunshu_gateway.engine.get_engine_for_model", _get_engine_for_model),
        patch("yunshu_gateway.engine.get_engine", _get_engine),
    ):
        result = await _tool_generate(
            {"model": "model-b", "messages": [{"role": "user", "content": "hi"}]}, 2
        )

    # Outer handler turns the propagated error into an isError tool result.
    assert result.get("result", {}).get("isError") is True or "error" in result
    assert default_called["hit"] is False  # still no cross-model fallback
