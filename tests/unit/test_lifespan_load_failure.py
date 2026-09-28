"""Single-model startup must survive a failed or timed-out model load.

Regression: the except branches called ``await engine.stop()`` on a name that
was never bound (``engine`` is a function local), so a bad model path crashed
lifespan with UnboundLocalError instead of starting "not ready".
"""

import asyncio

import pytest

import yunshu_engine.model_manager as model_manager
import yunshu_gateway.main as gw_main


@pytest.mark.parametrize("failure", ["error", "timeout"])
def test_lifespan_survives_model_load_failure(monkeypatch, failure):
    async def fake_instantiate(model_type, model_path):
        if failure == "timeout":
            await asyncio.sleep(10)
        raise RuntimeError("weights missing")

    monkeypatch.setattr(gw_main, "DEFAULT_MODEL", "/nonexistent/model")
    monkeypatch.setattr(model_manager, "instantiate_engine", fake_instantiate)
    monkeypatch.setattr(
        model_manager, "_detect_model_type", lambda path: None, raising=False
    )
    monkeypatch.setenv("YUNSHU_STARTUP_TIMEOUT", "0.05")

    async def run():
        async with gw_main.lifespan(gw_main.app):
            pass

    asyncio.run(run())
