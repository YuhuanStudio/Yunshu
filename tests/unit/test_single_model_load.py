"""A bound VLM checkpoint cannot be replaced through the model-manager load API."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import HTTPException, Request

from yunshu_engine.vlm_engine import VLMEngine
from yunshu_gateway.routers import models


def test_single_vlm_load_rejects_without_calling_bound_loader(monkeypatch):
    # Do not initialize or load MLX: the old failure is Python argument binding.
    engine = VLMEngine.__new__(VLMEngine)
    engine._model_path = "bound-model"
    monkeypatch.setattr(models, "get_engine", lambda: engine)
    monkeypatch.setattr(models, "get_model_manager", lambda: None)
    monkeypatch.setattr(models, "_check_permission", lambda *args: None)
    monkeypatch.setattr(models, "log_operation", lambda *args, **kwargs: None)
    request = Request(
        {"type": "http", "method": "POST", "path": "/v1/models/load", "headers": []}
    )
    with ThreadPoolExecutor(1) as pool:
        monkeypatch.setattr("yunshu_engine.mlx_executor.get_mlx_executor", lambda: pool)
        with pytest.raises(HTTPException) as exc:
            asyncio.run(
                models.load_model(models.LoadModelRequest(model="bound-model"), request)
            )
    assert exc.value.status_code == 400
    assert "model manager" in exc.value.detail
    assert engine._model_path == "bound-model"
    assert "bound-model" not in models._model_ops_inflight
