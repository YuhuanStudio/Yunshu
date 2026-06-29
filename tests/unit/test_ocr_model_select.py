"""the /v1/ocr endpoint had the wrong-model keystone, unswept.

(1) It grabbed the FIRST loaded OCREngine (and first loaded VLM fallback), ignoring the
    requested `model` — with two OCR models loaded, a request for B was served by A.
(2) _check_model_access(model) at the top is a no-op when `model` is empty (the default),
    and the resolved engine was never re-checked → a model-scoped key could OCR through a
    model it cannot access by omitting `model`. Now: select by model_id, and re-check the
    RESOLVED model against the key's scope (both the OCREngine and VLM-fallback paths)."""
from __future__ import annotations

import inspect
import os
import types

import pytest
from fastapi.testclient import TestClient


class _FakeOCR:
    """Stands in for OCREngine; returns a marker text identifying itself."""
    def __init__(self, tag):
        self.tag = tag

    async def extract_text(self, path, language=None, task="text"):
        return {"text": f"from-{self.tag}", "confidence": 1.0,
                "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


def _entry(model_id, engine):
    return types.SimpleNamespace(model_id=model_id, engine=engine, is_loaded=True)


@pytest.fixture
def _client(monkeypatch):
    os.environ["YUNSHU_AUTH_DISABLED"] = "true"
    from yunshu_engine.ocr_engine import OCREngine
    from yunshu_gateway import engine as eng_mod

    a = OCREngine.__new__(OCREngine)
    b = OCREngine.__new__(OCREngine)
    # graft the marker behavior onto the bare instances
    a.extract_text = _FakeOCR("modelA").extract_text
    b.extract_text = _FakeOCR("modelB").extract_text
    mgr = types.SimpleNamespace(
        list_entries=lambda: [_entry("ocr-A", a), _entry("ocr-B", b)],
    )
    monkeypatch.setattr(eng_mod, "get_model_manager", lambda: mgr)
    from yunshu_gateway.main import create_app
    yield TestClient(create_app(), raise_server_exceptions=False)
    os.environ.pop("YUNSHU_AUTH_DISABLED", None)


def test_ocr_serves_requested_model_not_first(_client):
    resp = _client.post("/v1/ocr",
                        data={"model": "ocr-B", "task": "text"},
                        files={"file": ("x.png", b"\x89PNG\r\n\x1a\n" + b"0" * 32, "image/png")})
    assert resp.status_code == 200, resp.text
    assert resp.json()["text"] == "from-modelB"  # NOT from-modelA (the first entry)


def test_ocr_empty_model_falls_back_to_first(_client):
    resp = _client.post("/v1/ocr",
                        data={"task": "text"},
                        files={"file": ("x.png", b"\x89PNG\r\n\x1a\n" + b"0" * 32, "image/png")})
    assert resp.status_code == 200, resp.text
    assert resp.json()["text"] == "from-modelA"  # first-of-type when no model named


def test_source_rechecks_resolved_model():
    from yunshu_gateway.routers import ocr
    src = inspect.getsource(ocr)
    # both the OCREngine and VLM-fallback paths re-check the resolved id
    assert "_check_model_access(request, ocr_model_id)" in src
    assert "_check_model_access(request, vlm_model_id)" in src
    # selection matches by model_id
    assert "entry.model_id == model or entry.model_id.lower() == _model_lower" in src
