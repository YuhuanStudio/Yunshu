"""(MED): /v1/messages/count_tokens ignored cached_content → undercounted by the
entire cached prefix that generation prepends into the system prompt.

create_message resolves a cached_content handle and prepends its stored text to req.system
(so prompt_tokens includes it); count_tokens had zero cached_content handling. They now share
ONE resolver (_resolve_cached_content_text) — generation marks a real READ (mutate=True),
count_tokens does a non-mutating estimation read (mutate=False) — so the estimate matches the
real prompt and the two can't drift (the recurring "claimed-mirrors but diverged" class).
"""

from __future__ import annotations

import inspect
import types

import pytest
from fastapi import HTTPException

from yunshu_gateway.routers import (
    anthropic as A,  # noqa: N812  # intentional short module alias
)
from yunshu_gateway.routers.anthropic import _resolve_cached_content_text


class _Entry:
    def __init__(self, messages, owner=None, model=None):
        self.messages = messages
        self.owner = owner
        self.model = model
        self.read_count = 0


class _Store:
    def __init__(self, entry):
        self._entry = entry
        self.use_called = False
        self.get_called = False

    def use(self, key):
        self.use_called = True
        if self._entry is not None:
            self._entry.read_count += 1
        return self._entry

    def get(self, key):
        self.get_called = True
        return self._entry


def _req():
    return types.SimpleNamespace(state=types.SimpleNamespace(role="", rbac_key=None))


def _patch_store(monkeypatch, entry):
    store = _Store(entry)
    monkeypatch.setattr("yunshu_gateway.explicit_cache.get_store", lambda: store)
    return store


def test_estimation_read_uses_get_not_use(monkeypatch):
    entry = _Entry([{"role": "user", "content": "CACHED PREFIX"}])
    store = _patch_store(monkeypatch, entry)
    assert (
        _resolve_cached_content_text("cachedContents/x", "m", _req(), mutate=False)
        == "CACHED PREFIX"
    )
    assert store.get_called and not store.use_called  # estimation must not bump usage
    assert entry.read_count == 0


def test_generation_read_bumps_usage(monkeypatch):
    entry = _Entry([{"role": "user", "content": "CACHED"}])
    store = _patch_store(monkeypatch, entry)
    # bare handle (no "cachedContents/" prefix) is normalized
    assert _resolve_cached_content_text("x", "m", _req(), mutate=True) == "CACHED"
    assert store.use_called and entry.read_count == 1


def test_model_mismatch_raises_400(monkeypatch):
    _patch_store(
        monkeypatch, _Entry([{"role": "user", "content": "C"}], model="other-model")
    )
    with pytest.raises(HTTPException) as e:
        _resolve_cached_content_text("x", "my-model", _req(), mutate=False)
    assert e.value.status_code == 400


def test_block_content_is_flattened(monkeypatch):
    entry = _Entry(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "A"},
                    {"type": "text", "text": "B"},
                ],
            }
        ]
    )
    _patch_store(monkeypatch, entry)
    assert _resolve_cached_content_text("x", "m", _req(), mutate=False) == "A B"


def test_absent_or_empty_returns_empty(monkeypatch):
    assert _resolve_cached_content_text(None, "m", _req(), mutate=False) == ""
    assert _resolve_cached_content_text("", "m", _req(), mutate=False) == ""
    _patch_store(monkeypatch, None)  # handle not found
    assert _resolve_cached_content_text("x", "m", _req(), mutate=False) == ""


def test_generation_and_count_tokens_share_the_resolver():
    gen = "\n".join(
        ln.split("#", 1)[0] for ln in inspect.getsource(A.create_message).splitlines()
    )
    ct = "\n".join(
        ln.split("#", 1)[0] for ln in inspect.getsource(A.count_tokens).splitlines()
    )
    assert "_resolve_cached_content_text(" in gen and "mutate=True" in gen
    assert "_resolve_cached_content_text(" in ct and "mutate=False" in ct
