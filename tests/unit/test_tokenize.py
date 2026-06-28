"""unit coverage for tokenize/detokenize router (zero coverage).

Tests:
- POST /v1/detokenize happy path, empty tokens, unknown model -> 404
- POST /v1/token_count happy path
- _resolve_context_limit fallback (unknown model -> 0)
- _resolve_tokenizer 404 path
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class FakeTokenizer:
    """Minimal tokenizer stub: deterministic encode/decode for assertions."""

    def encode(self, text, add_special_tokens=True):
        # Each whitespace-token becomes one int; ignore special-token flag.
        return [10 + i for i, _ in enumerate(text.split())]

    def decode(self, tokens, skip_special_tokens=True):
        # Just return a stable, sortable string showing how many tokens decoded.
        return f"decoded[{len(tokens)}]"


class FakeEngine:
    is_loaded = True

    def __init__(self):
        self._tokenizer = FakeTokenizer()


class FakeEntry:
    def __init__(self, model_id: str, engine):
        self.model_id = model_id
        self.engine = engine
        self.is_loaded = True


class FakeModelManager:
    """Stand-in for yunshu_engine.model_manager.ModelManager."""

    def __init__(self):
        self._entries: dict[str, FakeEntry] = {}

    def register(self, model_id: str, engine):
        self._entries[model_id] = FakeEntry(model_id, engine)

    def get_entry(self, model_id):
        return self._entries.get(model_id)

    def list_entries(self):
        return list(self._entries.values())


def _make_app(monkeypatch, *, register_model: str | None = "qwen-test"):
    """Build a FastAPI app with tokenize router and an installed fake manager."""
    import yunshu_gateway.routers.tokenize as tok_mod
    # Force auth-disabled so the `_check_permission` import from models.py
    # short-circuits cleanly (it reads os.environ).
    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "true")

    mgr = FakeModelManager()
    if register_model is not None:
        mgr.register(register_model, FakeEngine())

    # Patch the gateway-engine getter that tokenize.py imports at module load.
    monkeypatch.setattr(tok_mod, "get_model_manager", lambda: mgr)
    monkeypatch.setattr(tok_mod, "get_engine", lambda: None)

    app = FastAPI()
    app.include_router(tok_mod.router, prefix="/v1")
    return app, mgr


class TestDetokenize:
    def test_detokenize_happy_path(self, monkeypatch):
        app, _ = _make_app(monkeypatch)
        client = TestClient(app)
        r = client.post(
            "/v1/detokenize",
            json={"model": "qwen-test", "tokens": [1, 2, 3]},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["text"] == "decoded[3]"
        assert body["model"] == "qwen-test"

    def test_detokenize_empty_tokens(self, monkeypatch):
        app, _ = _make_app(monkeypatch)
        client = TestClient(app)
        r = client.post(
            "/v1/detokenize",
            json={"model": "qwen-test", "tokens": []},
        )
        assert r.status_code == 200, r.text
        assert r.json()["text"] == "decoded[0]"

    def test_detokenize_unknown_model_404(self, monkeypatch):
        app, _ = _make_app(monkeypatch, register_model=None)
        client = TestClient(app)
        r = client.post(
            "/v1/detokenize",
            json={"model": "no-such-model", "tokens": [1, 2, 3]},
        )
        assert r.status_code == 404, r.text
        assert "no-such-model" in r.text


class TestTokenize:
    def test_tokenize_single_string(self, monkeypatch):
        app, _ = _make_app(monkeypatch)
        client = TestClient(app)
        r = client.post(
            "/v1/tokenize",
            json={"model": "qwen-test", "text": "alpha beta gamma"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # FakeTokenizer returns [10, 11, 12] for three whitespace-tokens
        assert body["tokens"] == [10, 11, 12]
        assert body["count"] == 3
        assert body["model"] == "qwen-test"

    def test_tokenize_unknown_model_404(self, monkeypatch):
        app, _ = _make_app(monkeypatch, register_model=None)
        client = TestClient(app)
        r = client.post(
            "/v1/tokenize",
            json={"model": "nope", "text": "hello"},
        )
        assert r.status_code == 404, r.text


class TestTokenCount:
    def test_token_count_happy_path(self, monkeypatch):
        app, _ = _make_app(monkeypatch)
        client = TestClient(app)
        r = client.post(
            "/v1/token_count",
            json={"model": "qwen-test", "prompt": "one two three four"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # FakeTokenizer returns 4 ints for 4 whitespace-tokens.
        assert body["token_count"] == 4
        assert body["model"] == "qwen-test"
        # Unknown model context -> 0, hence over_context_limit == False
        assert body["max_context_tokens"] == 0
        assert body["over_context_limit"] is False

    def test_token_count_with_max_tokens_override(self, monkeypatch):
        """Caller-supplied max_tokens > 0 should propagate as the limit."""
        app, _ = _make_app(monkeypatch)
        client = TestClient(app)
        r = client.post(
            "/v1/token_count",
            json={"model": "qwen-test", "prompt": "a b c", "max_tokens": 128},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["max_context_tokens"] == 128
        # 3 tokens vs 128 limit -> not over
        assert body["over_context_limit"] is False

    def test_token_count_input_alias(self, monkeypatch):
        """OpenAI-style 'input' alias should be accepted as 'prompt'."""
        app, _ = _make_app(monkeypatch)
        client = TestClient(app)
        r = client.post(
            "/v1/token_count",
            json={"model": "qwen-test", "input": "a b"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["token_count"] == 2


class TestResolveContextLimit:
    """Direct unit tests for _resolve_context_limit fallback path."""

    def test_unknown_model_returns_zero(self, monkeypatch):
        import yunshu_gateway.routers.tokenize as tok_mod
        monkeypatch.setattr(tok_mod, "get_model_manager", lambda: FakeModelManager())
        monkeypatch.setattr(tok_mod, "get_engine", lambda: None)
        assert tok_mod._resolve_context_limit("does-not-exist") == 0

    def test_engine_attr_returns_max_position_embeddings(self, monkeypatch):
        import yunshu_gateway.routers.tokenize as tok_mod

        class EngineWithCtx:
            is_loaded = True
            max_position_embeddings = 4096

        mgr = FakeModelManager()
        mgr.register("ctx-model", EngineWithCtx())
        monkeypatch.setattr(tok_mod, "get_model_manager", lambda: mgr)
        monkeypatch.setattr(tok_mod, "get_engine", lambda: None)
        assert tok_mod._resolve_context_limit("ctx-model") == 4096


class TestResolveTokenizer:
    """Direct tests for _resolve_tokenizer fallbacks."""

    def test_resolve_tokenizer_exact_match(self, monkeypatch):
        import yunshu_gateway.routers.tokenize as tok_mod
        mgr = FakeModelManager()
        mgr.register("exact", FakeEngine())
        monkeypatch.setattr(tok_mod, "get_model_manager", lambda: mgr)
        monkeypatch.setattr(tok_mod, "get_engine", lambda: None)
        tok = tok_mod._resolve_tokenizer("exact")
        assert isinstance(tok, FakeTokenizer)

    def test_resolve_tokenizer_case_insensitive(self, monkeypatch):
        import yunshu_gateway.routers.tokenize as tok_mod
        mgr = FakeModelManager()
        mgr.register("Qwen3-7B", FakeEngine())
        monkeypatch.setattr(tok_mod, "get_model_manager", lambda: mgr)
        monkeypatch.setattr(tok_mod, "get_engine", lambda: None)
        # Lower-case lookup must succeed via the case-insensitive fallback
        tok = tok_mod._resolve_tokenizer("qwen3-7b")
        assert isinstance(tok, FakeTokenizer)

    def test_resolve_tokenizer_404_when_missing(self, monkeypatch):
        from fastapi import HTTPException

        import yunshu_gateway.routers.tokenize as tok_mod
        monkeypatch.setattr(tok_mod, "get_model_manager", lambda: FakeModelManager())
        monkeypatch.setattr(tok_mod, "get_engine", lambda: None)
        with pytest.raises(HTTPException) as ei:
            tok_mod._resolve_tokenizer("no-model")
        assert ei.value.status_code == 404
