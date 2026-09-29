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
        r = TestClient(app).post(
            "/v1/detokenize", json={"model": "qwen-test", "tokens": [1, 2, 3]}
        )
        assert r.status_code == 200, r.text
        assert r.json() == {"prompt": "decoded[3]"}

    def test_detokenize_empty_tokens(self, monkeypatch):
        app, _ = _make_app(monkeypatch)
        r = TestClient(app).post(
            "/v1/detokenize", json={"model": "qwen-test", "tokens": []}
        )
        assert r.status_code == 200, r.text
        assert r.json()["prompt"] == "decoded[0]"

    def test_detokenize_unknown_model_404(self, monkeypatch):
        app, _ = _make_app(monkeypatch, register_model=None)
        r = TestClient(app).post(
            "/v1/detokenize", json={"model": "no-such-model", "tokens": [1, 2, 3]}
        )
        assert r.status_code == 404, r.text
        assert "no-such-model" in r.text


class TestTokenize:
    def test_prompt(self, monkeypatch):
        app, _ = _make_app(monkeypatch)
        r = TestClient(app).post(
            "/v1/tokenize", json={"model": "qwen-test", "prompt": "alpha beta gamma"}
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["tokens"] == [10, 11, 12]
        assert body["count"] == 3
        assert "max_model_len" in body
        assert "token_strs" not in body

    def test_legacy_text_alias_and_batch(self, monkeypatch):
        app, _ = _make_app(monkeypatch)
        r = TestClient(app).post(
            "/v1/tokenize", json={"model": "qwen-test", "text": ["a b", "c"]}
        )
        assert r.status_code == 200, r.text
        assert r.json()["tokens"] == [[10, 11], [10]]
        assert r.json()["count"] == 3

    def test_model_optional_and_root_path(self, monkeypatch):
        import yunshu_gateway.routers.tokenize as tok_mod

        app, mgr = _make_app(monkeypatch)
        monkeypatch.setattr(
            tok_mod, "get_engine", lambda: mgr.get_entry("qwen-test").engine
        )
        monkeypatch.setattr(tok_mod, "get_model_manager", lambda: None)
        app.include_router(tok_mod.router)  # vLLM-native root path
        r = TestClient(app).post("/tokenize", json={"prompt": "x y"})
        assert r.status_code == 200, r.text
        assert r.json()["count"] == 2

    def test_token_strs(self, monkeypatch):
        app, _ = _make_app(monkeypatch)
        r = TestClient(app).post(
            "/v1/tokenize",
            json={"model": "qwen-test", "prompt": "a b", "return_token_strs": True},
        )
        assert r.json()["token_strs"] == ["decoded[1]", "decoded[1]"]

    def test_messages_use_chat_template(self, monkeypatch):
        import yunshu_gateway.routers.tokenize as tok_mod

        seen = {}

        class ChatTok(FakeTokenizer):
            def apply_chat_template(self, msgs, **kw):
                seen["msgs"] = msgs
                seen["kw"] = kw
                return "<u> " + " ".join(m["content"] for m in msgs)

        eng = FakeEngine()
        eng._tokenizer = ChatTok()
        app, mgr = _make_app(monkeypatch)
        mgr.register("chat", eng)
        r = TestClient(app).post(
            "/v1/tokenize",
            json={
                "model": "chat",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "hi there"}]}
                ],
            },
        )
        assert r.status_code == 200, r.text
        assert r.json()["count"] == 3
        assert seen["msgs"][0]["content"] == "hi there"
        assert seen["kw"]["add_generation_prompt"] is True
        assert tok_mod  # module imported

    def test_prompt_and_messages_are_exclusive(self, monkeypatch):
        app, _ = _make_app(monkeypatch)
        c = TestClient(app)
        assert c.post("/v1/tokenize", json={"model": "qwen-test"}).status_code in (
            400,
            422,
        )
        assert c.post(
            "/v1/tokenize",
            json={"model": "qwen-test", "prompt": "a", "messages": []},
        ).status_code in (400, 422)

    def test_unknown_model_404(self, monkeypatch):
        app, _ = _make_app(monkeypatch, register_model=None)
        r = TestClient(app).post(
            "/v1/tokenize", json={"model": "nope", "prompt": "hello"}
        )
        assert r.status_code == 404, r.text


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
