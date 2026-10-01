"""A prompt whose essentials exceed the context window is a 400, not a 500."""

from fastapi.testclient import TestClient

from yunshu_engine.context_window import ContextBudgetError


def _app():
    from yunshu_gateway.main import create_app

    app = create_app()

    @app.post("/v1/chat/completions-budget-probe")
    async def chat_probe():
        raise ContextBudgetError("Prompt needs at least 9000 tokens")

    return app


def test_openai_shape():
    r = TestClient(_app(), raise_server_exceptions=False).post(
        "/v1/chat/completions-budget-probe"
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "context_length_exceeded"
