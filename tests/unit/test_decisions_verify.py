"""The real-model verifier's check logic, driven structurally against a fake engine."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
import decisions_verify as dv  # noqa: E402

from yunshu_engine.decision_engine import DecisionEngine, DecisionResult  # noqa: E402


class IdWeightedEngine(DecisionEngine):
    """Probabilities depend only on the option ids, like the real head (order invariant)."""

    def __init__(self):
        super().__init__("/nonexistent/fake")
        self._loaded = True
        self.supports_multimodal = True

    async def decide(self, req):
        out = []
        for q in req.questions:
            if q.kind == "predicate":
                out.append([0.7, 0.3])
                continue
            if q.kind == "score":
                w = [i + 1.0 for i in range(len(q.options))]
            else:
                w = [float(sum(map(ord, o.id)) % 7 + 1) for o in q.options]
            out.append([x / sum(w) for x in w])
        return DecisionResult(out, 100)


def test_structural_checks_pass_against_a_fake_engine(monkeypatch):
    import openai

    from yunshu_gateway.main import create_app
    from yunshu_gateway.routers import decisions

    engine = IdWeightedEngine()

    async def resolve(_):
        return engine

    monkeypatch.setattr(decisions, "_resolve_decision_engine", resolve)
    app = create_app()
    # sync clients over an ASGI app: run the app through httpx's async transport in a thread
    with anyio_portal() as portal:
        http = SyncASGI(app, portal)
        oa = openai.OpenAI(api_key="k", base_url="http://t/v1", http_client=http)
        out = dv.run_checks(oa, http, "fake", semantic=False)
    assert out["order_invariance"] and out["determinism"] == "identical"
    assert out["bool_choice"]["choice"] in (True, False)


def test_expect_fails_closed():
    with pytest.raises(dv.CheckError):
        dv.expect(False, "x")


# a minimal sync httpx client over an ASGI app (httpx has no sync ASGI transport)
class anyio_portal:  # noqa: N801
    def __enter__(self):
        import anyio.from_thread

        self._cm = anyio.from_thread.start_blocking_portal()
        return self._cm.__enter__()

    def __exit__(self, *a):
        return self._cm.__exit__(*a)


class _Transport(httpx.BaseTransport):
    def __init__(self, app, portal):
        self.t = httpx.ASGITransport(app=app)
        self.portal = portal

    def handle_request(self, request):
        async def go():
            req = httpx.Request(
                request.method,
                request.url,
                headers=request.headers,
                content=request.read(),
            )
            resp = await self.t.handle_async_request(req)
            body = b"".join([c async for c in resp.stream])
            return httpx.Response(
                resp.status_code, headers=resp.headers, content=body, request=request
            )

        return self.portal.call(go)


class SyncASGI(httpx.Client):
    def __init__(self, app, portal):
        super().__init__(transport=_Transport(app, portal), base_url="http://t")
