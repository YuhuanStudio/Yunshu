"""Waves 933-934: error-contract consistency fixes.

W933: the legacy-tenant auth error formatter mapped any non-401 Anthropic-path status to
  invalid_request_error, so a 429 rate-limit denial on /v1/messages returned
  invalid_request_error instead of Anthropic's mandated rate_limit_error (403→permission_error
  likewise).
W934: logit_bias finite/range validation raised 422 on the Responses + Anthropic routers but
  400 on chat — OpenAI returns 400 for invalid params (SDKs treat 422 as a distinct
  UnprocessableEntityError). Unified to 400.
"""
from __future__ import annotations

import inspect


def test_w933_anthropic_error_type_map():
    from yunshu_gateway.middleware import tenant_auth
    src = inspect.getsource(tenant_auth._ErrorFormatter.auth_error)
    assert '429: "rate_limit_error"' in src
    assert '403: "permission_error"' in src
    assert '401: "authentication_error"' in src


def test_w934_logit_bias_uses_400_not_422():
    from yunshu_gateway.routers import anthropic, responses
    for mod in (responses, anthropic):
        src = inspect.getsource(mod)
        # no logit_bias validation still raises 422
        assert "status_code=422" not in src or "logit_bias" not in src
    # responses + anthropic now use 400 for the logit_bias finite/range checks
    rsrc = inspect.getsource(responses)
    asrc = inspect.getsource(anthropic)
    assert "logit_bias" in rsrc and "status_code=400" in rsrc
    assert 'HTTPException(status_code=400, detail=f"logit_bias' in asrc
