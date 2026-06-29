"""Auth contract for `_check_permission`: a local drop-in server serves
inference out of the box, while privileged ops stay deny-by-default — and a
configured token locks everything. This guards the startup banner's promise
('Inference endpoints remain accessible without auth') against regressions."""

import pytest
from fastapi import HTTPException

from yunshu_gateway.routers.models import _check_permission


class _Req:
    """Minimal stand-in — `_check_permission` only reads the Authorization header."""

    def __init__(self, authorization: str | None = None) -> None:
        self.headers = {"Authorization": authorization} if authorization else {}


@pytest.fixture(autouse=True)
def _clean_auth_env(monkeypatch):
    monkeypatch.delenv("YUNSHU_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("YUNSHU_AUTH_DISABLED", raising=False)


def test_inference_open_when_no_auth_configured():
    # No token, not disabled → inference must be served (drop-in local server).
    _check_permission(_Req(), "can_infer")  # must not raise


@pytest.mark.parametrize("perm", ["can_load_models", "can_unload_models"])
def test_privileged_denied_when_no_auth_configured(perm):
    with pytest.raises(HTTPException) as ei:
        _check_permission(_Req(), perm)
    assert ei.value.status_code == 401


def test_token_set_locks_inference_too(monkeypatch):
    monkeypatch.setenv("YUNSHU_AUTH_TOKEN", "secret")
    with pytest.raises(HTTPException) as ei:
        _check_permission(_Req(), "can_infer")  # no/!wrong header
    assert ei.value.status_code == 401
    with pytest.raises(HTTPException):
        _check_permission(_Req("Bearer wrong"), "can_infer")
    # Correct token → allowed
    _check_permission(_Req("Bearer secret"), "can_infer")


def test_disabled_allows_everything(monkeypatch):
    monkeypatch.setenv("YUNSHU_AUTH_DISABLED", "true")
    _check_permission(_Req(), "can_infer")
    _check_permission(_Req(), "can_load_models")
