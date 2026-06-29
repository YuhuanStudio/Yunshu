"""unit coverage for yunshu_control.audit_log.resolve_actor.

The audit module had no dedicated test file. This covers the 4 resolution
branches enumerated in the docstring.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from yunshu_control.audit_log import (
    _sanitize,
    log_operation,
    resolve_actor,
)


class FakeRBACKeyWithName:
    name = "alice"
    key_prefix = "ak_abcd1234"


class FakeRBACKeyOnlyPrefix:
    name = ""  # Empty string is falsy
    key_prefix = "ak_xyz98765"


class FakeRBACKeyNeither:
    name = ""
    key_prefix = ""


class FakeTenant:
    name = "acme"
    tenant_id = "t-001"


class FakeTenantIDOnly:
    name = ""
    tenant_id = "t-002"


def _make_request(**state_attrs):
    """Build a stub Request with arbitrary `.state` attributes."""
    state = SimpleNamespace(**state_attrs)
    return SimpleNamespace(state=state)


class TestResolveActor:
    def test_rbac_key_with_name_returns_name(self):
        req = _make_request(rbac_key=FakeRBACKeyWithName())
        assert resolve_actor(req) == "alice"

    def test_rbac_key_with_only_prefix_returns_key_prefix(self):
        req = _make_request(rbac_key=FakeRBACKeyOnlyPrefix())
        assert resolve_actor(req) == "key:ak_xyz98765"

    def test_rbac_key_with_neither_returns_rbac_user(self):
        req = _make_request(rbac_key=FakeRBACKeyNeither())
        assert resolve_actor(req) == "rbac_user"

    def test_static_tenant_with_name_returns_tenant_name(self):
        req = _make_request(tenant=FakeTenant())
        assert resolve_actor(req) == "tenant:acme"

    def test_static_tenant_with_only_id_returns_tenant_id(self):
        req = _make_request(tenant=FakeTenantIDOnly())
        assert resolve_actor(req) == "tenant:t-002"

    def test_no_state_attribute_returns_owner_default(self):
        # request without `.state` (e.g. some custom callers) → single-consumer default
        obj = object()
        assert resolve_actor(obj) == "owner"

    def test_none_request_returns_owner_default(self):
        assert resolve_actor(None) == "owner"

    def test_empty_state_returns_owner_default(self):
        req = _make_request()  # state present but no rbac_key/tenant
        assert resolve_actor(req) == "owner"

    def test_rbac_key_takes_precedence_over_tenant(self):
        """Resolution-order rule from the docstring (RBAC before tenant)."""
        req = _make_request(
            rbac_key=FakeRBACKeyWithName(),
            tenant=FakeTenant(),
        )
        assert resolve_actor(req) == "alice"


class TestLogOperation:
    """Smoke-test the structured log emission.

    The `yunshu.audit` logger sets `propagate = False`, so we attach a
    transient handler directly and capture records via a list.
    """

    @staticmethod
    def _attach_capturing_handler():
        logger = logging.getLogger("yunshu.audit")
        records: list[logging.LogRecord] = []

        class _ListHandler(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _ListHandler(level=logging.INFO)
        logger.addHandler(handler)
        return logger, handler, records

    def test_log_operation_emits_audit_line(self):
        logger, handler, records = self._attach_capturing_handler()
        try:
            log_operation(
                "model_load",
                "qwen-7b",
                "success",
                actor="alice",
                detail="ok",
            )
        finally:
            logger.removeHandler(handler)
        assert records, "expected at least one audit record"
        text = " ".join(r.getMessage() for r in records)
        assert "audit:" in text
        assert "op=model_load" in text
        assert "actor=alice" in text
        assert "resource=qwen-7b" in text
        assert "result=success" in text
        assert "detail=ok" in text

    def test_log_operation_defaults_actor_to_system(self):
        logger, handler, records = self._attach_capturing_handler()
        try:
            log_operation("startup", "server")
        finally:
            logger.removeHandler(handler)
        text = " ".join(r.getMessage() for r in records)
        assert "actor=system" in text


class TestSanitize:
    """The audit-line _sanitize replaces whitespace/=/double-quote."""

    def test_replaces_spaces(self):
        assert _sanitize("hello world") == "hello_world"

    def test_replaces_equals(self):
        assert _sanitize("key=value") == "key_value"

    def test_replaces_double_quotes(self):
        assert _sanitize('he said "hi"') == "he_said_'hi'"
