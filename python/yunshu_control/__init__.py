"""Yunshu Control — usage accounting + audit logging.

Refocused to a single-consumer local omni engine: the multi-tenant RBAC,
per-tenant, and request-queue machinery has been removed. What remains:

- ``audit_log``   — structured audit logger + ``resolve_actor`` (single-owner default)
- ``token_counter`` — legit per-modality usage counting consumed by the
  chat / anthropic / responses routers.

Auth is a single optional bearer-token gate (``YUNSHU_AUTH_TOKEN``) wired in
``yunshu_gateway.main``.
"""

from .audit_log import audit_logger, log_operation, resolve_actor

__all__ = [
    "audit_logger",
    "log_operation",
    "resolve_actor",
]
