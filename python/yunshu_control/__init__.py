"""Yunshu Control — usage accounting + audit logging.

Single-node engine: there is no tenant or RBAC layer (managed API keys and
their quotas live in the gateway). What remains here:

- ``audit_log``   — structured audit logger + ``resolve_actor`` (single-owner default)
- ``token_counter`` — legit per-modality usage counting consumed by the
  chat / anthropic / responses routers.

Auth (static ``YUNSHU_AUTH_TOKEN`` or managed API keys) is the gateway's
``AuthMiddleware``, wired in ``yunshu_gateway.main``.
"""

from .audit_log import audit_logger, log_operation, resolve_actor

__all__ = [
    "audit_logger",
    "log_operation",
    "resolve_actor",
]
