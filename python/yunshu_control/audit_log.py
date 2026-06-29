from __future__ import annotations

"""Yunshu Audit Logging — structured logging for admin/management operations.

Records who did what, when, on what resource, and the result.  Uses standard
Python ``logging`` at INFO level with a consistent ``audit:`` prefix so that
operators can route audit records to a dedicated file or SIEM via standard
logging configuration (``DictHandler``, ``RotatingFileHandler``, etc.).

No database, no persistence layer — just structured log lines.  This keeps the
implementation simple while satisfying the compliance requirement that every
admin mutation is recorded.

Log format::

    audit: op=<operation> actor=<who> resource=<what> result=<success|failure> [key=value ...]

Single-consumer note: this module is self-contained — it has no dependency on
the (removed) RBAC / multi-tenant stack. ``resolve_actor`` returns a sensible
single-owner identity for the local single-consumer deployment (constant
``"owner"`` when no per-request identity is attached, mirroring the
``current_actor`` value stamped by the simplified static-token auth middleware).
"""

import logging
import os
import sys

# Dedicated logger so operators can route audit records independently of
# the root / yunshu loggers (e.g. to a separate file or syslog).
audit_logger = logging.getLogger("yunshu.audit")


def _attach_default_handler() -> None:
    """Ensure audit records actually emit somewhere.

    Previously `yunshu.audit` had no handler attached, so `log_operation()`
    calls were silently dropped.

    Defaults:
      - `YUNSHU_AUDIT_LOG_FILE=<path>` → append JSONL-ish lines to that path
      - Else if a parent `yunshu` logger has handlers → propagate (no add)
      - Else → emit to stderr at INFO level
    """
    if audit_logger.handlers:
        return  # operator-attached; leave alone
    parent = logging.getLogger("yunshu")
    if parent.handlers:
        audit_logger.setLevel(logging.INFO)
        return  # propagation handles emission
    log_path = os.environ.get("YUNSHU_AUDIT_LOG_FILE", "").strip()
    if log_path:
        handler: logging.Handler = logging.FileHandler(log_path, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s [AUDIT] %(message)s"))
    audit_logger.addHandler(handler)
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False


_attach_default_handler()


def log_operation(
    operation: str,
    resource: str,
    result: str = "success",
    *,
    actor: str | None = None,
    **extra: object,
) -> None:
    """Emit a structured audit log line.

    Parameters
    ----------
    operation:
        Short verb describing the action, e.g. ``model_load``, ``lora_unload``,
        ``rbac_key_create``, ``batch_create``.
    resource:
        Target of the operation, e.g. model name, adapter ID, key name.
    result:
        ``"success"`` or ``"failure"``.
    actor:
        Identity of the caller.  When called from a gateway request handler,
        pass ``resolve_actor(request)`` to extract the per-request identity
        (single-consumer default: ``"owner"``).  Defaults to ``"system"`` for
        programmatic callers (startup, shutdown).
    **extra:
        Arbitrary key=value pairs appended to the log line for additional
        context (e.g. ``detail="..."``, ``count=5``).
    """
    actor = actor or "system"
    # Sanitize EVERY field (operation/actor/resource carry attacker-influenced
    # values — model names, key names, etc.). _sanitize strips newlines/CR/tabs
    # so a crafted value can't forge extra audit lines.
    parts = [
        f"op={_sanitize(str(operation))}",
        f"actor={_sanitize(str(actor))}",
        f"resource={_sanitize(str(resource))}",
        f"result={_sanitize(str(result))}",
    ]
    for k, v in extra.items():
        # Sanitize: replace whitespace so each key=value is a single token
        parts.append(f"{k}={_sanitize(str(v))}")
    audit_logger.info("audit: %s", " ".join(parts))


def resolve_actor(request: object) -> str:
    """Resolve the actor identity for an audit line.

    Single-consumer model: the multi-tenant RBAC machinery is gone, so every
    request maps to the constant single-owner identity (``"owner"``, overridable
    via ``YUNSHU_ACTOR_IDENTITY``) — the single digital being (Yunmo) this engine
    serves. ``request`` is accepted for call-site compatibility.
    """
    return _default_actor()


def _default_actor() -> str:
    """Single-consumer default actor identity.

    Overridable via ``YUNSHU_ACTOR_IDENTITY`` for operators who want a custom
    label in audit lines; defaults to ``"owner"``.
    """
    custom = os.environ.get("YUNSHU_ACTOR_IDENTITY", "").strip()
    return custom or "owner"


def _sanitize(value: str) -> str:
    """Replace characters that would break structured log parsing or forge lines.

    Strips newlines/CR/tabs (log-injection vector: a crafted value containing
    ``\\n audit: op=...`` could otherwise forge a second audit record) and
    collapses spaces/``=``/quotes so each ``key=value`` stays a single token.
    """
    return (
        value.replace("\r", "_")
        .replace("\n", "_")
        .replace("\t", "_")
        .replace(" ", "_")
        .replace("=", "_")
        .replace('"', "'")
    )
