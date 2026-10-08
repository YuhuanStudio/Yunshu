"""Report the constraint actually installed, independent of output JSON validity."""

from __future__ import annotations


def report(engine, requested, backend=None, *, enforced=False):
    from .request_tracker import current_request_info

    info = current_request_info.get()
    value = {
        "requested": requested is not None,
        "enforced": bool(enforced),
        "engine": engine,
        "grammar_backend": backend,
        "reason": "enforced"
        if enforced
        else ("setup_failed" if requested is not None else "not_requested"),
    }
    if info is not None:
        info.structured_output = value
    return value


def backend_name(value):
    return type(value).__name__ if value is not None else None
