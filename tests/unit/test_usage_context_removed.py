"""The tenant token-billing box had no producer (new_billing_box: zero callers), so the
record_billed_tokens calls in the routers were permanent no-ops. Keep it deleted."""

import importlib
import pathlib

import pytest


def test_usage_context_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("yunshu_gateway.usage_context")


def test_routers_do_not_reference_it():
    root = pathlib.Path(__file__).resolve().parents[2] / "python" / "yunshu_gateway"
    for p in root.rglob("*.py"):
        assert "record_billed_tokens" not in p.read_text(), p
