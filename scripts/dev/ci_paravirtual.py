"""Pytest plugin for exercising VM fallbacks on a physical Mac.

Load with ``pytest -p scripts.dev.ci_paravirtual``. Tests may explicitly override
this fixture to exercise a physical path; hardware identification itself remains
unpatched so its device-name/architecture cases still test the real detector.
This models capability selection, not VM performance or Metal implementation.
"""

from functools import lru_cache

import pytest


@pytest.fixture(autouse=True)
def force_paravirtual_metal(monkeypatch, request):
    if request.node.name.startswith("test_identification_requires_both_vm_identifiers"):
        return
    from yunshu_engine.utils import hardware

    monkeypatch.setattr(hardware, "is_paravirtual_metal", lru_cache()(lambda: True))
