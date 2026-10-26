"""Every lazy export of ``yunshu_engine`` resolves.

The table kept names from deleted modules (.checkpoint, .request_dedup) and a class
request.py no longer defines, so ``from yunshu_engine import X`` failed only when used.
"""

from __future__ import annotations

import importlib
import inspect
import re

import yunshu_engine


def test_every_lazy_export_resolves():
    pairs = re.findall(r'"(\w+)": "(\.\w+)"', inspect.getsource(yunshu_engine))
    assert len(pairs) > 50
    broken = []
    for name, module in pairs:
        try:
            getattr(importlib.import_module(module, "yunshu_engine"), name)
        except (ImportError, AttributeError) as exc:
            broken.append((name, module, type(exc).__name__))
    assert broken == []
