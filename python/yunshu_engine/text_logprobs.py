"""Moved to ``yunshu_engine.blocks.head`` (M3). This name is the same module object (sys.modules alias)."""

import sys

from .blocks import head as _block

sys.modules[__name__] = _block
