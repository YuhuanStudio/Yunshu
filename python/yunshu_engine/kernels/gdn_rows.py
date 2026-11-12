"""Moved to ``yunshu_engine.blocks.recurrent`` (M3). This name is the same module object (sys.modules alias)."""

import sys

from ..blocks import recurrent as _block

sys.modules[__name__] = _block
