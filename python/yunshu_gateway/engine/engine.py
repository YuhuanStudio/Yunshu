"""Legacy engine module — replaced by BatchedEngine.

This file is kept for import compatibility. All engine logic now lives in
the L4 engine module (yunshu_engine.batched_engine).
"""

from yunshu_engine.batched_engine import BatchedEngine as Engine
from yunshu_engine.engine import EngineConfig

__all__ = ["Engine", "EngineConfig"]
