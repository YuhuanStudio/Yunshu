"""Shared engine types — extracted from engine.py for cross-module use.

EngineConfig and RequestPhase are used by multiple engine modules
(vlm_engine, audio_engine, image_engine, model_manager) without
needing the full legacy Engine class.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class RequestPhase(Enum):
    WAITING = auto()
    PREFILLING = auto()
    DECODING = auto()
    FINISHED = auto()


@dataclass
class EngineConfig:
    """Engine tuning parameters.

    Maps to BatchGenerator constructor + oMLX's SchedulerConfig.
    """

    completion_batch_size: int = 32
    prefill_batch_size: int = 8
    prefill_step_size: int = 2048
    max_kv_size: Optional[int] = None
    step_interval_ms: float = 1.0
    deferred_clear_delay: int = 8
    cache_cleanup_interval: int = 512
