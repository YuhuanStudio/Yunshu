"""Gated improvement loop: candidate commit -> yv gate -> replay A/B -> canary -> promote | roll back.

Every step is logged (JSONL) before and after it runs. Any stage that is not an explicit pass, raises, or
returns incomplete evidence rolls the candidate back. Stages are injected so the loop is testable with fakes.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

STAGES = ("yv_gate", "replay_ab", "canary")


@dataclass
class Result:
    candidate: str
    outcome: str  # promoted | rolled_back
    failed_stage: str | None
    detail: str


class Loop:
    def __init__(
        self,
        log: Path,
        stages: dict[str, Callable[[str], dict]],
        promote: Callable[[str], None],
        rollback: Callable[[str], None],
    ):
        if set(stages) != set(STAGES):
            raise ValueError(f"stages must be exactly {STAGES}")
        self.log, self.stages, self._promote, self._rollback = (
            log,
            stages,
            promote,
            rollback,
        )

    def _w(self, **kw) -> None:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("a") as f:
            f.write(json.dumps({"t": time.time(), **kw}) + "\n")

    def run(self, candidate: str) -> Result:
        self._w(candidate=candidate, event="start")
        for name in STAGES:
            self._w(candidate=candidate, event="stage_begin", stage=name)
            try:
                r = self.stages[name](candidate)
                ok = (
                    isinstance(r, dict)
                    and r.get("pass") is True
                    and r.get("complete") is True
                )
                detail = (
                    str(r.get("reason", ""))
                    if isinstance(r, dict)
                    else "non-dict result"
                )
            except Exception as e:  # any crash is a fail
                ok, detail = False, f"{type(e).__name__}: {e}"
            self._w(
                candidate=candidate, event="stage_end", stage=name, ok=ok, detail=detail
            )
            if not ok:
                self._rollback(candidate)
                self._w(candidate=candidate, event="rolled_back", stage=name)
                return Result(
                    candidate, "rolled_back", name, detail or "stage did not pass"
                )
        self._promote(candidate)
        self._w(candidate=candidate, event="promoted")
        return Result(candidate, "promoted", None, "")
