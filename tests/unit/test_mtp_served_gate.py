"""The real-model gate must reject unengaged MTP and changed token IDs."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location(
    "verify_mtp_spec",
    Path(__file__).resolve().parents[2] / "scripts/verify/verify_mtp_spec.py",
)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


class Engine:
    def __init__(self, *, mismatch=False, incomplete=False):
        self._batch_runner = SimpleNamespace(drafter=object(), draft_kind="mtp")
        self.mismatch = mismatch
        self.incomplete = incomplete

    async def generate_stream(self, **_):
        ids = [1, 3] if self.mismatch and self._batch_runner.drafter else [1, 2]
        yield SimpleNamespace(
            new_token_ids=ids, finished=not self.incomplete, error=None
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch,expected", [(False, True), (True, False)])
async def test_gate_checks_token_parity_and_restores_drafter(mismatch, expected):
    engine = Engine(mismatch=mismatch)
    drafter = engine._batch_runner.drafter
    assert await gate.verify(engine) is expected
    assert engine._batch_runner.drafter is drafter


@pytest.mark.asyncio
async def test_gate_fails_closed_without_mtp():
    engine = Engine()
    engine._batch_runner.draft_kind = "dflash"
    with pytest.raises(RuntimeError, match="MTP is not engaged"):
        await gate.verify(engine)


@pytest.mark.asyncio
async def test_gate_fails_closed_on_incomplete_stream():
    engine = Engine(incomplete=True)
    drafter = engine._batch_runner.drafter
    with pytest.raises(RuntimeError, match="incomplete"):
        await gate.verify(engine)
    assert engine._batch_runner.drafter is drafter
