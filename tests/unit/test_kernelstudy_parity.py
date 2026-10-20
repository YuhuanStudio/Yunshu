"""CPU contracts for fail-closed lossless rejudgement."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "kernelstudy_parity",
    Path(__file__).parents[2] / "scripts/research/kernelstudy_parity.py",
)
parity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(parity)


def test_canonical_windows_preserve_absolute_decode_positions():
    for end in (32768, 65536, 131072):
        full = parity.canonical_lengths(end, 7)
        for count in range(1, 8):
            assert parity.canonical_lengths(end, count) == full[-count:]


@pytest.mark.parametrize("end,count", [(1, 7), (32, 0), (32, 8)])
def test_invalid_window_rejected(end, count):
    with pytest.raises(ValueError):
        parity.canonical_lengths(end, count)


def rows():
    return [
        dict(
            context=n,
            T=t,
            ours={"within_existing_atol_4e_3": True},
            external={"within_existing_atol_4e_3": True},
            deterministic=True,
            verify_equals_decode=True,
            cache_tensor_hit_equals_miss=True,
        )
        for n in (32768, 65536, 131072)
        for t in range(1, 8)
    ]


def test_missing_duplicate_or_failed_cell_cannot_pass():
    cells = rows()
    assert parity.validate(cells)
    with pytest.raises(ValueError):
        parity.validate(cells[:-1])
    with pytest.raises(ValueError):
        parity.validate(cells[:-1] + [cells[0]])
    cells[-1]["verify_equals_decode"] = False
    assert not parity.validate(cells)
