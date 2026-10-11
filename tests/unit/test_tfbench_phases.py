import importlib.util
from pathlib import Path

import pytest

s = importlib.util.spec_from_file_location(
    "tb", Path(__file__).parents[2] / "scripts/research/tfbench.py"
)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)


def test_default_runs_everything_and_selection_is_exact():
    assert m.parse_phases(",".join(m.ALL_PHASES)) == set(m.ALL_PHASES)
    assert m.parse_phases("cold,warm") == {"cold", "warm"}


def test_bad_or_orphan_selections_fail_closed():
    for bad in ("", "cold,bogus", "warm,turn2"):
        with pytest.raises(SystemExit):
            m.parse_phases(bad)
