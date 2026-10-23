"""CPU check of scripts/research/anth1021_real.py helpers (no server, no model)."""

import importlib.util
import pathlib

import pytest

p = pathlib.Path(__file__).parents[2] / "scripts/research/anth1021_real.py"
spec = importlib.util.spec_from_file_location("anth1021_real", p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_expect_raises_fail():
    mod.expect(True, "x")
    with pytest.raises(mod.Fail):
        mod.expect(False, "boom")


def test_wait_ready_fails_fast_when_server_exited():
    class P:
        returncode = 3

        def poll(self):
            return 3

    with pytest.raises(mod.Fail):
        mod.wait_ready("http://127.0.0.1:1", P(), secs=5)
