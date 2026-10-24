"""CPU check of scripts/research/defaultfix_real.py helpers (no server, no model)."""

import importlib.util
import json
import pathlib

import pytest

p = pathlib.Path(__file__).parents[2] / "scripts/research/defaultfix_real.py"
spec = importlib.util.spec_from_file_location("defaultfix_real", p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_pad_body_hits_the_target_size_and_stays_json():
    raw = mod.pad_body({"model": "m"}, 1)
    assert abs(len(raw) - 1024 * 1024) < 64
    assert json.loads(raw)["model"] == "m"


def test_expect_raises_fail():
    with pytest.raises(mod.Fail):
        mod.expect(False, "boom")


def test_wait_ready_fails_fast_when_server_exited():
    class P:
        returncode = 3

        def poll(self):
            return 3

    with pytest.raises(mod.Fail):
        mod.wait_ready("http://127.0.0.1:1", P(), secs=5)
