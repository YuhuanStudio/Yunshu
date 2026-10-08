import importlib.util
import pathlib

import pytest

spec = importlib.util.spec_from_file_location(
    "tfbench", pathlib.Path(__file__).parents[2] / "scripts/research/tfbench.py"
)
tfbench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tfbench)


def test_rehearsal_prefix_cuts_before_last_user():
    r = "<|im_start|>user\nA<|im_end|>\n<|im_start|>assistant\nB<|im_end|>\n<|im_start|>user\nC<|im_end|>\n"
    assert tfbench.rehearsal_prefix(r).endswith("B<|im_end|>\n")
    assert "C" not in tfbench.rehearsal_prefix(r)


def test_rehearsal_prefix_missing_marker_fails_closed():
    with pytest.raises(ValueError):
        tfbench.rehearsal_prefix("no user header here")
