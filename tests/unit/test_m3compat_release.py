"""Fail closed when a release HTTP run lacks committed-token evidence."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "m3compat_release", Path(__file__).parents[2] / "scripts/dev/m3compat_release.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_receipts_require_one_complete_nonempty_record_per_request():
    log = 'm3compat_receipt {"ids": [17, 21], "fp_exact_rows": {}}\n'
    assert module.read_receipts(log, 1)["ids"] == [17, 21]
    with pytest.raises(RuntimeError, match="missing or empty"):
        module.read_receipts(log, 2)
    with pytest.raises(RuntimeError, match="missing or empty"):
        module.read_receipts('m3compat_receipt {"ids": []}\n', 1)
    with pytest.raises(RuntimeError, match="missing or empty"):
        module.read_receipts("", 1)
