"""Catch custom-header and dry-run errors before consuming a GPU queue slot."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


def test_dry_run_does_not_import_mlx(tmp_path):
    probe = ROOT / "scripts/research/nax_prefill_probe.py"
    code = (
        "import runpy, sys; "
        f"sys.argv = [{str(probe)!r}, '--mode', 'micro', '--tiny', "
        f"'--out', {str(tmp_path / 'unused.jsonl')!r}, '--dry-run']; "
        f"runpy.run_path({str(probe)!r}, run_name='__main__'); "
        "assert 'mlx.core' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "unused.jsonl").exists()


def test_header_keeps_buffer_address_spaces_and_omits_builtin_utils():
    pytest.importorskip("mlx.core")
    path = ROOT / "scripts/research/nax_qmm_tiles.py"
    spec = importlib.util.spec_from_file_location("nax_qmm_header_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    header = module.header()
    assert "struct Limits" not in header
    assert re.search(r"const int [KNM]\s*\[\[buffer", header) is None
    assert "const constant int& K [[buffer" in header
    assert "const int K," in header
