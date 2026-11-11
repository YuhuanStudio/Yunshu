"""Importing yunshu_kv (or its disk_budget) must not load mlx, numpy or the heavy KV modules."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

HEAVY = [
    "mlx",
    "mlx.core",
    "numpy",
    "xxhash",
    "yunshu_kv.tiered",
    "yunshu_kv.manager",
    "yunshu_kv.serialization",
    "yunshu_kv.hash",
    "yunshu_kv.compression",
    "yunshu_kv.warm_tier",
    "yunshu_kv.radix_attention",
    "yunshu_kv.ssd_sqlite_store",
]


def _run(code: str) -> subprocess.CompletedProcess:
    env = {"PYTHONPATH": str(ROOT / "python"), "PATH": "/usr/bin:/bin"}
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


@pytest.mark.parametrize("target", ["yunshu_kv", "yunshu_kv.disk_budget"])
def test_import_does_not_load_heavy_modules(target):
    code = (
        f"import sys, {target}; "
        f"loaded = [m for m in {HEAVY!r} if m in sys.modules]; "
        "assert not loaded, loaded"
    )
    child = _run(code)
    assert child.returncode == 0, child.stderr


def test_lazy_export_resolves_on_first_access():
    code = (
        "import sys, yunshu_kv; "
        "assert 'yunshu_kv.manager' not in sys.modules; "
        "from yunshu_kv.manager import KVCacheManager; "
        "assert yunshu_kv.KVCacheManager is KVCacheManager"
    )
    child = _run(code)
    assert child.returncode == 0, child.stderr
