"""The release gate's memory-returns check subtracts what the server keeps on purpose: prefix-cache
checkpoints in RAM and the MLX allocator's bounded reuse pool (~6 GiB on 128 GiB)."""

import importlib.util
import io
import urllib.request
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "process_memory",
    Path(__file__).resolve().parents[2] / "scripts/research/process_memory.py",
)
pm = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pm)

METRICS = (
    "# HELP yunshu_apc_resident_bytes x\n"
    f"yunshu_apc_resident_bytes {2 * 2**30}\n"
    f'yunshu_gpu_memory_bytes{{type="active"}} {20 * 2**30}\n'
    f'yunshu_gpu_memory_bytes{{type="cache"}} {int(5.5 * 2**30)}\n'
)


def test_retained_adds_allocator_pool_to_prefix_cache(monkeypatch):
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: io.BytesIO(METRICS.encode())
    )
    assert pm.allocator_pool_gib("http://x") == 5.5
    assert pm.retained_gib("http://x") == 7.5


def test_retained_is_unknown_without_the_prefix_cache_figure(monkeypatch):
    text = 'yunshu_gpu_memory_bytes{type="cache"} 1024\n'
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *a, **k: io.BytesIO(text.encode())
    )
    assert pm.retained_gib("http://x") is None
