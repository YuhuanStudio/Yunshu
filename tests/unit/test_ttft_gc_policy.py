"""CPU checks for diagnostic-only index GC suppression."""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "ttft_gc_policy",
    Path(__file__).resolve().parents[2]
    / "scripts/research/ttft_probe_site/gc_policy.py",
)
policy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(policy)


class Collector:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.transitions = []

    def isenabled(self):
        return self.enabled

    def disable(self):
        self.enabled = False
        self.transitions.append("disable")

    def enable(self):
        self.enabled = True
        self.transitions.append("enable")


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("fails", [True, False])
def test_index_gc_preserves_state_on_success_and_failure(enabled, fails):
    collector = Collector(enabled)

    def build(value):
        assert not collector.isenabled()
        if fails:
            raise RuntimeError("index failed")
        return value

    wrapped = policy.defer_index_gc(build, collector)
    if fails:
        with pytest.raises(RuntimeError, match="index failed"):
            wrapped(42)
    else:
        assert wrapped(42) == 42
    assert collector.isenabled() is enabled
    assert collector.transitions == (["disable", "enable"] if enabled else [])


def test_nested_index_gc_only_restores_outer_enabled_state():
    collector = Collector()

    def inner():
        assert not collector.isenabled()
        return 42

    nested = policy.defer_index_gc(inner, collector)

    def outer():
        value = nested()
        assert not collector.isenabled()
        return value

    assert policy.defer_index_gc(outer, collector)() == 42
    assert collector.transitions == ["disable", "enable"]


def test_index_gc_preserves_incremental_copy_proposals():
    from yunshu_engine.copy_drafter import CopyDrafter

    baseline, candidate = CopyDrafter(), CopyDrafter()
    candidate._sync_index = policy.defer_index_gc(candidate._sync_index)
    prompt = list(range(32)) * 1024 + list(range(8))
    baseline.extend(prompt)
    candidate.extend(prompt)
    for token in range(8, 40):
        assert baseline.lookup() == candidate.lookup()
        assert baseline.last_match == candidate.last_match
        assert baseline._index == candidate._index
        baseline.extend([token])
        candidate.extend([token])


def test_probe_process_installs_requested_policy_without_loading_mlx(tmp_path):
    import json
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    site = root / "scripts/research/ttft_probe_site"
    env = dict(
        os.environ,
        PYTHONPATH=f"{site}:{root / 'python'}",
        TTFT_PROBE_OUT=str(tmp_path / "samples"),
        TTFT_PROBE_COPY_INDEX_GC="defer",
        TTFT_PROBE_CALLS="0",
        TTFT_PROBE_STACKS="0",
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import gc,sys; from yunshu_engine.copy_drafter import CopyDrafter; assert hasattr(CopyDrafter._sync_index, '__wrapped__'); c=CopyDrafter(); c.extend([1,2,3,4]); c._sync_index(); assert gc.isenabled(); assert 'mlx.core' not in sys.modules",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert child.returncode == 0, child.stderr
    runtime = list(tmp_path.glob("runtime.*.json"))
    assert runtime and json.loads(runtime[0].read_text())["copy_index_gc"] == "defer"
    assert "policy engaged: defer" in child.stderr
