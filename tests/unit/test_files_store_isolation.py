"""An unfinished batch one test leaves behind is never resumed by a later test's gateway lifespan."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_default_files_store_is_this_tests_own_and_empty():
    from yunshu_gateway import files_store
    from yunshu_gateway.routers import batches

    store = files_store.get_store()
    assert "files-store" in str(store.root)
    assert batches.BatchRunner.next_batch(store) is None


def test_a_batch_left_by_the_route_checks_does_not_reach_the_next_lifespan():
    """The pair that failed test_custom_route_does_not_drop_tool (and, as an anthropic request hung in the tracker,
    faults/test_queue) when they shared an xdist worker."""
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
            "tests/unit/test_route_checks_harness.py::test_check_has_no_harness_bug[batches]",
            "tests/unit/test_agent_client_compat.py::test_custom_route_does_not_drop_tool",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert r.returncode == 0, r.stdout[-2500:]
