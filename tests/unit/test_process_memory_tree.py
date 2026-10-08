import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
import process_memory as pm


def test_descendants_include_worker_children_and_grandchildren():
    rows = [(1, 0, 10), (10, 1, 5), (11, 10, 5), (12, 11, 5), (99, 0, 5)]
    assert pm.descendants(rows, 10) == {
        10,
        11,
        12,
    }  # a single-PID sampler sees only {10}
    assert pm.descendants(rows, 1) == {1, 10, 11, 12}


def test_tree_sum_counts_worker_process(monkeypatch):
    class Info:
        def __init__(self):
            self.phys_footprint = 0
            self.resident_size = 0
            self.wired_size = 0

    footprints = {10: 1 << 30, 11: 15 << 30}
    monkeypatch.setattr(
        pm.subprocess,
        "check_output",
        lambda *a, **k: "10 1 100\n11 10 100\n99 1 100\n",
    )

    class Fn:
        argtypes = restype = None

        def __call__(self, pid, flavor, ref):
            ref._obj.phys_footprint = footprints[pid]
            return 0

    class Lib:
        proc_pid_rusage = Fn()

    monkeypatch.setattr(pm.ctypes, "CDLL", lambda *a, **k: Lib())
    out = pm.process_tree_memory(10)
    assert out["physical_footprint_sum_bytes"] == 16 << 30


def test_system_used_bytes_from_vm_stat():
    text = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages active:    100.\nPages wired down:    50.\n"
        "Pages occupied by compressor:    10.\n"
    )
    assert pm.system_used_bytes(text) == 160 * 16384
