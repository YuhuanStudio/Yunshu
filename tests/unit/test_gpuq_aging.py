"""A p1/p0 job must not be starved for hours by a stream of fresh p1 jobs."""

import importlib.util
import sys
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "gpuq", Path(__file__).resolve().parents[2] / "scripts" / "dev" / "gpuq.py"
)
sys.path.insert(0, str(Path(spec.origin).parent))
gpuq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpuq)


def job(jid, age_s, prio, owner, **kw):
    return {
        "id": jid,
        "state": "pending",
        "priority": prio,
        "submitted": time.time() - age_s,
        "label": jid,
        "cwd": "/x",
        "env": {"GPUQ_OWNER": owner},
        "timeout_s": 1560.0,
        **kw,
    }


def test_old_p1_beats_fresh_p1_stream_of_other_owner():
    now = time.time()
    ran = job(
        "r", 9000, 1, "b", state="done", started=now - 5, ended=now - 1, timeout_s=300.0
    )
    old = job("old", 5000, 1, "a")
    ran2 = job(
        "r2",
        9000,
        1,
        "a",
        state="done",
        started=now - 2,
        ended=now - 1,
        timeout_s=300.0,
    )
    fresh = [job(f"f{i}", 10, 1, "b") for i in range(3)]
    # owner a ran most recently, so round-robin alone would keep "old" behind b's stream
    assert gpuq._pick([ran, ran2, old, *fresh])["id"] == "old"


def test_old_p0_ties_p1_instead_of_waiting_forever():
    assert gpuq._eff_priority(job("gate0", 3000, 0, "c")) == 1
    assert gpuq._eff_priority(job("fresh", 10, 1, "d")) == 1
    assert gpuq._eff_priority(job("young", 10, 0, "c")) == 0
