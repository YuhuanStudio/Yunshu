"""M3 forwards and M5 gpuq servers run at the same time, so their local ports must not overlap."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(rel, name):
    sys.path.insert(0, str((ROOT / rel).parent))
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_agentcompat_forwards_stay_outside_the_m5_range():
    ac = _load("scripts/research/agent_compat/agentcompat.py", "agentcompat_ports")
    sys.path.insert(0, str(ROOT / "scripts"))
    from verify import stages

    m5 = set(range(18990, int(stages.PORT_LAST) + 1))
    local = {ac.LOCAL_SERVER_PORT, ac.LOCAL_CONTROL_PORT, ac.MCP_PORT}
    assert not local & m5
    assert all(18990 <= p <= 18999 for p in local)
