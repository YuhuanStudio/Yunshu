"""Every yv stage runs in some suite, or is on the explicit known-gap list (HARNESS row 72)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

# Stages no suite runs yet; a new stage must be added to a suite or listed here on purpose.
KNOWN_UNSUITED = {
    "deployment",
    "deployment-idle",
    "service_deploy",
    "toolparse",
    "modelprobe",
    "lane_bandwidth",
    "jump_forward",
    "websearch",
    "tavily",
    "searchrank",
}


def test_every_stage_belongs_to_a_suite_or_is_known_unsuited():
    from verify.suites import STAGES, SUITES

    covered = {s for v in SUITES.values() for s in v.get("stages", [])}
    unlisted = [s for s in STAGES if s not in covered and s not in KNOWN_UNSUITED]
    assert unlisted == []
    assert set(STAGES) >= KNOWN_UNSUITED
