"""The release gate must record a FAIL when a client step exits non-zero."""

import re
import subprocess
from pathlib import Path

GATE = Path(__file__).resolve().parents[2] / "scripts/release/gate.sh"


def test_no_bare_client_invocations():
    assert not re.findall(
        r"^\s*\$\{CLIENT\[@\]\} --prefix [a-z]", GATE.read_text(), re.M
    )


def test_failed_client_step_records_fail():
    m = re.search(r"^client\(\)\{.*?^\}", GATE.read_text(), re.M | re.S)
    assert m, "client() wrapper missing"
    script = (
        "CLIENT=(false)\n"
        'rec(){ echo "REC $*"; }\n'
        + m.group(0)
        + "\nclient families.x. family --url u\n"
    )
    out = subprocess.run(["zsh", "-c", script], capture_output=True, text=True)
    assert "REC families.x.client FAIL" in out.stdout
