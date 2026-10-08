"""The gate's long soaks must write progress to the job log: gpuq stops a job whose log
stops growing (stall_s), and soak-mmlu (300 questions, ~30 min) printed only to a side file,
so the 0.1.4 gate stopped it as "stalled" after 20 minutes while it was answering."""

import re
from pathlib import Path

GATE = Path(__file__).resolve().parents[2] / "scripts/release/gate.sh"


def test_soak_scripts_tee_their_progress_into_the_job_log():
    text = GATE.read_text()
    for script in ("soak_mmlu_pro.py", "soak_realistic.py"):
        call = re.search(rf"{script}.*?(?=\n\s*s=\$\()", text, re.S)
        assert call, script
        assert "| tee " in call.group(0), f"{script} output must stream to the job log"
