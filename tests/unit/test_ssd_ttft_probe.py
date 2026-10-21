"""CPU-only scripted transport: tier and restart assertions precede costly runs."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "ssd_probe",
    Path(__file__).resolve().parents[2] / "scripts/research/ssd_ttft_probe.py",
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_matrix_restarts_once_and_checks_tiers_and_rehearsal():
    rows, servers = [], []
    tiers = iter(["none", "ram", "none", "ssd", "ram", "ram", "ram"])

    class Server:
        def __init__(self, *args, **kwargs):
            servers.append(self)
            self.url = self.model = "fake"
            self.log = SimpleNamespace(read_text=lambda: "dflash")
            self.killed = 0

        def kill(self):
            self.killed += 1

    def send(url, body):
        tier = next(tiers)
        return dict(
            ct=1,
            ttft_s=0.1,
            cached=0 if tier == "none" else 10,
            pt=11,
            sha="same",
            _text="reply",
            xy={"cache": {"tier": tier}},
        )

    fake = SimpleNamespace(
        Srv=Server,
        send=send,
        req=lambda m, p, t: {"messages": [{"role": "user", "content": p}]},
        load_prompt=lambda p: p,
        engaged_spec_mode=lambda *a: "dflash",
        rehearse=lambda *a: {"rehearsal_usage": {"prompt_tokens": 10}},
    )
    probe.run(fake, {}, "m", "tag", [4096], 1, rows.append, idle=0)
    assert len(servers) == 2 and all(s.killed == 1 for s in servers)
    assert [r["phase"] for r in rows if "ctx" in r] == [
        "cold",
        "ram_before_restart",
        "spill",
        "ssd",
        "ram_after_restart",
        "followup",
        "followup_rehearsed",
    ]
    assert rows[-1] == {"complete": True}


def test_matrix_rejects_false_ssd_hit_and_output_drift():
    row = dict(
        ct=1, ttft_s=0.1, cached=10, pt=11, sha="a", xy={"cache": {"tier": "ram"}}
    )
    with pytest.raises(ValueError, match="expected ssd"):
        probe.validate(row, "ssd")
    with pytest.raises(ValueError, match="output mismatch"):
        probe.validate(row, reference={"sha": "b"})
