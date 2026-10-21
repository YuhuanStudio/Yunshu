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


def test_disk_inventory_exposes_counts_and_sizes_without_token_ids(tmp_path):
    import json

    header = json.dumps({"__metadata__": {"token_ids": "12345,67890"}}).encode()
    path = tmp_path / "exact_example.safetensors"
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"payload")
    (tmp_path / "exact_torn.safetensors").write_bytes(b"broken")
    rows = probe.disk_inventory(tmp_path)
    assert rows == [{"tokens": 2, "bytes": path.stat().st_size}]
    assert "12345" not in json.dumps(rows)


def test_automatic_rehearsal_proof_requires_completion_and_reply_reuse():
    probe.validate_next_turn(dict(ct=2304, pt=32768), dict(cached=34816), 1)
    for count, cached in ((0, 34816), (1, 32767)):
        with pytest.raises(ValueError):
            probe.validate_next_turn(
                dict(ct=2304, pt=32768), dict(cached=cached), count
            )


def test_eviction_wait_requires_the_measured_checkpoint_before_restart():
    fills = []

    def inventory(root):
        return [{"tokens": 123, "bytes": 8}] if fills else []

    assert probe.ensure_spilled(
        "root", [123], fills.append, wait_s=0, inventory=inventory
    ) == [{"tokens": 123, "bytes": 8}]
    assert fills == [0]
    with pytest.raises(ValueError, match="did not persist"):
        probe.ensure_spilled(
            "root", [999], fills.append, wait_s=0, max_fill=1, inventory=inventory
        )
