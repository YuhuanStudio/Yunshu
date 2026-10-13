"""CPU fixtures for the single-reservation concurrency stage."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
from verify.conc_pair import run_pair  # noqa: E402


@pytest.mark.parametrize("fault", [None, "short", "missing", "incomplete", "rc"])
def test_pair_requires_both_complete_arms(monkeypatch, tmp_path, fault):
    calls = []

    def fake_run(argv, check):
        calls.append(argv[0])
        assert check
        if fault == "rc" and argv[0] == "base":
            raise RuntimeError("base failed")
        rows = [dict(part="conc32", trial=i, cts=[1024, 1024]) for i in range(3)]
        if fault == "short":
            rows[-1]["cts"] = [1024, 993]
        if fault == "incomplete":
            rows.pop()
        if fault != "missing":
            rows.append(dict(complete=True))
        Path(argv[1]).write_text("\n".join(json.dumps(r) for r in rows))

    monkeypatch.setattr("verify.conc_pair.subprocess.run", fake_run)
    out = tmp_path / "pair.jsonl"
    commands = {arm: [arm, "{arm_out}"] for arm in ("base", "cand")}
    if fault:
        with pytest.raises(RuntimeError):
            run_pair(out, commands)
        assert calls == ["base"]
        assert '"complete": true' not in out.read_text()
    else:
        run_pair(out, commands)
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        assert calls == ["base", "cand"]
        assert [r["arm"] for r in rows[:-1]] == ["base"] * 3 + ["cand"] * 3
        assert rows[-1] == {"complete": True}


def test_long_speed_keeps_long_workload():
    from verify.suites import parse_suite

    long = parse_suite("long")
    speed = parse_suite("long-speed")
    assert speed["stages"] == ["speed"]
    for key in long.keys() - {"name", "stages"}:
        assert speed[key] == long[key], key
    assert speed["decode_tokens"] == 2048
    assert speed["turn2_tokens"] == 256
    assert speed["long_ask"] is True
    assert speed["speed_tol_pct"] == 3.0
