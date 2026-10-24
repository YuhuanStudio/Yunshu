"""CPU-only contracts for the sampled decode timing driver."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts.research.sampledspec_speed import (
    arm_order,
    validate_row,
    verify_engagement,
)


def test_interleaved_order():
    assert arm_order(3) == ["base", "cand", "cand", "base", "base", "cand"]
    assert arm_order(1, 1) == ["cand", "base"]
    assert [arm for rep in range(3) for arm in arm_order(1, rep)] == arm_order(3)


@pytest.mark.parametrize("option,value", [("--reps", "0"), ("--rep-offset", "-1")])
def test_invalid_rep_ranges_fail_before_importing_the_backend(
    tmp_path, monkeypatch, capsys, option, value
):
    from scripts.research.sampledspec_speed import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sampledspec_speed.py",
            "--base-src",
            str(tmp_path / "base"),
            "--cand-src",
            str(tmp_path / "cand"),
            "--out",
            str(tmp_path / "out"),
            "--dry-run",
            option,
            value,
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "--reps must be positive" in capsys.readouterr().err


@pytest.mark.parametrize("cached", [0, 1])
def test_split_rep_uses_real_request_params_and_fails_before_later_arms(
    tmp_path, monkeypatch, cached
):
    from scripts.research import sampledspec_speed as speed

    source = tmp_path / "cand"
    output = tmp_path / "cpu.jsonl"
    servers, requests = [], []
    backend = SimpleNamespace(D="fixture-draft")

    class Server:
        def __init__(self, engine, env, label):
            assert env["YUNSHU_VLM_APC_DISK"] == "0"
            assert env["YUNSHU_VLM_APC_MEMORY_GB"] == "0"
            self.candidate = str(source.resolve()) == backend.YUNSHU_SRC
            self.log = tmp_path / f"{label}.log"
            self.log.write_text("")
            self.url, self.model, self.killed = self, "fixture-model", False
            servers.append(self)

        def kill(self):
            self.killed = True

        def verify_spec_mode(self):
            pass

    def send(server, request):
        extra = request["extra"]
        if extra is not None:
            requests.append(request)
            if server.candidate:
                with server.log.open("a") as log:
                    log.write(
                        f"DFlash fast verification: keyed={extra['temperature'] > 0}\n"
                    )
        return dict(
            cached=cached,
            ct=request["tokens"],
            dec_tps=40,
            xy={"speculative": {"rounds": 16}},
        )

    backend.Srv = Server
    backend.send = send
    backend.req = lambda model, text, tokens, extra=None: dict(
        text=text, tokens=tokens, extra=extra
    )
    backend.decode_prompt = lambda *a: "fixture prompt"
    monkeypatch.setitem(sys.modules, "tfbench", backend)
    monkeypatch.setattr(speed.shutil, "rmtree", lambda *a, **kw: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sampledspec_speed.py",
            "--base-src",
            str(tmp_path / "base"),
            "--cand-src",
            str(source),
            "--ctx",
            "16384",
            "--kind",
            "code",
            "--reps",
            "1",
            "--rep-offset",
            "1",
            "--tokens",
            "64",
            "--out",
            str(output),
        ],
    )
    if cached:
        with pytest.raises(RuntimeError, match="APC hit"):
            speed.main()
        assert len(servers) == 1 and not output.exists()
    else:
        speed.main()
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert rows[-1] == {"complete": True}
        assert [row["arm"] for row in rows[:-1]] == ["cand"] * 3 + ["base"] * 3
        assert [row["index"] for row in rows[:-1]] == [2] * 3 + [3] * 3
        assert all(row["rep"] == 1 for row in rows[:-1])
        assert [row["engaged"] for row in rows[:3]] == [
            "fast-greedy",
            "fast-keyed",
            "fast-keyed",
        ]
        for first, second in zip(requests[:3], requests[3:], strict=True):
            assert first == second
        assert [row["extra"] for row in requests[:3]] == list(speed.SAMPLING.values())
    assert all(server.killed for server in servers)


def test_timing_rows_fail_closed():
    row = dict(cached=0, ct=65, dec_tps=40, xy={"speculative": {"rounds": 16}})
    assert validate_row(row) == 4
    for bad in (
        {**row, "cached": 1},
        {**row, "cached": None},
        {**row, "ct": 2},
        {**row, "xy": {}},
    ):
        with pytest.raises(RuntimeError):
            validate_row(bad)


def test_each_sampled_candidate_cell_must_engage_the_keyed_tree():
    for sampling in ("sampled", "qwen"):
        for log in ("", "DFlash fast verification: keyed=False"):
            with pytest.raises(RuntimeError, match="candidate cell"):
                verify_engagement("cand", sampling, log)
        assert (
            verify_engagement("cand", sampling, "DFlash fast verification: keyed=True")
            == "fast-keyed"
        )
    assert verify_engagement("base", "sampled", "") == "unreported"
    assert (
        verify_engagement("cand", "greedy", "DFlash fast verification: keyed=False")
        == "fast-greedy"
    )


def test_dry_run_never_imports_mlx_or_starts_server(tmp_path):
    path = Path(__file__).resolve().parents[2] / "scripts/research/sampledspec_speed.py"
    result = subprocess.run(
        [
            sys.executable,
            str(path),
            "--base-src",
            str(tmp_path / "base"),
            "--cand-src",
            str(tmp_path / "cand"),
            "--out",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["complete"] is True
    assert not (tmp_path / "out").exists()
