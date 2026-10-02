"""Do not record GPU A/B numbers while an independent full unit suite runs."""

import json
from types import SimpleNamespace

from scripts.research.probe_checkpoint_http import assert_no_full_unit_suites


def test_http_arms_use_one_canonical_second_turn(monkeypatch, tmp_path):
    from scripts.research import probe_checkpoint_http as probe

    sent = []
    log = tmp_path / "server.log"
    log.write_text("Speculative decoding: mtp")
    for kind in ("prose", "code"):
        (tmp_path / f"turn2-reply-{kind}-1024.json").write_text(
            json.dumps(dict(reply="fixed reference answer"))
        )
    server = SimpleNamespace(
        url="http://unused", model="model", log=log, kill=lambda: None
    )
    monkeypatch.setattr(probe, "assert_no_full_unit_suites", lambda: None)
    monkeypatch.setattr(probe.t, "Srv", lambda *a: server)
    monkeypatch.setattr(probe.t, "load_prompt", lambda name: "prompt")
    monkeypatch.setattr(probe.t, "was_contended", lambda: False)
    monkeypatch.setattr(probe.t, "YUNSHU_BIN", probe.t.YUNSHU_BIN)
    monkeypatch.setattr(probe.t, "YUNSHU_SRC", probe.t.YUNSHU_SRC)

    def send(url, body):
        sent.append(body)
        return dict(_text="this arm's different answer", ct=1, pt=32, sha="digest")

    monkeypatch.setattr(probe.t, "send", send)
    out = tmp_path / "results.jsonl"
    monkeypatch.setattr(
        probe.sys,
        "argv",
        ["probe", "--mode", "sync", "--ctx", "1024", "--out", str(out)],
    )
    probe.main()
    turns = [r for r in sent if len(r["messages"]) == 3]
    assert len(turns) == 2
    assert all(r["messages"][1]["content"] == "fixed reference answer" for r in turns)
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert records[0]["request_sha256"] == records[1]["request_sha256"]
    assert records[-1]["phase"] == "complete"


def process(pid, *argv):
    return SimpleNamespace(info={"pid": pid, "cmdline": list(argv)})


def test_full_suite_is_reported_not_fatal(capsys):
    suite = ("/external/.venv/bin/python", "-m", "pytest", "tests/unit", "-q")
    assert assert_no_full_unit_suites([process(113, *suite)]) == [113]
    assert "113" in capsys.readouterr().err


def test_niced_full_suite_is_ignored():
    suite = ("python", "-m", "pytest", "tests/unit", "-q")
    p = process(114, *suite)
    p.info["nice"] = 15
    assert assert_no_full_unit_suites([p]) == []


def test_agent_prompt_mentions_do_not_match_actual_pytest_process():
    assert_no_full_unit_suites(
        [
            process(
                15753,
                "codex",
                "exec",
                "Run python -m pytest tests/unit -q before commit",
            )
        ]
    )


def test_focused_cpu_test_is_not_a_full_suite():
    assert_no_full_unit_suites(
        [
            process(
                114,
                "python",
                "-m",
                "pytest",
                "tests/unit/test_apc_deferred_checkpoint.py",
            )
        ]
    )
