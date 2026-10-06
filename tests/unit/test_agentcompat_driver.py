"""agentcompat driver helpers: every new census item must be named in AGENT_COMPAT.md; verdict is fail-closed."""

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "scripts/research/agent_compat")
)
import agentcompat as ac  # noqa: E402
import census_diff as cd  # noqa: E402


def test_doc_gaps_names_every_new_item():
    new = {
        "header_values": ["anthropic-beta=thinking-display-updates-2026-08-18"],
        "fields": ["/v1/messages thinking.display"],
        "headers": ["x-opencode-session-id"],
        "endpoints": ["POST /v1/foo?beta"],
    }
    assert len(ac.doc_gaps(new, "")) == 4
    doc = "thinking-display-updates-2026-08-18 thinking.display x-opencode-session-id /v1/foo"
    assert ac.doc_gaps(new, doc) == []
    assert len(ac.doc_gaps(new, "thinking.display")) == 3


def test_census_signature_diff(tmp_path):
    def mk(name, beta, extra):
        d = tmp_path / name / "s"
        d.mkdir(parents=True)
        body = {"model": "m", "thinking": {"type": "adaptive", **extra}}
        (d / "requests.jsonl").write_text(
            __import__("json").dumps(
                {
                    "method": "POST",
                    "path": "/v1/messages?beta=true",
                    "headers": {"anthropic-beta": beta, "Host": "x"},
                    "body": body,
                }
            )
            + "\n"
        )
        return tmp_path / name

    old, new = mk("old", "a,b", {}), mk("new", "a,b,c", {"display": "updates"})
    d = cd.diff(cd.signature(old), cd.signature(new))
    assert d["header_values"] == ["anthropic-beta=c"]
    assert d["fields"] == ["/v1/messages thinking.display"]
    assert cd.diff(cd.signature(new), cd.signature(old)) == {}


def test_run_watched_kills_client_when_tunnel_drops():
    import subprocess

    tunnel = subprocess.Popen(["sleep", "1"])
    rc, text = ac.run_watched(["sleep", "60"], tunnel, 60)
    assert rc == 125 and "TUNNEL DROPPED" in text


def test_run_watched_times_out():
    import subprocess

    tunnel = subprocess.Popen(["sleep", "30"])
    try:
        rc, text = ac.run_watched(["sleep", "60"], tunnel, 1)
    finally:
        tunnel.kill()
    assert rc == 124 and "TIMEOUT" in text


def test_replay_thin_keeps_last_request():
    import replay

    assert replay.thin(list(range(6)), 2) == [0, 5]
    assert replay.thin(list(range(6)), 0) == list(range(6))
