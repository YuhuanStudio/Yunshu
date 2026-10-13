import importlib.util
from pathlib import Path

import pytest

s = importlib.util.spec_from_file_location(
    "tb", Path(__file__).parents[2] / "scripts/research/tfbench.py"
)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)


def test_default_runs_everything_and_selection_is_exact():
    assert m.parse_phases(",".join(m.ALL_PHASES)) == set(m.ALL_PHASES)
    assert m.parse_phases("cold,warm") == {"cold", "warm"}


def test_bad_or_orphan_selections_fail_closed():
    for bad in ("", "cold,bogus", "warm,turn2"):
        with pytest.raises(SystemExit):
            m.parse_phases(bad)


def test_conc32_requests_long_code_on_every_turn(monkeypatch, tmp_path):
    from types import SimpleNamespace

    requests = []

    def send(_url, body):
        requests.append([dict(msg) for msg in body["messages"]])
        return dict(
            _text="code",
            ttft_s=1,
            dec_tps=10,
            cached=0,
            pt=32768,
            ct=1024,
            finish="length",
            sha="same",
        )

    monkeypatch.setattr(m, "send", send)
    monkeypatch.setattr(m, "load_prompt", lambda _name: "material " * 4000)
    with (tmp_path / "rows.jsonl").open("w") as out:
        m.part_conc32(
            SimpleNamespace(url="fake", model="fake"), out, SimpleNamespace(smoke=False)
        )
    assert len(requests) == 8  # two warmups, then two requests in each of three rounds
    for messages in requests[2:]:
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"].endswith(m.CONC32_TASK)
        assert "at least 400 lines" in messages[-1]["content"]


def test_conc32_short_output_still_fails():
    with pytest.raises(RuntimeError, match="993"):
        m.check_decode_len(dict(ct=993, finish="stop"), 1024, "conc32")
