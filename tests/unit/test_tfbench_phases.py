import importlib.util
import io
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


def test_identity_cells_accept_a_model_ended_reply_speed_cells_do_not():
    stopped = dict(finish="stop", ct=93)
    with pytest.raises(RuntimeError):
        m.check_decode_len(stopped, 256, "cell")
    m.check_decode_len(stopped, 256, "cell", allow_eos=True)
    m.check_decode_len(dict(finish="length", ct=256), 256, "cell", allow_eos=True)
    for bad in (
        dict(finish="stop", ct=0),
        dict(finish="stop", ct=300),
        dict(finish="error", ct=10),
    ):
        with pytest.raises(RuntimeError):
            m.check_decode_len(bad, 256, "cell", allow_eos=True)


def test_identity_stage_requests_allow_eos_and_speed_does_not():
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).parents[2] / "scripts"))
    from verify import stages

    ctx = SimpleNamespace(suite={"decode_tokens": 256})
    assert "--allow-eos" in stages._dec_args(ctx, stage_logprobs=True)
    assert "--allow-eos" not in stages._dec_args(ctx)


@pytest.mark.parametrize("allow_eos", [False, True])
def test_followup_speed_cell_ignores_eos_identity_cell_does_not(monkeypatch, allow_eos):
    """Flash-Next ends "continue at the same length" after ~150 tokens, which failed every speed
    run at the follow-up cell: a speed follow-up asks past EOS, an identity follow-up never does."""
    from types import SimpleNamespace

    sent = []

    def send(_url, body):
        sent.append(body)
        n = body.get("max_tokens") or body.get("max_completion_tokens")
        return dict(
            _text="x",
            ttft_s=1,
            dec_tps=10,
            cached=0,
            pt=1024,
            ct=n,
            finish="length",
            sha="s",
        )

    monkeypatch.setattr(m, "send", send)
    monkeypatch.setattr(m, "decode_prompt", lambda *_a: "material")
    monkeypatch.setattr(m, "check_rehearsal_engaged", lambda *_a: None)
    monkeypatch.setattr(m, "emit", lambda *_a, **_k: None)
    a = SimpleNamespace(
        phases="cold,turn2",
        decode_tokens=64,
        turn2_tokens=32,
        smoke=False,
        only_ctx=[1024],
        only_kind=["prose"],
        long_ask=False,
        warm_reps=1,
        request_extra=None,
        logprobs=False,
        rehearse=False,
        think_s=0,
        allow_eos=allow_eos,
        engine="yunshu",
    )
    m.part_decode(
        SimpleNamespace(url="fake", model="fake"),
        io.StringIO(),
        a,
    )
    cold, turn2 = sent
    assert "ignore_eos" not in cold
    assert turn2.get("ignore_eos", False) is (not allow_eos)
