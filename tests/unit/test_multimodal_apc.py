"""CPU preflight for the raw-ID media checkpoint measurement."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/research/multimodal_apc.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("multimodal_apc", SCRIPT)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def test_arguments_fixture_and_identity_rule():
    a = p.parser().parse_args(
        ["--model", "/model", "--out", "/result", "--sizes", "1", "4096"]
    )
    assert a.sizes == [1, 4096]
    assert p.messages(1) != p.messages(1, (30, 30, 200))
    cold = {"ids": [1, 2], "cached": 0}
    warm = {"ids": [1, 2], "cached": 107}
    p.validate_pair(cold, warm, True)
    with pytest.raises(ValueError, match="raw token"):
        p.validate_pair(cold, {**warm, "ids": [1, 3]}, True)
    with pytest.raises(ValueError, match="not engaged"):
        p.validate_pair(cold, cold, True)
    skipped = {**cold, "restore_skipped": True}
    p.validate_pair(cold, skipped, True, allow_skip=True)
    with pytest.raises(ValueError, match="not engaged"):
        p.validate_pair(cold, skipped, True)


def test_joint_media_fixture_is_deterministic_pcm_and_keeps_image():
    import base64
    import io
    import wave

    msg = p.messages(1)
    combined = p.with_audio(msg)
    assert combined == p.with_audio(msg)
    assert len(msg[0]["content"]) == 2
    assert combined[0]["content"][0] == msg[0]["content"][0]
    with wave.open(
        io.BytesIO(base64.b64decode(combined[0]["content"][1]["input_audio"]["data"]))
    ) as f:
        assert (f.getnchannels(), f.getframerate(), f.getnframes()) == (1, 16000, 8000)


@pytest.mark.asyncio
async def test_fake_engine_captures_hidden_tokens_and_finishes():
    from types import SimpleNamespace

    class Engine:
        _apc_backend = None

        def _runner_events(self):
            yield ("", 7, "reasoning", None, 0, None)
            yield ("red", 8, "normal", "stop", 0, None)

        async def generate_stream(self, **kwargs):
            for event in self._runner_events():
                yield SimpleNamespace(
                    new_text=event[0],
                    new_token_ids=[event[1]],
                    finished=event[3] is not None,
                    cached_tokens=107,
                    prompt_tokens=108,
                )

    engine = Engine()
    result = await p.probe(engine, p.messages(1))
    assert result["ids"] == [7, 8]
    assert result["cached"] == 107 and result["text"] == "red"


@pytest.mark.asyncio
async def test_anthropic_probe_on_fake_http_before_loading_a_model():
    class Engine:
        _apc_backend = None

        def _runner_events(self):
            yield ("red", 8, "normal", "stop", 0, None)

    engine = Engine()

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return dict(
                type="message",
                stop_reason="end_turn",
                content=[dict(type="text", text="red")],
                usage=dict(
                    input_tokens=8,
                    cache_read_input_tokens=49,
                    cache_creation_input_tokens=0,
                ),
            )

    class Client:
        async def post(self, path, json):
            assert path == "/v1/messages"
            assert json["messages"][0]["content"][0]["cache_control"] == {
                "type": "ephemeral"
            }
            list(engine._runner_events())
            return Response()

    result = await p.anthropic_probe(engine, Client(), p.anthropic_body("omni"))
    assert result["ids"] == [8] and result["cached"] == 49 and result["pt"] == 57


def test_image_conditioned_paired_set_has_known_answers():
    for i, expected in [(0, 101), (1, 93), (2, 181), (3, 81), (199, 82)]:
        msg, gold = p.arithmetic_item(i)
        assert gold == expected
        assert msg[0]["content"][0]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        )
    assert (
        len({p.arithmetic_item(i)[0][0]["content"][1]["text"] for i in range(200)})
        == 200
    )


@pytest.mark.asyncio
async def test_paused_followup_restores_its_producer_before_retry(monkeypatch):
    monkeypatch.setenv("GPUQ_DEVICE", "m5")
    pauses = iter([True, False])
    monkeypatch.setattr(p, "was_paused", lambda *a: next(pauses))
    calls = []

    async def sample():
        calls.append("followup")
        return dict(ttft_s=len(calls))

    async def restore():
        calls.append("producer")

    result = await p.measured(sample, restore)
    assert calls == ["followup", "producer", "followup"]
    assert result["timing_valid"] and result["paused_retries"] == 1


@pytest.mark.asyncio
async def test_m3_has_no_valid_timing(monkeypatch):
    monkeypatch.setenv("GPUQ_DEVICE", "m3")
    monkeypatch.setattr(p, "was_paused", lambda *a: True)

    async def sample():
        return dict(ttft_s=1)

    assert not (await p.measured(sample))["timing_valid"]
