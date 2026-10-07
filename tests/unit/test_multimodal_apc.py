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
