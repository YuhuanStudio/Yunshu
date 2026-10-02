"""Text speculation selects only tested routes; native Qwen MTP belongs to VLM."""

from __future__ import annotations

import inspect
import json
from unittest.mock import MagicMock

import pytest

from yunshu_engine import settings, spec_select
from yunshu_engine.batched_engine import BatchedEngine
from yunshu_engine.model_manager import ModelType, _detect_model_type


def _engine(**attrs):
    eng = object.__new__(BatchedEngine)
    base = dict(_ngram_proposer=object(), _ngram_greedy_default=False)
    base.update(attrs)
    for k, v in base.items():
        setattr(eng, k, v)
    return eng


def _route(eng, **kw):
    args = dict(
        spec_decode=True,
        stream=False,
        temperature=0.0,
        logprobs=False,
        use_engine_loop=False,
    )
    args.update(kw)
    return eng._spec_route(**args)


def test_remaining_text_routes():
    eng = _engine()
    assert _route(eng) == "ngram"
    assert _route(eng, spec_decode=False) is None
    assert _route(_engine(_ngram_greedy_default=True), spec_decode=False) == "ngram"
    assert _route(eng, temperature=0.7) is None
    assert _route(eng, logprobs=True) is None
    assert _route(eng, use_engine_loop=True) is None
    assert _route(eng, gemma4_eligible=lambda: True) == "gemma4_assistant"
    assert _route(eng, stream=True, gemma4_eligible=lambda: True) is None


@pytest.mark.parametrize("flag", ["eagle", "mtp", "mlxvlm_mtp"])
def test_retired_flags_cannot_select_unsafe_text_routes(monkeypatch, flag):
    monkeypatch.setenv("YUNSHU_SPEC_UNVERIFIED", flag)
    monkeypatch.setenv("YUNSHU_DRAFT_MODEL", "/missing/eagle")
    # Even stale state from a plugin cannot re-enable the removed dispatch.
    eng = _engine(_spec_enabled=True, _spec_decoder=object(), _mtp_decoder=object())
    assert _route(eng) == "ngram"
    assert _route(eng, stream=True) is None
    assert (
        not {"YUNSHU_SPEC_UNVERIFIED", "YUNSHU_DRAFT_MODEL"} & settings.REGISTRY.keys()
    )
    assert not hasattr(BatchedEngine, "_generate_mtp")
    assert not hasattr(BatchedEngine, "_generate_speculative")


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_6"])
def test_native_mtp_checkpoint_dispatches_to_vlm(tmp_path, model_type):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "vision_config": {},
                "text_config": {"mtp_num_hidden_layers": 1},
            }
        )
    )
    assert _detect_model_type(str(tmp_path)) == ModelType.VLM


def test_plain_text_checkpoint_dispatches_to_llm(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    assert _detect_model_type(str(tmp_path)) == ModelType.LLM


@pytest.mark.parametrize(
    "override,mode", [("mtp", "mtp"), ("/draft/dflash", "dflash"), ("off", "none")]
)
def test_vlm_spec_selection_survives_text_flag_removal(monkeypatch, override, mode):
    monkeypatch.setenv("YUNSHU_VLM_DRAFT", override)
    monkeypatch.setenv("YUNSHU_MTP", "1")
    assert spec_select.choose({}, spec_family=True, mtp_capable=True).kind == mode
    assert spec_select.choose({}, spec_family=False, mtp_capable=True).kind == "none"


def test_text_scheduler_receives_only_ngram():
    eng = _engine(_engine_core=MagicMock())
    eng._ngram_proposer = MagicMock()
    eng._engine_core.scheduler._ngram_proposer = None
    eng._wire_spec_decoders_to_scheduler()
    eng._engine_core.scheduler.enable_ngram_spec.assert_called_once()
    eng._engine_core.scheduler.set_spec_decoder.assert_not_called()
    eng._engine_core.scheduler.set_mtp_decoder.assert_not_called()


def test_live_streaming_path_has_holdback():
    assert "StopHoldbackBuffer" in inspect.getsource(
        BatchedEngine._stream_generate_fast
    )
    assert "UNREACHABLE" in inspect.getsource(BatchedEngine._stream_generate_ngram_spec)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ngram,gemma,enabled",
    [(None, None, False), (object(), None, True), (None, object(), True)],
)
async def test_monitoring_reports_remaining_text_routes(
    monkeypatch, ngram, gemma, enabled
):
    from yunshu_gateway import engine as gateway_engine
    from yunshu_gateway.routers import monitoring

    eng = _engine(
        model_name="text", _ngram_proposer=ngram, _gemma4_assistant_proposer=gemma
    )
    monkeypatch.setattr(monitoring, "_check_permission", lambda _r: None)
    monkeypatch.setattr(gateway_engine, "get_engine", lambda: eng)
    monkeypatch.setattr(gateway_engine, "get_model_manager", lambda: None)
    model = (await monitoring.spec_decode_stats(MagicMock()))["models"][0]
    assert model["spec_enabled"] is enabled
    assert model["ngram_enabled"] is (ngram is not None)
    assert model["mtp_enabled"] is False
