"""Flash-Next's DFlash features are mixed layers, never MTP hyper-states."""

import mlx.core as mx
import pytest

from tests.unit.test_qwen4_mtp import tiny_config


@pytest.fixture
def target(monkeypatch):
    from mlx_vlm.models.qwen4_exp.config import ModelConfig
    from mlx_vlm.models.qwen4_exp.language import LanguageModel, Qwen4ExpDecoderLayer

    monkeypatch.setattr(Qwen4ExpDecoderLayer, "__call__", Qwen4ExpDecoderLayer.__call__)

    from yunshu_engine.qwen4_dflash import install

    monkeypatch.setattr(LanguageModel, "__call__", LanguageModel.__call__)
    monkeypatch.setattr(LanguageModel, "_yunshu_dflash_capture", False, raising=False)
    monkeypatch.setattr(
        LanguageModel, "speculative_verify_dflash_hidden", None, raising=False
    )
    with mx.stream(mx.cpu):
        cfg = ModelConfig.from_dict(
            {
                "model_type": "qwen4_exp",
                "text_config": tiny_config(),
                "vision_config": {"deepstack_visual_indexes": []},
            }
        )
        lm = LanguageModel(cfg.text_config, cfg)
        install(lm)
        yield lm


def test_prefill_and_decode_keep_every_dflash_layer(target):
    with mx.stream(mx.cpu):
        cache = target.make_cache()
        prompt = target(
            mx.array([[1, 2, 3]]),
            cache=cache,
            capture_layer_ids=[0, 1],
            return_hidden=True,
        )
        step = target(mx.array([[4]]), cache=cache, capture_layer_ids=[0, 1])
        assert [x.shape for x in prompt.hidden_states] == [(1, 3, 32)] * 2
        assert [x.shape for x in step.hidden_states] == [(1, 1, 32)] * 2
        features = mx.concatenate(
            [
                mx.concatenate(prompt.hidden_states, axis=-1),
                mx.concatenate(step.hidden_states, axis=-1),
            ],
            axis=1,
        )
        assert features.shape == (1, 4, 64)


def test_dflash_verify_preserves_native_target_and_hc_state(target):
    from yunshu_engine.qwen4_dflash import capture_forward

    with mx.stream(mx.cpu):
        ids = mx.array([[1, 2]])
        native = target._batch_invariant_decode(ids, cache=target.make_cache())
        out, hyper = capture_forward(target, ids, target.make_cache(), [0, 1])
        assert hyper.shape == (1, 2, 64)
        assert [x.shape for x in out.hidden_states] == [(1, 2, 32)] * 2
        assert mx.array_equal(native.logits, out.logits).item()
        assert mx.array_equal(native.hidden_states[-1], hyper).item()
        from yunshu_engine.qwen4_dflash import stream_mean

        assert mx.array_equal(out.hidden_states[-1], stream_mean(hyper, 2, 32)).item()
        captured, final, transaction = target.speculative_verify_dflash_hidden(
            ids, target.make_cache(), [0, 1]
        )
        assert len(captured) == 2 and final.shape == hyper.shape
        assert callable(transaction.abort)
        transaction.abort()


def test_mtp_retains_full_hyper_state(target):
    with mx.stream(mx.cpu):
        out = target(mx.array([[1, 2]]), cache=target.make_cache(), return_hidden=True)
        assert [x.shape for x in out.hidden_states] == [(1, 2, 64)]


def test_official_features_are_stream_mean_not_final_learned_mixer():
    from yunshu_engine.qwen4_dflash import stream_mean

    with mx.stream(mx.cpu):
        streams = mx.array([[[1.0, 3.0, 9.0, 5.0]]])
        assert stream_mean(streams, 2, 2).tolist() == [[[5.0, 4.0]]]
