"""Qwen4 native-head loader contracts; tiny arrays run on the CPU."""

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest

from yunshu_engine.model_patches import sanitize_qwen4_checkpoint
from yunshu_engine.qwen4_mtp import configure_lane, load_native_head


def tiny_config():
    return dict(
        model_type="qwen4_exp",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_value_heads=2,
        linear_num_key_heads=1,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=32,
        moe_intermediate_size=32,
        rms_norm_eps=1e-6,
        vocab_size=64,
        max_position_embeddings=4096,
        hc_count=2,
        hc_lowrank=8,
        ple_layer_ids=[],
        ple_embed_dim=32,
        heads_per_ngram=2,
        indexer_n_heads=2,
        indexer_head_dim=16,
        indexer_budget=8,
        indexer_compress_ratio=4,
        layer_types=["linear_attention", "qwen_sparse_attention"],
    )


def test_sanitize_preserves_head_and_calls_target_without_it():
    seen = []
    model = SimpleNamespace(sanitize=lambda weights: seen.append(weights) or weights)
    base, head = sanitize_qwen4_checkpoint(model, {"base": 1, "mtp.fc.weight": 2})
    assert base == {"base": 1}
    assert head == {"fc.weight": 2}
    assert seen == [base]
    with pytest.raises(ValueError, match="Duplicate"):
        sanitize_qwen4_checkpoint(model, {"mtp.a": 1, "model.mtp.a": 2})


def test_qwen4_checkpoint_detection(tmp_path):
    from yunshu_engine.mlxvlm_mtp import is_mtp_capable

    (tmp_path / "config.json").write_text(
        json.dumps(
            {"model_type": "qwen4_exp", "text_config": {"mtp_num_hidden_layers": 1}}
        )
    )
    (tmp_path / "head.safetensors").touch()
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.mtp.fc_hidden.weight": "head.safetensors"}})
    )
    assert is_mtp_capable(str(tmp_path))


def test_lane_admission_and_depth():
    lm = SimpleNamespace(
        speculative_verify_hidden=lambda: None,
        speculative_draft_hidden=lambda: None,
        _supports_batch_invariant_decode=lambda: True,
    )
    assert configure_lane(lm, object(), None) == ({"qwen4_native": True}, 4)
    assert configure_lane(lm, object(), 5)[1] == 5
    assert configure_lane(lm, None, None)[1] is None
    lm._supports_batch_invariant_decode = lambda: False
    with pytest.raises(RuntimeError, match="plain decode"):
        configure_lane(lm, object(), None)


def test_native_head_strict_load_and_hyper_state():
    from mlx.utils import tree_flatten
    from mlx_vlm.speculative.drafters.qwen4_exp_mtp.config import Qwen4ExpMTPConfig
    from mlx_vlm.speculative.drafters.qwen4_exp_mtp.qwen4_exp_mtp import (
        Qwen4ExpMTPDraftModel,
    )

    with mx.stream(mx.cpu):
        config = {"text_config": tiny_config()}
        reference = Qwen4ExpMTPDraftModel(Qwen4ExpMTPConfig.from_dict(config))
        weights = dict(tree_flatten(reference.parameters()))
        loaded = load_native_head(config, weights)
        embed = mx.ones((1, 2, 32))
        hidden = mx.ones((1, 2, 64))
        assert mx.array_equal(
            reference.fuse_inputs(embed, hidden), loaded.fuse_inputs(embed, hidden)
        ).item()
        assert loaded.make_cache()[0].__class__.__name__ == "QSAKVCache"
        with pytest.raises(ValueError, match="hidden shape"):
            loaded.fuse_inputs(embed, mx.ones((1, 2, 32)))
        with pytest.raises(ValueError):
            load_native_head(
                config, {k: v for k, v in weights.items() if k != "fc_hidden.weight"}
            )


def test_probe_cpu_dry_run(tmp_path):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "scripts/research/qwen4_mtp_probe.py"
    spec = importlib.util.spec_from_file_location("qwen4_probe", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    out = tmp_path / "result.json"
    assert probe.main(["--dry-run", "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["complete"] and result["dry_run"] and result["passed"]


def test_mixed_head_uses_tensor_geometry_not_top_level_quantization():
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_vlm.speculative.drafters.qwen4_exp_mtp.config import Qwen4ExpMTPConfig
    from mlx_vlm.speculative.drafters.qwen4_exp_mtp.qwen4_exp_mtp import (
        Qwen4ExpMTPDraftModel,
    )

    with mx.stream(mx.cpu):
        config = {
            "text_config": tiny_config(),
            "quantization": {"bits": 3, "group_size": 64},
        }
        reference = Qwen4ExpMTPDraftModel(Qwen4ExpMTPConfig.from_dict(config))
        nn.quantize(
            reference,
            bits=4,
            group_size=32,
            class_predicate=lambda path, _: path == "fc_hidden",
        )
        loaded = load_native_head(config, dict(tree_flatten(reference.parameters())))
        assert loaded.fc_hidden.bits == 4
        assert loaded.fc_hidden.group_size == 32
        embed, hidden = mx.ones((1, 2, 32)), mx.ones((1, 2, 64))
        assert mx.array_equal(
            reference.fuse_inputs(embed, hidden), loaded.fuse_inputs(embed, hidden)
        ).item()


def test_tiny_target_has_multimodal_rope_configuration():
    from mlx_vlm.models.qwen4_exp.config import ModelConfig
    from mlx_vlm.models.qwen4_exp.language import LanguageModel

    with mx.stream(mx.cpu):
        cfg = ModelConfig.from_dict(
            {
                "model_type": "qwen4_exp",
                "text_config": tiny_config(),
                "vision_config": {"deepstack_visual_indexes": []},
            }
        )
        target = LanguageModel(cfg.text_config, cfg)
        positions, _ = target.get_rope_index(mx.array([[1, 3, 5]]))
        assert positions.shape == (1, 3)
        assert positions.tolist() == [[0, 1, 2]]


def test_common_loader_sanitizes_target_once():
    from yunshu_engine.vlm_engine import _prepare_vlm_weights

    calls = []
    model = SimpleNamespace(sanitize=lambda w: calls.append(w) or {"converted": 1})
    cfg = SimpleNamespace()
    model_class = SimpleNamespace()
    base, head = sanitize_qwen4_checkpoint(
        model,
        {"raw": 1, "mtp.fc_hidden.weight": 2},
        prepare=lambda w: _prepare_vlm_weights(model, model_class, cfg, w),
    )
    assert calls == [{"raw": 1}]
    assert base == {"converted": 1}
    assert head == {"fc_hidden.weight": 2}


def test_native_tiny_rounds_on_cpu(monkeypatch):
    import importlib.util
    from pathlib import Path

    import mlx.core as mx

    path = Path(__file__).resolve().parents[2] / "scripts/research/qwen4_mtp_probe.py"
    spec = importlib.util.spec_from_file_location("qwen4_native_cpu", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        from mlx_vlm.speculative import common, mtp

        cpu_stream = mx.new_stream(mx.cpu)
        monkeypatch.setattr(common, "generation_stream", cpu_stream)
        monkeypatch.setattr(mtp, "generation_stream", cpu_stream)
        result = probe.probe()
        assert result["device"] == "cpu"
        assert result["passed"], json.dumps(result)
        assert len(result["checks"]) == 7
    finally:
        mx.set_default_device(previous)


def _tiny_policy_run(monkeypatch, depth, min_prob, max_tokens=24):
    """(plain tokens, policy tokens, [(drafted, accepted)]) on the CPU tiny config."""
    import importlib.util
    from pathlib import Path

    import mlx.core as mx
    from mlx.utils import tree_flatten

    path = Path(__file__).resolve().parents[2] / "scripts/research/qwen4_mtp_probe.py"
    spec = importlib.util.spec_from_file_location("qwen4_policy_probe", path)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        from mlx_vlm.models.qwen4_exp.config import ModelConfig
        from mlx_vlm.models.qwen4_exp.language import LanguageModel
        from mlx_vlm.speculative import common, mtp
        from mlx_vlm.speculative.drafters.qwen4_exp_mtp.config import Qwen4ExpMTPConfig
        from mlx_vlm.speculative.drafters.qwen4_exp_mtp.qwen4_exp_mtp import (
            Qwen4ExpMTPDraftModel,
        )

        from yunshu_engine import qwen4_draft_policy as policy

        cpu_stream = mx.new_stream(mx.cpu)
        monkeypatch.setattr(common, "generation_stream", cpu_stream)
        monkeypatch.setattr(mtp, "generation_stream", cpu_stream)
        mx.random.seed(17)
        config = probe.tiny_config()
        model_config = ModelConfig.from_dict(
            {
                "model_type": "qwen4_exp",
                "text_config": config,
                "vision_config": {
                    "depth": 1,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_heads": 2,
                    "out_hidden_size": 32,
                    "deepstack_visual_indexes": [],
                },
            }
        )
        target = LanguageModel(model_config.text_config, model_config)
        ref = Qwen4ExpMTPDraftModel(
            Qwen4ExpMTPConfig.from_dict({"text_config": config})
        )
        head = load_native_head(
            {"text_config": config}, dict(tree_flatten(ref.parameters()))
        )
        prompt = mx.array([[1, 3, 5, 7, 9]], dtype=mx.int32)

        def sample(logits):
            return mx.argmax(logits, axis=-1)

        plain_cache = target.make_cache()
        out = target(prompt, cache=plain_cache)
        token = int(sample(out.logits[:, -1:]).item())
        plain = [token]
        for _ in range(max_tokens - 1):
            out = target._batch_invariant_decode(mx.array([[token]]), cache=plain_cache)
            token = int(sample(out.logits[:, -1:]).item())
            plain.append(token)

        cache = target.make_cache()
        out = target(prompt, cache=cache, return_hidden=True)
        first = sample(out.logits[:, -1:])
        stats: list = []
        policy._STATE["stats"] = stats
        try:
            ids = [first.item()] + [
                t[0]
                for t, _ in policy.rounds(
                    target,
                    head,
                    cache,
                    out.hidden_states[-1],
                    out.shared_kv_states,
                    prompt_tokens=prompt,
                    first_bonus=first.reshape(-1),
                    max_tokens=max_tokens,
                    sampler=sample,
                    draft_block_size=depth + 1,
                    token_dtype=mx.int32,
                    stop_check=None,
                    eos_token_ids=None,
                    min_prob=min_prob,
                )
            ]
        finally:
            policy._STATE["stats"] = None
        return plain, ids, stats
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
@pytest.mark.parametrize("min_prob", [0.0, 0.001, 0.5, 0.99])
def test_draft_policy_identity(monkeypatch, depth, min_prob):
    plain, ids, stats = _tiny_policy_run(monkeypatch, depth, min_prob)
    assert ids == plain
    assert stats and all(1 <= drafted <= depth for drafted, _ in stats)


def test_draft_policy_early_stop_and_zero_threshold(monkeypatch):
    # A random head is nowhere near 50% sure: every round keeps only the first draft.
    _, _, gated = _tiny_policy_run(monkeypatch, 4, 0.5)
    assert gated and {d for d, _ in gated} == {1}
    # Threshold 0 never stops: full depth except where the token budget ends a round.
    _, _, free = _tiny_policy_run(monkeypatch, 4, 0.0)
    assert max(d for d, _ in free) == 4 and sum(d for d, _ in free) > sum(
        d for d, _ in gated
    )
    # Threshold 0 is the upstream fixed-depth round, token for token.
    plain, ids, _ = _tiny_policy_run(monkeypatch, 4, 0.0)
    assert ids == plain
