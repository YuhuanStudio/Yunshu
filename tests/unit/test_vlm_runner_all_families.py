"""Every VLM family on the runner: non-Qwen runner build, audio through
prepare_media, Gemma channel reasoning, and the VLM gateway 400s."""

import importlib
from types import SimpleNamespace

import mlx.core as mx

from yunshu_engine.vlm_batch_runner import RunStats, VLMBatchRunner
from yunshu_engine.vlm_engine import VLMEngine


def test_non_qwen_family_gets_runner_without_speculative_decode(monkeypatch):
    eng = VLMEngine.__new__(VLMEngine)
    eng._config = {"model_type": "gemma4"}
    eng._model = SimpleNamespace(language_model=object())
    eng._processor = object()
    eng._apc_backend = None
    eng._apc_semantic_hash = None
    eng._executor = None
    eng._model_path = "/models/gemma-4-e4b-it"
    eng._mx_large_model = False
    # An unsupported cache must still run without APC.
    eng.backend_capabilities = lambda lm=None: SimpleNamespace(
        cache=SimpleNamespace(has_sliding_window=True)
    )
    monkeypatch.setattr(
        "mlx_vlm.apc.model_apc_plan", lambda lm: SimpleNamespace(restorable=False)
    )
    monkeypatch.setenv("YUNSHU_VLM_DRAFT", "/nonexistent/drafter")

    def _no_spec(*a, **k):
        raise AssertionError("speculative decoding set up for a non-Qwen family")

    monkeypatch.setattr("yunshu_engine.mlxvlm_mtp.is_mtp_capable", _no_spec)
    runner = eng._build_batch_runner("/models/gemma-4-e4b-it")
    assert isinstance(runner, VLMBatchRunner)
    assert runner.drafter is None and runner.apc_manager is None


def test_prepare_media_forwards_audio_features(monkeypatch):
    utils = importlib.import_module("mlx_vlm.utils")
    seen = {}

    def fake_prepare_inputs(processor, images=None, audio=None, prompts=None, **kw):
        seen.update(images=images, audio=audio, prompt=prompts)
        return {
            "input_ids": mx.array([[1, 2, 3]]),
            "input_features": mx.ones((1, 4, 8)),
            "feature_attention_mask": mx.ones((1, 4)),
        }

    monkeypatch.setattr(utils, "prepare_inputs", fake_prepare_inputs)

    class Embeds:
        def to_dict(self):
            return {"inputs_embeds": mx.zeros((1, 3, 2))}

    captured = {}

    def get_input_embeddings(input_ids, pixel_values, mask=None, **data):
        captured.update(data)
        return Embeds()

    model = SimpleNamespace(
        config=SimpleNamespace(image_token_index=None),
        language_model=object(),
        get_input_embeddings=get_input_embeddings,
    )
    runner = VLMBatchRunner(model, processor=object())
    ids, kwargs, salt = runner.prepare_media("<audio> hi", audio=["wave"])
    assert seen["audio"] == ["wave"] and seen["images"] is None
    assert ids.tolist() == [1, 2, 3]
    assert "input_features" in captured and "feature_attention_mask" in captured
    assert "input_features" in kwargs and "inputs_embeds" in kwargs
    assert salt is None  # no APC manager


OPEN, CLOSE, EOS = 700, 701, 2
PIECES = {
    OPEN: "",
    CLOSE: "",
    10: "thought",
    11: "\n",
    12: "sky is",
    13: " blue",
    14: "Blue.",
}


class _Detok:
    def __init__(self):
        self.last_segment = ""

    def reset(self):
        self.last_segment = ""

    def add_token(self, t):
        self.last_segment = PIECES[t]

    def finalize(self):
        self.last_segment = ""


class _GemmaTok:
    detokenizer = _Detok()

    def encode(self, text, add_special_tokens=False):
        # <think> is not a single token for Gemma: several pieces.
        return {"<think>": [50, 51], "</think>": [52, 51]}[text]


def test_gemma_channel_reasoning_split_and_label_dropped():
    eng = VLMEngine.__new__(VLMEngine)
    eng._tokenizer = _GemmaTok()
    eng._reasoning_channel_ids = (OPEN, CLOSE)
    eng._get_eos_ids = lambda: [EOS]
    passed = {}

    def iter_tokens(input_ids, stats, **kw):
        passed.update(kw)
        for t in (OPEN, 10, 11, 12, 13, CLOSE, 14, EOS):
            stats.generated += 1
            yield t

    eng._batch_runner = SimpleNamespace(iter_tokens=iter_tokens)
    events = list(
        eng._runner_events(
            [5, 6],
            max_tokens=32,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            seed=None,
            stop=None,
            stop_token_ids=None,
            repetition_penalty=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            logit_bias=None,
            json_schema=None,
            enable_thinking=True,
            thinking_budget=100,
            cancel_event=None,
            stats=RunStats(),
        )
    )
    reasoning = "".join(e[0] for e in events if e[2] == "reasoning")
    content = "".join(e[0] for e in events if e[2] == "normal")
    assert reasoning == "sky is blue"
    assert content == "Blue."
    assert events[-1][3] == "stop"
    # The budget uses the model's own markers.
    assert passed["thinking_start_token"] == "<|channel>"
    assert passed["thinking_end_token"] == "<channel|>"


def test_gemma4_thinking_defaults_off_in_both_entry_points():
    eng = VLMEngine.__new__(VLMEngine)
    eng._config = {"model_type": "gemma4"}
    eng._model_path = "/models/whatever"
    assert eng._default_enable_thinking(None) is False
    assert eng._default_enable_thinking(True) is True
    eng._config = {"model_type": "qwen3_5"}
    assert eng._default_enable_thinking(None) is None


def test_gateway_rejects_lora_and_logits_processors_for_vlm():
    import inspect

    from yunshu_gateway.routers import chat

    src = inspect.getsource(chat._handle_vlm_chat)
    assert "is not supported for multimodal" in src
    assert "status_code=400" in src
    assert "_apply_lora_adapter(vlm_engine" not in inspect.getsource(chat)


def test_restorable_sliding_window_family_gets_checkpoint_apc(monkeypatch):
    from mlx_vlm.models.cache import KVCache, RotatingKVCache

    eng = VLMEngine("/models/gemma-4-e2b-it-4bit")
    eng._config = {"model_type": "gemma4"}
    lm = SimpleNamespace(make_cache=lambda: [RotatingKVCache(max_size=8), KVCache()])
    eng._model = SimpleNamespace(language_model=lm)
    eng._processor = object()
    eng._mx_large_model = False
    monkeypatch.setenv("YUNSHU_VLM_APC_MEMORY_GB", "1")
    monkeypatch.setenv("YUNSHU_VLM_APC_DISK", "0")
    monkeypatch.setattr(eng, "_install_apc_identity", lambda lm: None)
    runner = eng._build_batch_runner(eng._model_path)
    assert runner.apc_manager is not None
    coordinator = runner.apc_manager.coordinator(lm)
    assert coordinator.is_checkpoint and coordinator.enabled
    runner.apc_manager.close()
    eng._executor.shutdown()


def test_media_salt_includes_geometry_and_audio_mask_but_not_text_length(monkeypatch):
    utils = importlib.import_module("mlx_vlm.utils")
    raw = {
        "input_ids": mx.array([[1, 2, 3]]),
        "pixel_values": mx.ones((16, 8)),
        "image_grid_thw": mx.array([[1, 4, 4]]),
        "input_features": mx.ones((1, 4, 8)),
        "feature_attention_mask": mx.array([[1, 1, 1, 0]]),
    }
    monkeypatch.setattr(utils, "prepare_inputs", lambda *a, **k: dict(raw))
    model = SimpleNamespace(
        config=SimpleNamespace(image_token_index=None),
        language_model=object(),
        get_input_embeddings=lambda *a, **k: SimpleNamespace(to_dict=lambda: {}),
    )
    runner = VLMBatchRunner(model, processor=object(), apc_manager=object())
    salt = runner.prepare_media("one turn")[2]
    assert runner.prepare_media("same image with additional text")[2] == salt
    raw["input_ids"] = mx.array([[1, 2, 3, 4]])
    assert runner.prepare_media("longer turn")[2] == salt
    raw["image_grid_thw"] = mx.array([[1, 2, 8]])
    assert runner.prepare_media("different grid, identical patch pixels")[2] != salt
    raw["image_grid_thw"] = mx.array([[1, 4, 4]])
    raw["feature_attention_mask"] = mx.array([[1, 1, 0, 0]])
    assert runner.prepare_media("different valid audio features")[2] != salt


def test_media_ids_include_wrapper_config_and_native_audio():
    from yunshu_engine.vlm_batch_runner import media_token_ids

    model = SimpleNamespace(
        config=SimpleNamespace(
            image_token_id=900, audio_token_id=901, video_token_id=902
        ),
        language_model=SimpleNamespace(config=SimpleNamespace()),
    )
    assert media_token_ids(model) == {900, 901, 902}
    native = SimpleNamespace(
        config=SimpleNamespace(
            thinker_config=SimpleNamespace(image_token_index=903, audio_token_index=904)
        )
    )
    assert media_token_ids(native) == {903, 904}


def test_image_cache_control_maps_wrapper_media_expansion(monkeypatch):
    eng = VLMEngine.__new__(VLMEngine)
    eng._model = SimpleNamespace(
        config=SimpleNamespace(image_token_id=900),
        language_model=SimpleNamespace(config=SimpleNamespace()),
    )
    eng._tokenizer = SimpleNamespace(encode=lambda *a, **k: [1, 900, 2, 3])
    eng._apply_vlm_template_with_cache = lambda *a, **k: "prompt"
    monkeypatch.setattr(
        "yunshu_engine.prompt_caching.rendered_boundaries",
        lambda *a, **k: {
            "points": [(2, 300)],
            "writes": [(2, 300)],
            "lookup_points": [2],
        },
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "file://fixture"}}],
        }
    ]
    plan = dict(markers={"MARK": 300}, tools={}, messages=messages)
    resolved = eng._resolve_prompt_cache_plan(
        plan, messages, [1, 900, 900, 2, 3], {"inputs_embeds": object()}, False, {}
    )
    assert resolved["points"] == [(3, 300)] and resolved["lookup_points"] == [3]


def test_image_cache_control_maps_processor_wrappers_and_skips_unsafe_head(monkeypatch):
    eng = VLMEngine.__new__(VLMEngine)
    eng._model = SimpleNamespace(
        config=SimpleNamespace(image_token_id=900),
        language_model=SimpleNamespace(config=SimpleNamespace()),
    )
    eng._tokenizer = SimpleNamespace(encode=lambda *a, **k: [1, 900, 2, 3])
    eng._processor = SimpleNamespace(
        image_token_id=900,
        boi_token="BEGIN",
        eoi_token="END",
        tokenizer=SimpleNamespace(
            encode=lambda text, **k: [901] if text == "BEGIN" else [902]
        ),
    )
    eng._apply_vlm_template_with_cache = lambda *a, **k: "prompt"
    monkeypatch.setattr(
        "yunshu_engine.prompt_caching.rendered_boundaries",
        lambda *a, **k: {
            "points": [(1, 300), (2, 300), (3, 300)],
            "writes": [(2, 300)],
            "lookup_points": [1, 2],
        },
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "file://fixture"}}],
        }
    ]
    plan = dict(markers={"MARK": 300}, tools={}, messages=messages)
    result = eng._resolve_prompt_cache_plan(
        plan,
        messages,
        [1, 901, 900, 900, 902, 2, 3],
        {"inputs_embeds": object()},
        False,
        {},
    )
    assert result["points"] == [(5, 300), (6, 300)]
    assert result["writes"] == [(5, 300)] and result["lookup_points"] == [5]
    # Unknown processor transformations never cause an inference 500 over a hint.
    assert (
        eng._resolve_prompt_cache_plan(
            plan,
            messages,
            [1, 999, 900, 902, 2, 3],
            {"inputs_embeds": object()},
            False,
            {},
        )
        is None
    )
