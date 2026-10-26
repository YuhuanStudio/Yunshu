"""mlx-audio fixes carried after the 0.5.9 pin (dbcc13c, 2cd66c6, ebcdbfc). CPU-only, no weights."""

import inspect
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from yunshu_engine import audio_patches

whisper = pytest.importorskip("mlx_audio.stt.models.whisper.whisper")
qwen3 = pytest.importorskip("mlx_audio.stt.models.qwen3_asr.qwen3_asr")
kokoro_pipeline = pytest.importorskip("mlx_audio.tts.models.kokoro.pipeline")
kokoro_model = pytest.importorskip("mlx_audio.tts.models.kokoro.kokoro")


class _Tok:
    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return self._vocab

    def convert_tokens_to_ids(self, name):
        return self._vocab.get(name, 50257)  # unk == eot


@pytest.fixture
def whisper_restored(monkeypatch):
    monkeypatch.setattr(whisper.HFTokenizerWrapper, "no_speech", whisper.HFTokenizerWrapper.no_speech)
    return whisper.HFTokenizerWrapper


@pytest.mark.parametrize(
    "vocab, expected",
    [
        ({"<|nocaptions|>": 50362}, 50362),  # pre-large-v3
        ({"<|nospeech|>": 50363}, 50363),  # large-v3
        ({"<|startoftranscript|>": 1}, None),
    ],
)
def test_whisper_no_speech_resolves_either_name_or_none(whisper_restored, vocab, expected):
    cls = whisper_restored
    stock_has_fix = "<|nocaptions|>" in inspect.getsource(cls.__dict__["no_speech"].fget)
    assert audio_patches.patch_whisper_no_speech() is not stock_has_fix
    wrapper = cls(_Tok(vocab))
    assert wrapper.no_speech == expected
    assert audio_patches.patch_whisper_no_speech() is False  # idempotent


def test_whisper_stock_defect_is_what_the_patch_fixes(whisper_restored):
    cls = whisper_restored
    if "<|nocaptions|>" in inspect.getsource(cls.__dict__["no_speech"].fget):
        pytest.skip("installed mlx-audio already carries 2cd66c6")
    # stock returns unk, i.e. eot, for a pre-large-v3 vocabulary
    assert cls(_Tok({"<|nocaptions|>": 50362})).no_speech == 50257


class _FakeQwen:
    sample_rate = 100
    _tokenizer = object()
    _feature_extractor = object()
    extract_language = qwen3.Qwen3ASRModel.extract_language

    def __init__(self):
        self.languages_seen = []
        self.replies = ["language English<asr_text>hello", "language Chinese<asr_text>ni hao"]

    def _generate_single_chunk(self, chunk_audio, *, language, **kwargs):
        self.languages_seen.append(language)
        text = self.replies[len(self.languages_seen) - 1] if language is None else "forced"
        return text, 5, 3


def _qwen_fixed(fn):
    """Patched by Yunshu (earlier in this process) or fixed upstream."""
    if getattr(fn, "_yunshu_chunk_language", False):
        return True
    return "chunk_language" in inspect.getsource(fn)


@pytest.fixture
def qwen_restored(monkeypatch):
    monkeypatch.setattr(qwen3.Qwen3ASRModel, "_generate_one", qwen3.Qwen3ASRModel._generate_one)


def _run(fn):
    audio = np.random.default_rng(0).normal(size=1200).astype(np.float32)  # 12 s at 100 Hz
    model = _FakeQwen()
    out = fn(model, audio, chunk_duration=6.0, min_chunk_duration=1.0, language=None)
    return model, out


def test_qwen3_language_is_detected_per_chunk(qwen_restored):
    stock = qwen3.Qwen3ASRModel._generate_one
    already_fixed = _qwen_fixed(stock)
    assert audio_patches.patch_qwen3_asr_chunk_language() is not already_fixed
    model, out = _run(qwen3.Qwen3ASRModel._generate_one)
    assert model.languages_seen == [None, None]  # the request language stays unset for every chunk
    assert [s["language"] for s in out.segments] == ["English", "Chinese"]
    assert out.text == "hello ni hao"
    assert out.language == ["English", "Chinese"]
    assert audio_patches.patch_qwen3_asr_chunk_language() is False


def test_qwen3_stock_defect_is_what_the_patch_fixes(qwen_restored):
    stock = qwen3.Qwen3ASRModel._generate_one
    if _qwen_fixed(stock):
        pytest.skip("already fixed (upstream or patched earlier in this process)")
    model, out = _run(stock)
    assert model.languages_seen == [None, "English"]  # first chunk's detection forced on the next


def test_qwen3_rewrite_refuses_unrecognised_source():
    assert audio_patches.rewrite_qwen3_generate_one("def _generate_one(self):\n    pass\n") is None
    stock = qwen3.Qwen3ASRModel._generate_one
    if not _qwen_fixed(stock):
        fixed = audio_patches.rewrite_qwen3_generate_one(inspect.getsource(stock))
        assert fixed is not None and "chunk_language" in fixed
        assert audio_patches.rewrite_qwen3_generate_one(fixed) is None  # already fixed


@pytest.fixture
def kokoro_restored(monkeypatch):
    monkeypatch.setattr(kokoro_pipeline.KokoroPipeline, "load_single_voice", kokoro_pipeline.KokoroPipeline.load_single_voice)
    had = hasattr(kokoro_model.Model, "post_load_hook")
    yield
    if not had and hasattr(kokoro_model.Model, "post_load_hook"):
        del kokoro_model.Model.post_load_hook


def test_kokoro_reads_voices_from_the_loaded_snapshot(kokoro_restored, tmp_path, monkeypatch):
    if hasattr(kokoro_model.Model, "post_load_hook"):
        pytest.skip("installed mlx-audio already carries ebcdbfc")
    (tmp_path / "voices").mkdir()
    pack = mx.arange(6, dtype=mx.float32).reshape(2, 1, 3)
    mx.save_safetensors(str(tmp_path / "voices" / "af_local.safetensors"), {"voice": pack})
    assert audio_patches.patch_kokoro_local_voices() is True
    model = kokoro_model.Model.post_load_hook(SimpleNamespace(), tmp_path)
    assert model._voice_model_path == tmp_path

    pipeline = SimpleNamespace(voices={}, model=model, repo_id="some/other-repo", lang_code="a")
    # a local voice must never reach the hub download path of the stock function
    monkeypatch.setattr(kokoro_pipeline, "snapshot_download", lambda *a, **k: pytest.fail("hub download"))
    got = kokoro_pipeline.KokoroPipeline.load_single_voice(pipeline, "af_local")
    assert mx.array_equal(got, pack).item()
    assert pipeline.voices["af_local"] is got
    # a voice that is not in the snapshot falls through to the stock lookup
    with pytest.raises(BaseException):
        kokoro_pipeline.KokoroPipeline.load_single_voice(pipeline, "af_missing")
    assert audio_patches.patch_kokoro_local_voices() is False
