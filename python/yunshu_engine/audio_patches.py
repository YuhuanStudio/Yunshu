# Upstream (derived): Blaizzy/mlx-audio (MIT) dbcc13c, 2cd66c6, ebcdbfc (fixes after the pinned 0.5.9)
# Patches upstream mlx-audio symbols (see vendor.json kind=patches; `just vendor-check` flags source changes)
"""Small fixes from mlx-audio after the pinned 0.5.9, applied until the pin moves.

mlx-audio 0.6 needs mlx >= 0.32.4, which this repo does not take yet, so the fixes Yunshu's own
code paths reach are carried here. Each patch checks that the installed code still has the defect
(so it turns itself off once a bump includes the fix) and is idempotent.

- 2cd66c6 Whisper ``no_speech``: pre-large-v3 vocabularies (tiny .. large-v2, the default
  ``whisper-small`` included) call the token ``<|nocaptions|>``; the HF wrapper looked up
  ``<|nospeech|>`` and got unk (== eot), so eot landed in the suppress list and decoding never
  stopped.
- ebcdbfc Kokoro voices from the loaded snapshot: a local model directory (or a pinned revision)
  keeps its own ``voices/*.safetensors``; stock reads them from the hub repo instead.
- dbcc13c Qwen3-ASR per-chunk language: with ``language=None`` the first chunk's detection
  replaced the request language, forcing it on every later chunk of audio longer than the chunk
  duration (20 min). Applied as an exact source rewrite of ``Qwen3ASRModel._generate_one``.

Not ported, because Yunshu never reaches them: f37c9a9 (Qwen3-ASR batched token history) needs
``batch_size > 1`` together with a repetition penalty; no Yunshu route passes ``batch_size``.
``stream_transcribe`` (the streaming half of dbcc13c) is not called either.
"""

from __future__ import annotations

import inspect
import logging
import textwrap
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_QWEN3_REWRITES = (
    (
        "                if language is None:\n"
        "                    language, text = self.extract_language(text)\n"
        "                all_texts.append(text)\n",
        "                chunk_language = language\n"
        "                if language is None:\n"
        "                    chunk_language, text = self.extract_language(text)\n"
        "                all_texts.append(text)\n",
    ),
    (
        '                        "language": language,\n'
        '                        "start": offset_sec,\n'
        '                        "end": offset_sec + len(chunk_audio) / self.sample_rate,\n',
        '                        "language": chunk_language,\n'
        '                        "start": offset_sec,\n'
        '                        "end": offset_sec + len(chunk_audio) / self.sample_rate,\n',
    ),
    (
        "            if language is None:\n"
        "                language, text = self.extract_language(text)\n"
        "\n"
        "            all_texts.append(text)\n",
        "            # Detect per chunk; the request's language must stay unset so that\n"
        "            # later chunks are auto-detected too.\n"
        "            chunk_language = language\n"
        "            if language is None:\n"
        "                chunk_language, text = self.extract_language(text)\n"
        "\n"
        "            all_texts.append(text)\n",
    ),
    (
        '                    "language": language,\n'
        '                    "start": offset_sec,\n'
        '                    "end": offset_sec + actual_chunk_duration,\n',
        '                    "language": chunk_language,\n'
        '                    "start": offset_sec,\n'
        '                    "end": offset_sec + actual_chunk_duration,\n',
    ),
)


def rewrite_qwen3_generate_one(source: str) -> str | None:
    """``_generate_one`` source with the per-chunk language fix, or None if the defect is absent
    or the source no longer matches (each snippet must occur exactly once)."""
    if "chunk_language" in source:
        return None
    for old, _ in _QWEN3_REWRITES:
        if source.count(old) != 1:
            return None
    for old, new in _QWEN3_REWRITES:
        source = source.replace(old, new)
    return source


def patch_qwen3_asr_chunk_language() -> bool:
    try:
        from mlx_audio.stt.models.qwen3_asr import qwen3_asr as module
    except ImportError:
        return False
    cls = module.Qwen3ASRModel
    original = cls._generate_one
    if getattr(original, "_yunshu_chunk_language", False):
        return False
    source = rewrite_qwen3_generate_one(inspect.getsource(original))  # class-body indentation
    if source is None:
        return False
    source = textwrap.dedent(source)
    namespace: dict[str, Any] = {}
    exec(compile(source, f"<yunshu patch of {module.__name__}._generate_one>", "exec"), vars(module), namespace)
    patched = namespace["_generate_one"]
    patched._yunshu_chunk_language = True  # type: ignore[attr-defined]
    cls._generate_one = patched
    return True


def patch_whisper_no_speech() -> bool:
    try:
        from mlx_audio.stt.models.whisper import whisper as module
    except ImportError:
        return False
    cls = module.HFTokenizerWrapper
    current = cls.__dict__.get("no_speech")
    if current is None or "<|nocaptions|>" in inspect.getsource(current.fget):
        return False

    def no_speech(self):
        """No speech token (named <|nocaptions|> in pre-large-v3 vocabularies)."""
        # convert_tokens_to_ids returns unk (== eot) for a missing token, which would put eot in
        # the suppress list and decoding would never stop.
        vocab = self.hf_tokenizer.get_vocab()
        for name in ("<|nospeech|>", "<|nocaptions|>"):
            if name in vocab:
                return vocab[name]
        return None

    cls.no_speech = property(no_speech)
    return True


def _local_voice(model: Any, voice: str) -> str | None:
    root = getattr(model, "_voice_model_path", None)
    if root is None:
        return None
    path = Path(root) / "voices" / f"{voice}.safetensors"
    return str(path) if path.is_file() else None


def patch_kokoro_local_voices() -> bool:
    try:
        from mlx_audio.tts.models.kokoro import kokoro as kokoro_module
        from mlx_audio.tts.models.kokoro import pipeline as pipeline_module
    except ImportError:
        return False
    model_cls = kokoro_module.Model
    if hasattr(model_cls, "post_load_hook"):
        return False
    pipeline_cls = pipeline_module.KokoroPipeline
    stock = pipeline_cls.load_single_voice

    def post_load_hook(model, model_path):
        model._voice_model_path = Path(model_path)
        return model

    def load_single_voice(self, voice: str):
        if voice not in self.voices and not voice.endswith(".safetensors"):
            local = _local_voice(self.model, voice)
            if local is not None:
                # The same local snapshot as the weights, including a pinned revision.
                self.voices[voice] = pipeline_module.load_voice_tensor(local)
                return self.voices[voice]
        return stock(self, voice)

    model_cls.post_load_hook = staticmethod(post_load_hook)
    pipeline_cls.load_single_voice = load_single_voice
    return True


def install_stt() -> dict[str, bool]:
    return {
        "whisper_no_speech": patch_whisper_no_speech(),
        "qwen3_asr_chunk_language": patch_qwen3_asr_chunk_language(),
    }


def install_tts() -> dict[str, bool]:
    return {"kokoro_local_voices": patch_kokoro_local_voices()}
