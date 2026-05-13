"""Language Identification (LID) module.

Detects the language of audio input using simple phonetic and spectral
features. For production use, this would integrate with a trained model
(e.g., Silero LID, Whisper's language detection).

Used by:
- ASR pipeline to auto-select language
- Realtime API for multilingual routing
- Audio preprocessing for model selection
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Common language patterns for text-based detection
_LANG_PATTERNS: dict[str, list[str]] = {
    "zh": [
        r"[一-鿿]",  # CJK Unified Ideographs
    ],
    "ja": [
        r"[぀-ゟ]",  # Hiragana
        r"[゠-ヿ]",  # Katakana
    ],
    "ko": [
        r"[가-힯]",  # Hangul Syllables
    ],
    "ar": [
        r"[؀-ۿ]",  # Arabic
    ],
    "ru": [
        r"[Ѐ-ӿ]",  # Cyrillic
    ],
    "th": [
        r"[฀-๿]",  # Thai
    ],
    "hi": [
        r"[ऀ-ॿ]",  # Devanagari
    ],
}

# Latin-script languages detected by common words
_LATIN_LANG_HINTS: dict[str, list[str]] = {
    "en": ["the", "is", "are", "and", "this", "that", "with", "for"],
    "fr": ["le", "la", "les", "de", "des", "est", "une", "dans"],
    "de": ["der", "die", "das", "und", "ist", "ein", "nicht", "mit"],
    "es": ["el", "la", "los", "las", "de", "en", "es", "por"],
    "pt": ["o", "a", "os", "as", "de", "em", "um", "uma"],
    "it": ["il", "la", "lo", "le", "di", "in", "che", "per"],
    "nl": ["de", "het", "een", "van", "in", "is", "dat", "op"],
}


@dataclass
class LIDResult:
    language: str
    confidence: float
    all_scores: dict[str, float]


def detect_language(text: str) -> LIDResult:
    """Detect language from text using character and word patterns.

    Fast, no ML required. Uses Unicode character ranges for CJK/Arabic/Cyrillic
    and common word matching for Latin-script languages.

    Args:
        text: Input text to identify.

    Returns:
        LIDResult with detected language, confidence, and all scores.
    """
    if not text or not text.strip():
        return LIDResult(language="und", confidence=0.0, all_scores={})

    scores: dict[str, float] = {}

    # Check Unicode patterns
    for lang, patterns in _LANG_PATTERNS.items():
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, text))
        if count > 0:
            scores[lang] = count / len(text)

    # For Latin-script text, check common words
    if not scores or max(scores.values()) < 0.1:
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if words:
            word_set = set(words)
            for lang, hints in _LATIN_LANG_HINTS.items():
                matches = sum(1 for h in hints if h in word_set)
                if matches > 0:
                    scores[lang] = matches / len(hints)

    if not scores:
        return LIDResult(language="und", confidence=0.0, all_scores={})

    # Sort by score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_lang = sorted_scores[0][0]
    best_score = sorted_scores[0][1]

    # Compute confidence as ratio of best to second-best
    if len(sorted_scores) > 1:
        second_score = sorted_scores[1][1]
        confidence = best_score / (best_score + second_score) if (best_score + second_score) > 0 else 1.0
    else:
        confidence = 1.0

    return LIDResult(
        language=best_lang,
        confidence=round(confidence, 3),
        all_scores={lang: round(score, 3) for lang, score in sorted_scores},
    )


def detect_language_from_audio(audio_bytes: bytes, sample_rate: int = 16000) -> LIDResult:
    """Detect language from audio using simple spectral features.

    For now, returns 'und' (undetermined). In production, this would
    integrate with a trained LID model (e.g., Silero LID).
    """
    # Placeholder: would use spectral analysis or trained model
    return LIDResult(language="und", confidence=0.0, all_scores={})
