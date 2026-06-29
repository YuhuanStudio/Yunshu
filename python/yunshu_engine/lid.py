"""Language Identification (LID) module.

Detects the language of audio input using simple phonetic and spectral
features. For production use, this would integrate with a trained model
(e.g., Silero LID, Whisper's language detection).

Used by:
- ASR pipeline to auto-select language
- Realtime API for multilingual routing
- Audio preprocessing for model selection
"""

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
        confidence = (
            best_score / (best_score + second_score)
            if (best_score + second_score) > 0
            else 1.0
        )
    else:
        confidence = 1.0

    return LIDResult(
        language=best_lang,
        confidence=round(confidence, 3),
        all_scores={lang: round(score, 3) for lang, score in sorted_scores},
    )


def detect_language_from_audio(
    audio_bytes: bytes, sample_rate: int = 16000
) -> LIDResult:
    """Detect language from audio using simple spectral heuristics.

    Uses energy distribution across frequency bands as a rough proxy.
    Tonal languages (zh, th, vi) tend to have more energy in higher formants.
    For accurate detection, integrate a trained LID model (Silero, Whisper).
    """
    if not audio_bytes or len(audio_bytes) < 2:
        return LIDResult(language="und", confidence=0.0, all_scores={})

    try:
        import numpy as np

        samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        if len(samples) < sample_rate // 4:
            return LIDResult(language="und", confidence=0.0, all_scores={})

        # Simple spectral energy ratio: low (0-1kHz) vs mid (1-4kHz) vs high (4-8kHz)
        n = len(samples)
        fft = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

        low_mask = freqs < 1000
        mid_mask = (freqs >= 1000) & (freqs < 4000)
        high_mask = (freqs >= 4000) & (freqs < 8000)

        low_energy = np.sum(fft[low_mask]) if np.any(low_mask) else 0.0
        mid_energy = np.sum(fft[mid_mask]) if np.any(mid_mask) else 0.0
        high_energy = np.sum(fft[high_mask]) if np.any(high_mask) else 0.0
        total = low_energy + mid_energy + high_energy

        if total == 0:
            return LIDResult(language="und", confidence=0.0, all_scores={})

        mid_ratio = mid_energy / total
        high_ratio = high_energy / total

        scores: dict[str, float] = {}

        # Tonal languages tend to have higher mid-frequency energy
        if mid_ratio > 0.4:
            scores["zh"] = mid_ratio * 0.6
            scores["th"] = mid_ratio * 0.3
            scores["vi"] = mid_ratio * 0.2

        # Languages with more high-frequency energy (consonant-heavy)
        if high_ratio > 0.15:
            scores["ja"] = high_ratio * 0.5
            scores["ko"] = high_ratio * 0.4

        # Default: likely Latin-script language (en, fr, de, es)
        if not scores or max(scores.values()) < 0.1:
            scores["en"] = 0.15
            scores["fr"] = 0.10
            scores["de"] = 0.08
            scores["es"] = 0.08

        if not scores:
            return LIDResult(language="und", confidence=0.0, all_scores={})

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_lang = sorted_scores[0][0]
        best_score = sorted_scores[0][1]

        if len(sorted_scores) > 1:
            second_score = sorted_scores[1][1]
            confidence = (
                best_score / (best_score + second_score)
                if (best_score + second_score) > 0
                else 1.0
            )
        else:
            confidence = 0.5

        return LIDResult(
            language=best_lang,
            confidence=round(min(confidence, 0.7), 3),
            all_scores={lang: round(score, 3) for lang, score in sorted_scores},
        )
    except Exception:
        logger.debug("audio LID failed", exc_info=True)
        return LIDResult(language="und", confidence=0.0, all_scores={})
