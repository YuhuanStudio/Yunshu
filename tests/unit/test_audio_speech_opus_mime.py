"""(MED): /v1/audio/speech response_format=opus returned Ogg-Opus bytes but labeled
them Content-Type: audio/opus.

The opus path transcodes with `-f ogg -c:a libopus`, producing an Ogg-ENCAPSULATED Opus stream
(magic "OggS"). The MIME for that container is audio/ogg (what OpenAI's /audio/speech returns);
"audio/opus" denotes raw/CMAF Opus packets. Strict clients keying on Content-Type mis-handled
the Ogg bytes. Fixed the single _AUDIO_MIME["opus"] mapping to audio/ogg.
"""

from __future__ import annotations

from yunshu_gateway.routers.audio import _AUDIO_MIME, _FFMPEG_FMT


def test_opus_mime_matches_its_ogg_container():
    # the transcode uses the Ogg muxer (-f ogg) → the Content-Type must be audio/ogg
    assert "-f" in _FFMPEG_FMT["opus"] and "ogg" in _FFMPEG_FMT["opus"]
    assert _AUDIO_MIME["opus"] == "audio/ogg"


def test_other_format_mimes_unchanged():
    # the rest of the OpenAI format→MIME table is correct and must stay so
    assert _AUDIO_MIME["mp3"] == "audio/mpeg"
    assert _AUDIO_MIME["aac"] == "audio/aac"
    assert _AUDIO_MIME["flac"] == "audio/flac"
    assert _AUDIO_MIME["wav"] == "audio/wav"
