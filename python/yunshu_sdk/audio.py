"""Yunshu SDK — Audio namespace (TTS + ASR)."""

from __future__ import annotations

from typing import Optional

import httpx


class AudioSpeech:
    """TTS result."""

    def __init__(self, data: bytes, content_type: str = "audio/wav"):
        self.data = data
        self.content_type = content_type

    def save(self, path: str):
        with open(path, "wb") as f:
            f.write(self.data)


class AudioTranscription:
    """ASR result."""

    def __init__(self, data: dict):
        self.text = data.get("text", "")
        self.language = data.get("language", "")
        self.duration = data.get("duration", 0.0)
        self.segments = data.get("segments", [])


class _Speech:
    def __init__(self, http: httpx.Client):
        self._http = http

    def create(
        self,
        model: str,
        input: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: str = "wav",
        **kwargs,
    ) -> AudioSpeech:
        payload = {
            "model": model,
            "input": input,
            "voice": voice,
            "speed": speed,
            "response_format": response_format,
        }
        payload.update(kwargs)
        resp = self._http.post("/v1/audio/speech", json=payload)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "audio/wav")
        return AudioSpeech(resp.content, content_type)


class _Transcriptions:
    def __init__(self, http: httpx.Client):
        self._http = http

    def create(
        self,
        model: str,
        file: Optional[bytes] = None,
        file_path: Optional[str] = None,
        language: Optional[str] = None,
    ) -> AudioTranscription:
        files = {}
        data = {"model": model}
        if language:
            data["language"] = language

        if file_path:
            with open(file_path, "rb") as f:
                files["file"] = (file_path, f.read(), "audio/wav")
        elif file:
            files["file"] = ("audio.wav", file, "audio/wav")

        resp = self._http.post(
            "/v1/audio/transcriptions",
            data=data,
            files=files,
        )
        resp.raise_for_status()
        return AudioTranscription(resp.json())


class Audio:
    """Audio namespace (client.audio)."""

    def __init__(self, http: httpx.Client):
        self.speech = _Speech(http)
        self.transcriptions = _Transcriptions(http)
