"""Local custom voices backed by the bounded Files store.

Enrollment records a consent recording; it does not verify speaker identity.
Speech uses the stored reference only with a model that supports ref_audio.
"""

from __future__ import annotations

import asyncio
import io
import json
import time

from fastapi import APIRouter, HTTPException, Request
from starlette.datastructures import UploadFile

from ..files_store import FileStoreError, get_store
from .models import _check_permission

router = APIRouter(tags=["audio"])
_LIMIT = 10 * 1024 * 1024
_MIMES = {
    "audio/mpeg",
    "audio/wav",
    "audio/x-wav",
    "audio/ogg",
    "audio/aac",
    "audio/flac",
    "audio/webm",
    "audio/mp4",
}


async def recording(value):
    if not isinstance(value, UploadFile) or value.content_type not in _MIMES:
        raise HTTPException(
            400, "Provide an audio recording with a supported MIME type"
        )
    data = await value.read(_LIMIT + 1)
    if not data or len(data) > _LIMIT:
        raise HTTPException(413, "Recording must be nonempty and at most 10 MiB")

    # Decode now on CPU, before enrollment, so unusable samples do not succeed.
    def decode():
        try:
            import soundfile as sf
        except ImportError as exc:
            raise HTTPException(503, "Voice enrollment requires yunshu[audio]") from exc

        try:
            with sf.SoundFile(io.BytesIO(data)) as source:
                rate = source.samplerate
                if (
                    not source.frames
                    or source.frames / rate > 60
                    or source.frames * source.channels * 2 > _LIMIT
                ):
                    raise ValueError("Decoded recording exceeds enrollment limits")
                samples = source.read(dtype="int16", always_2d=True)
            output = io.BytesIO()
            sf.write(output, samples, rate, format="WAV", subtype="PCM_16")
            if output.tell() > _LIMIT:
                raise ValueError("Decoded recording exceeds 10 MiB")
            return output.getvalue()
        except Exception as exc:
            raise HTTPException(
                400, "Cannot decode recording; use PCM WAV or a supported local codec"
            ) from exc

    return await asyncio.to_thread(decode)


def put_record(data, record):
    store = get_store()
    try:
        blob = store.put(
            data, "recording.wav", purpose="user_data", mime_type="audio/wav"
        )
        record["_recording"] = blob["id"]
        # JSON is a normal bounded file, so Files quota and TTL apply to both parts.
        meta = store.put(
            json.dumps(record).encode(),
            "voice.json",
            purpose="user_data",
            mime_type="application/vnd.yunshu.voice+json",
        )
    except FileStoreError as exc:
        if "blob" in locals():
            store.delete(blob["id"])
        raise HTTPException(exc.status, exc.message) from exc
    prefix = "cons_" if record["object"] == "audio.voice_consent" else "voice_"
    return {**record, "id": prefix + meta["id"][5:]}


def get_record(identifier, prefix):
    if not isinstance(identifier, str) or not identifier.startswith(prefix):
        raise HTTPException(404, "Local voice or consent not found")
    store = get_store()
    try:
        fid = "file_" + identifier[len(prefix) :]
        meta = store.get_meta(fid)
        if meta["mime_type"] != "application/vnd.yunshu.voice+json":
            raise ValueError("Not voice metadata")
        record = json.loads(store.read(fid))
        expected = "audio.voice_consent" if prefix == "cons_" else "audio.voice"
        if record["object"] != expected:
            raise ValueError("Wrong enrollment type")
        store.get_meta(record["_recording"])
        return {**record, "id": identifier}
    except (FileStoreError, ValueError, KeyError) as exc:
        raise HTTPException(404, "Local voice or consent not found or expired") from exc


def public(record):
    return {k: v for k, v in record.items() if not k.startswith("_")}


@router.post("/audio/voice_consents")
async def create_consent(request: Request):
    _check_permission(request, "can_infer")
    async with request.form() as form:
        name, language = form.get("name"), form.get("language")
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 256:
            raise HTTPException(400, "name must contain 1–256 characters")
        if not isinstance(language, str) or not language.strip() or len(language) > 64:
            raise HTTPException(400, "language is required")
        data = await recording(form.get("recording"))
    record = await asyncio.to_thread(
        put_record,
        data,
        {
            "object": "audio.voice_consent",
            "name": name,
            "language": language,
            "created_at": int(time.time()),
        },
    )
    return public(record)


@router.post("/audio/voices")
async def create_voice(request: Request):
    _check_permission(request, "can_infer")
    async with request.form() as form:
        if form.get("type", "audio_sample") != "audio_sample":
            raise HTTPException(
                400,
                "Prompt-designed OpenAI Live voices are unsupported; enroll an audio_sample for local TTS",
            )
        name = form.get("name")
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 256:
            raise HTTPException(400, "name must contain 1–256 characters")
        consent = get_record(form.get("consent"), "cons_")
        ref_text = form.get("ref_text")
        if ref_text is not None and (
            not isinstance(ref_text, str)
            or not ref_text.strip()
            or len(ref_text) > 32768
        ):
            raise HTTPException(
                400,
                "ref_text must be a nonempty transcript of at most 32768 characters",
            )
        data = await recording(form.get("audio_sample"))
    record = await asyncio.to_thread(
        put_record,
        data,
        {
            "object": "audio.voice",
            "name": name,
            "type": "audio_sample",
            "created_at": int(time.time()),
            "_consent": consent["id"],
            "_ref_text": ref_text,
        },
    )
    return public(record)


def resolve_voice(req, engine):
    if not isinstance(req.voice, dict):
        return req
    import inspect

    record = get_record(req.voice["id"], "voice_")
    get_record(record["_consent"], "cons_")
    model = getattr(engine, "_model", None)
    if model is None:
        raise HTTPException(503, "TTS model is not loaded")
    if "ref_audio" not in inspect.signature(model.generate).parameters:
        raise HTTPException(
            400,
            "This TTS model cannot clone voices; choose a model with ref_audio support or a built-in voice",
        )
    ref_text = req.ref_text or record.get("_ref_text")
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) == "qwen3_tts":
        if getattr(config, "tts_model_type", "base") != "base":
            raise HTTPException(
                400,
                "Qwen3-TTS CustomVoice/VoiceDesign ignore reference cloning; choose a Base model",
            )
        if not ref_text:
            raise HTTPException(
                400,
                "Qwen3-TTS Base requires ref_text; provide the sample transcript at enrollment or in the speech request",
            )
    if req.ref_audio:
        raise HTTPException(400, "Custom voice and ref_audio cannot be combined")
    path = str(get_store().blob_path(record["_recording"]))
    return req.model_copy(
        update={"voice": "alloy", "ref_audio": path, "ref_text": ref_text}
    )


def list_custom_voices():
    records = []
    for meta in get_store().list(purpose="user_data"):
        if meta["mime_type"] != "application/vnd.yunshu.voice+json":
            continue
        identifier = "voice_" + meta["id"][5:]
        try:
            record = get_record(identifier, "voice_")
            get_record(record["_consent"], "cons_")
            records.append(public(record))
        except HTTPException:
            continue
    return records
