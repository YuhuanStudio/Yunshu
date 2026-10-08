"""Native MLX model management behind Ollama's request/response shapes.

Copies are persistent symlink names: weights are never duplicated. Pull accepts
Hugging Face MLX repositories; GGUF and Ollama registry blobs are unsupported.
"""

from __future__ import annotations

import asyncio
import shutil
import threading
from pathlib import Path
from urllib.parse import quote

from fastapi import HTTPException

from yunshu_engine import paths

from .engine import get_model_manager

_LOCK = asyncio.Lock()
_DOWNLOADS: dict[str, threading.Event] = {}


def cancel_download(name: str) -> bool:
    event = _DOWNLOADS.get(name)
    if event is None:
        return False
    event.set()
    return True


_PREFIX = ".ollama--"


def model_link(name: str) -> Path:
    if (
        not isinstance(name, str)
        or not name.strip()
        or name.startswith("/")
        or any(p in ("", ".", "..") for p in name.split("/"))
        or "\\" in name
        or "\x00" in name
    ):
        raise HTTPException(400, "invalid model name")
    return paths.models_dir() / (_PREFIX + quote(name, safe=""))


def manager_for(request, permission):
    from .routers.models import _check_permission

    _check_permission(request, permission)
    manager = get_model_manager()
    if manager is None:
        raise HTTPException(
            400, "Model management requires --models-dir (multi-model mode)"
        )
    return manager


def registered_entry(manager, name):
    # Inference wildcard aliases are not model names and must not authorize
    # deleting an unrelated checkpoint or make an unpulled repository look present.
    for wanted in (name, name.removesuffix(":latest")):
        for entry in manager.list_entries():
            if entry.model_id.lower() == wanted.lower():
                return entry
    return None


def entry_for(manager, name):
    entry = registered_entry(manager, name)
    if entry is None:
        raise HTTPException(404, f"model '{name}' not found")
    return entry


async def copy_model(request, source: str, destination: str):
    manager = manager_for(request, "can_load_models")
    target = model_link(destination)
    async with _LOCK:
        entry = entry_for(manager, source)
        if (
            registered_entry(manager, destination) is not None
            or target.exists()
            or target.is_symlink()
        ):
            raise HTTPException(409, f"model '{destination}' already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(Path(entry.model_path).resolve(), target_is_directory=True)
        try:
            manager.register_model(
                destination,
                str(target),
                estimated_bytes=entry.estimated_bytes,
                model_type=entry.model_type,
            )
        except Exception:
            target.unlink()
            raise


async def delete_model(request, name: str):
    manager = manager_for(request, "can_unload_models")
    async with _LOCK:
        entry = entry_for(manager, name)
        target = Path(entry.model_path)
        root = paths.models_dir().resolve()
        # Delete only names/data in the configured model directory. A symlink is
        # unlinked, never followed into an external checkpoint or HF cache.
        if not target.parent.resolve().is_relative_to(root) or target == root:
            raise HTTPException(
                400, "This model is outside the configured models directory"
            )
        if not target.is_symlink():
            if not target.resolve().is_relative_to(root):
                raise HTTPException(
                    400, "Model path escapes the configured models directory"
                )
            if any(
                e.model_id != entry.model_id
                and Path(e.model_path).resolve() == target.resolve()
                for e in manager.list_entries()
            ):
                raise HTTPException(
                    409, "Delete model copies referring to this checkpoint first"
                )
        if entry.is_loaded and not await manager.unload_model(entry.model_id):
            raise HTTPException(409, f"model '{name}' is in use")
        try:
            manager.unregister_model(entry.model_id)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(409, str(exc)) from exc
        if target.is_symlink():
            target.unlink()
        else:
            await asyncio.to_thread(shutil.rmtree, target)


async def pull_model(request, name: str):
    manager = manager_for(request, "can_load_models")
    target = model_link(name)
    async with _LOCK:
        if registered_entry(manager, name):
            return
        if len(name.split("/")) != 2 or ":" in name:
            raise HTTPException(
                400,
                "Use a Hugging Face MLX repository id (org/name); Ollama registry / GGUF models are unsupported",
            )

        from huggingface_hub import snapshot_download
        from tqdm.auto import tqdm

        cancelled = threading.Event()
        _DOWNLOADS[name] = cancelled

        class CancellableProgress(tqdm):
            def update(self, n=1):
                if cancelled.is_set():
                    raise RuntimeError("Download cancelled")
                return super().update(n)

        try:
            snapshot = Path(
                await asyncio.to_thread(
                    snapshot_download, repo_id=name, tqdm_class=CancellableProgress
                )
            )
            if cancelled.is_set():
                raise RuntimeError("Download cancelled")
        except Exception as exc:
            raise HTTPException(400, f"model '{name}' download failed") from exc
        finally:
            _DOWNLOADS.pop(name, None)
        if not (snapshot / "config.json").is_file() or not any(
            snapshot.glob("*.safetensors")
        ):
            raise HTTPException(
                400,
                "Repository is not a native safetensors model; GGUF conversion is unsupported",
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise HTTPException(409, f"model '{name}' already exists on disk")
        target.symlink_to(snapshot.resolve(), target_is_directory=True)
        try:
            manager.register_model(name, str(target))
        except Exception:
            target.unlink()
            raise
