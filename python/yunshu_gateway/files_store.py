"""Local file store behind the Files and Batch APIs, plus file-reference helpers.

Layout under the store root (default ``~/.yunshu/files``)::

    blobs/<id>          raw bytes
    meta/<id>.json      metadata (one JSON object per file)
    batches/<id>.json   batch record; ``<id>.in.jsonl`` normalized input,
                        ``<id>.out.jsonl`` / ``<id>.err.jsonl`` partial outputs

Single-consumer model: whoever presents the static token owns every file.
Ids are validated against a strict pattern before they ever touch a path, so
traversal is impossible. All writes are atomic (temp file + ``os.replace``).
"""

from __future__ import annotations

import base64
import contextlib
import copy
import json
import mimetypes
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from yunshu_engine import settings

_ID_RE = re.compile(r"^(file|batch|msgbatch)[_-][0-9a-f]{24}$")
_lock = threading.RLock()


class FileStoreError(Exception):
    """Store-level error carrying an HTTP status."""

    def __init__(self, status: int, message: str, code: str | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code


class FileNotFound(FileStoreError):  # noqa: N818
    def __init__(self, file_id: str):
        super().__init__(404, f"No such file: '{file_id}'", "file_not_found")


class FileTooLarge(FileStoreError):  # noqa: N818
    def __init__(self, limit: int):
        super().__init__(413, f"File exceeds the {limit} byte limit", "file_too_large")


def new_id(prefix: str = "file") -> str:
    return f"{prefix}_{secrets.token_hex(12)}"


def valid_id(value: str) -> bool:
    return isinstance(value, str) and bool(_ID_RE.match(value))


def normalize_id(value: str) -> str:
    """Accept the OpenAI-style ``file-...`` spelling for our ``file_...`` ids."""
    if isinstance(value, str) and value.startswith("file-"):
        return "file_" + value[5:]
    return value


def guess_mime(
    filename: str | None = None, data: bytes | None = None, declared: str | None = None
) -> str:
    """Best-effort content type: declared (if specific) > magic bytes > name."""
    if declared and declared not in ("application/octet-stream", "binary/octet-stream"):
        return declared.split(";")[0].strip()
    if data:
        head = data[:16]
        if head.startswith(b"%PDF"):
            return "application/pdf"
        if head.startswith(b"\x89PNG"):
            return "image/png"
        if head.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if head.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if head[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"
    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed:
            return guessed
        if filename.lower().endswith((".jsonl", ".ndjson")):
            return "application/x-jsonl"
    return "application/octet-stream"


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class FileStore:
    def __init__(self, root: Path, max_bytes: int, ttl_days: float | None = None):
        self.root = Path(root)
        self.max_bytes = int(max_bytes)
        self.ttl_days = ttl_days
        for sub in ("blobs", "meta", "batches"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    # -- paths -------------------------------------------------------------
    def _check(self, file_id: str) -> str:
        file_id = normalize_id(file_id)
        if not valid_id(file_id):
            raise FileNotFound(str(file_id)[:80])
        return file_id

    def _blob(self, fid: str) -> Path:
        return self.root / "blobs" / fid

    def _meta(self, fid: str) -> Path:
        return self.root / "meta" / f"{fid}.json"

    # -- files -------------------------------------------------------------
    def put(
        self,
        data: bytes,
        filename: str,
        purpose: str = "user_data",
        mime_type: str | None = None,
        expires_after: int | None = None,
        downloadable: bool = False,
    ) -> dict[str, Any]:
        if len(data) > self.max_bytes:
            raise FileTooLarge(self.max_bytes)
        now = int(time.time())
        expires_at = None
        if expires_after is not None:
            expires_at = now + int(expires_after)
        elif self.ttl_days:
            expires_at = now + int(self.ttl_days * 86400)
        fid = new_id("file")
        meta = {
            "id": fid,
            "filename": os.path.basename(filename or "") or "file",
            "purpose": purpose,
            "bytes": len(data),
            "mime_type": guess_mime(filename, data, mime_type),
            "created_at": now,
            "expires_at": expires_at,
            "downloadable": bool(downloadable),
        }
        with _lock:
            _atomic_write(self._blob(fid), data)
            _atomic_write(self._meta(fid), json.dumps(meta).encode())
        return meta

    def get_meta(self, file_id: str) -> dict[str, Any]:
        fid = self._check(file_id)
        try:
            meta = json.loads(self._meta(fid).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            raise FileNotFound(file_id) from None
        if meta.get("expires_at") and meta["expires_at"] <= time.time():
            self.delete(fid)
            raise FileNotFound(file_id)
        return meta

    def read(self, file_id: str) -> bytes:
        meta = self.get_meta(file_id)
        try:
            return self._blob(meta["id"]).read_bytes()
        except FileNotFoundError:
            raise FileNotFound(file_id) from None

    def blob_path(self, file_id: str) -> Path:
        return self._blob(self.get_meta(file_id)["id"])

    def delete(self, file_id: str) -> None:
        fid = self._check(file_id)
        with _lock:
            m = self._meta(fid)
            if not m.exists():
                raise FileNotFound(file_id)
            self._blob(fid).unlink(missing_ok=True)
            m.unlink(missing_ok=True)

    def list(self, purpose: str | None = None) -> list[dict[str, Any]]:
        """All live files, newest first (ties broken by id for stable paging)."""
        out = []
        for p in (self.root / "meta").glob("file_*.json"):
            try:
                meta = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if meta.get("expires_at") and meta["expires_at"] <= time.time():
                with contextlib.suppress(FileStoreError):
                    self.delete(meta["id"])
                continue
            if purpose and meta.get("purpose") != purpose:
                continue
            out.append(meta)
        out.sort(key=lambda m: (m["created_at"], m["id"]), reverse=True)
        return out

    # -- batch state -------------------------------------------------------
    def _bpath(self, bid: str, suffix: str = ".json") -> Path:
        if not valid_id(bid):
            raise FileNotFound(str(bid)[:80])
        return self.root / "batches" / f"{bid}{suffix}"

    def save_batch(self, rec: dict[str, Any]) -> None:
        with _lock:
            _atomic_write(self._bpath(rec["id"]), json.dumps(rec).encode())

    def load_batch(self, bid: str) -> dict[str, Any] | None:
        try:
            return json.loads(self._bpath(bid).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def update_batch(self, bid: str, **fields: Any) -> dict[str, Any] | None:
        """Read-modify-write under the lock so concurrent cancels are not lost."""
        with _lock:
            rec = self.load_batch(bid)
            if rec is None:
                return None
            rec.update(fields)
            self.save_batch(rec)
            return rec

    def list_batches(self, api: str) -> list[dict[str, Any]]:
        out = []
        for p in (self.root / "batches").glob("*.json"):
            try:
                rec = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if rec.get("api") == api:
                out.append(rec)
        out.sort(key=lambda r: (r["created_at"], r["id"]), reverse=True)
        return out

    def delete_batch(self, bid: str) -> None:
        for suffix in (".json", ".in.jsonl", ".out.jsonl", ".err.jsonl"):
            self._bpath(bid, suffix).unlink(missing_ok=True)

    def batch_file(self, bid: str, kind: str) -> Path:
        """kind: in | out | err."""
        return self._bpath(bid, f".{kind}.jsonl")

    def append_line(self, bid: str, kind: str, line: dict[str, Any]) -> None:
        with _lock, open(self.batch_file(bid, kind), "ab") as fh:
            fh.write(json.dumps(line, separators=(",", ":")).encode() + b"\n")
            fh.flush()
            os.fsync(fh.fileno())

    def read_lines(self, bid: str, kind: str) -> list[dict[str, Any]]:
        p = self.batch_file(bid, kind)
        if not p.exists():
            return []
        out = []
        for raw in p.read_bytes().split(b"\n"):
            if raw.strip():
                try:
                    out.append(json.loads(raw))
                except json.JSONDecodeError:
                    continue  # torn last line after a crash
        return out


_store: FileStore | None = None


def get_store() -> FileStore:
    """Process-wide store; rebuilt when the configured directory changes."""
    global _store
    root = settings.get("YUNSHU_FILES_DIR") or str(Path.home() / ".yunshu" / "files")
    root = str(Path(str(root)).expanduser())
    with _lock:
        if _store is None or str(_store.root) != root:
            _store = FileStore(
                Path(root),
                int(settings.get("YUNSHU_FILES_MAX_BYTES")),
                settings.get("YUNSHU_FILES_TTL_DAYS"),
            )
        else:
            _store.max_bytes = int(settings.get("YUNSHU_FILES_MAX_BYTES"))
            _store.ttl_days = settings.get("YUNSHU_FILES_TTL_DAYS")
        return _store


def reset_store() -> None:
    global _store
    _store = None


# ---------------------------------------------------------------------------
# Helpers for the chat / messages / responses routers
# ---------------------------------------------------------------------------


class FileRefError(FileStoreError):
    """A request references a file that is missing or unusable."""


def _store_or_error(file_id: str) -> tuple[FileStore, dict[str, Any]]:
    store = get_store()
    try:
        return store, store.get_meta(file_id)
    except FileNotFound:
        raise FileRefError(
            404, f"File not found: '{file_id}'", "file_not_found"
        ) from None


def read_file_bytes(file_id: str) -> bytes:
    try:
        return get_store().read(file_id)
    except FileNotFound:
        raise FileRefError(
            404, f"File not found: '{file_id}'", "file_not_found"
        ) from None


def file_mime(file_id: str) -> str:
    _, meta = _store_or_error(file_id)
    return meta.get("mime_type") or guess_mime(meta.get("filename"))


def text_of(file_id: str) -> str:
    """Decode a stored file as text (utf-8, undecodable bytes replaced)."""
    return read_file_bytes(file_id).decode("utf-8", errors="replace")


def _data_url(file_id: str) -> str:
    data = read_file_bytes(file_id)
    b64 = base64.b64encode(data).decode()
    return f"data:{file_mime(file_id)};base64,{b64}"


def resolve_file_block(block: dict[str, Any]) -> dict[str, Any]:
    """Return ``block`` with any stored-file reference inlined; other blocks
    are returned unchanged. Handles:

    * Anthropic ``{"type": "image"|"document", "source": {"type": "file", "file_id"}}``
      -> base64 source (text/* documents become a ``text`` source).
    * OpenAI Responses ``input_file`` / ``input_image`` with ``file_id``
      -> ``file_data`` / ``image_url`` data URL.
    * Chat Completions ``{"type": "file", "file": {"file_id"}}`` -> ``file_data``.
    * ``image_url`` whose url is a stored file id -> data URL.
    """
    if not isinstance(block, dict):
        return block
    t = block.get("type")
    src = block.get("source")
    if (
        t in ("image", "document")
        and isinstance(src, dict)
        and src.get("type") == "file"
    ):
        fid = src.get("file_id")
        mime = file_mime(fid)
        data = read_file_bytes(fid)
        new = {k: v for k, v in block.items() if k != "source"}
        if t == "document" and mime.startswith("text/"):
            new["source"] = {
                "type": "text",
                "media_type": "text/plain",
                "data": data.decode("utf-8", errors="replace"),
            }
        else:
            new["source"] = {
                "type": "base64",
                "media_type": mime,
                "data": base64.b64encode(data).decode(),
            }
        return new
    if t == "input_file" and block.get("file_id"):
        fid = block["file_id"]
        _, meta = _store_or_error(fid)
        new = {k: v for k, v in block.items() if k != "file_id"}
        new["filename"] = block.get("filename") or meta.get("filename")
        new["file_data"] = _data_url(fid)
        return new
    if (
        t == "file"
        and isinstance(block.get("file"), dict)
        and block["file"].get("file_id")
    ):
        f = dict(block["file"])
        fid = f.pop("file_id")
        _, meta = _store_or_error(fid)
        f.setdefault("filename", meta.get("filename"))
        f["file_data"] = _data_url(fid)
        return {**block, "file": f}
    if t == "input_image" and block.get("file_id"):
        new = {k: v for k, v in block.items() if k != "file_id"}
        new["image_url"] = _data_url(block["file_id"])
        return new
    if t == "image_url":
        iu = block.get("image_url")
        url = iu.get("url") if isinstance(iu, dict) else iu
        if isinstance(url, str) and valid_id(normalize_id(url)):
            data_url = _data_url(url)
            return {
                **block,
                "image_url": {**iu, "url": data_url}
                if isinstance(iu, dict)
                else data_url,
            }
    return block


def resolve_file_refs(obj: Any) -> Any:
    """Deep-copy ``obj`` (a messages/input structure) resolving every file block."""
    if isinstance(obj, list):
        return [resolve_file_refs(x) for x in obj]
    if isinstance(obj, dict):
        resolved = resolve_file_block(obj)
        if resolved is not obj:
            return resolved
        return {k: resolve_file_refs(v) for k, v in obj.items()}
    return copy.copy(obj)


def has_file_refs(obj: Any) -> bool:
    if isinstance(obj, list):
        return any(has_file_refs(x) for x in obj)
    if isinstance(obj, dict):
        if resolve_needed(obj):
            return True
        return any(has_file_refs(v) for v in obj.values())
    return False


def resolve_needed(block: dict[str, Any]) -> bool:
    t = block.get("type")
    src = block.get("source")
    if (
        t in ("image", "document")
        and isinstance(src, dict)
        and src.get("type") == "file"
    ):
        return True
    if t in ("input_file", "input_image") and block.get("file_id"):
        return True
    if (
        t == "file"
        and isinstance(block.get("file"), dict)
        and block["file"].get("file_id")
    ):
        return True
    if t == "image_url":
        iu = block.get("image_url")
        url = iu.get("url") if isinstance(iu, dict) else iu
        return isinstance(url, str) and valid_id(normalize_id(url))
    return False
