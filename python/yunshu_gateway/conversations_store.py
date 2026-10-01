"""Local store behind the OpenAI Conversations API.

Layout under the store root (default ``~/.yunshu/conversations``)::

    <conv_id>.json   {"id", "object", "created_at", "metadata", "items": [...]}

One JSON file per conversation; every write is atomic (temp file + ``os.replace``) and serialised by a
process-wide lock. Ids are validated against a strict pattern before they touch a path, so traversal
is impossible. Items are kept in insertion order (oldest first); the API layer reverses for ``desc``.
"""

from __future__ import annotations

import copy
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from yunshu_engine import settings

_ID_RE = re.compile(r"^conv_[0-9a-f]{24}$")
_lock = threading.RLock()

MAX_CREATE_ITEMS = 20
MAX_METADATA_KEYS = 16

_ITEM_PREFIX = {
    "message": "msg",
    "function_call": "fc",
    "function_call_output": "fco",
    "reasoning": "rs",
    "web_search_call": "ws",
    "mcp_call": "mcp",
    "mcp_list_tools": "mcpl",
    "mcp_approval_request": "mcpr",
    "mcp_approval_response": "mcpa",
    "compaction": "cmp",
    "custom_tool_call": "ctc",
    "custom_tool_call_output": "ctco",
}


class ConversationError(Exception):
    """Store-level error carrying an HTTP status and OpenAI error code."""

    def __init__(
        self,
        status: int,
        message: str,
        code: str | None = None,
        param: str | None = None,
    ):
        super().__init__(message)
        self.status = status
        self.message = message
        self.code = code
        self.param = param


class ConversationNotFound(ConversationError):  # noqa: N818
    def __init__(self, conv_id: str):
        super().__init__(
            404,
            f"Conversation '{str(conv_id)[:80]}' not found",
            "conversation_not_found",
        )


class ItemNotFound(ConversationError):  # noqa: N818
    def __init__(self, item_id: str):
        super().__init__(404, f"Item '{str(item_id)[:80]}' not found", "item_not_found")


def new_id() -> str:
    return f"conv_{secrets.token_hex(12)}"


def valid_id(value: Any) -> bool:
    return isinstance(value, str) and bool(_ID_RE.match(value))


def item_id_for(kind: str) -> str:
    return f"{_ITEM_PREFIX.get(kind, 'item')}_{secrets.token_hex(12)}"


def check_metadata(metadata: Any) -> dict[str, str]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ConversationError(
            400, "metadata must be an object of strings", "invalid_type", "metadata"
        )
    if len(metadata) > MAX_METADATA_KEYS:
        raise ConversationError(
            400,
            f"metadata can have at most {MAX_METADATA_KEYS} keys",
            "metadata_too_large",
            "metadata",
        )
    for k, v in metadata.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ConversationError(
                400,
                "metadata keys and values must be strings",
                "invalid_type",
                "metadata",
            )
        if len(k) > 64 or len(v) > 512:
            raise ConversationError(
                400,
                "metadata keys are at most 64 and values at most 512 characters",
                "metadata_too_large",
                "metadata",
            )
    return dict(metadata)


def _norm_content(content: Any, role: str) -> list[dict]:
    text_type = "output_text" if role == "assistant" else "input_text"
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": text_type, "text": content}]
    if not isinstance(content, list):
        raise ConversationError(
            400, "content must be a string or an array", "invalid_type", "content"
        )
    parts: list[dict] = []
    for p in content:
        if isinstance(p, str):
            parts.append({"type": text_type, "text": p})
        elif isinstance(p, dict):
            q = dict(p)
            if q.get("type") == "text" or not q.get("type") and "text" in q:
                q["type"] = text_type
            parts.append(q)
        else:
            raise ConversationError(
                400, "content parts must be objects", "invalid_type", "content"
            )
    return parts


def normalize_item(item: Any) -> dict:
    """Validate one input item and give it the stored shape (id, status, normalised content)."""
    if not isinstance(item, dict):
        raise ConversationError(400, "items must be objects", "invalid_type", "items")
    it = copy.deepcopy(item)
    kind = it.get("type") or ("message" if "role" in it or "content" in it else None)
    if not kind or not isinstance(kind, str):
        raise ConversationError(
            400, "each item needs a 'type'", "missing_required_parameter", "items"
        )
    it["type"] = kind
    if kind == "message":
        role = it.get("role") or "user"
        if role not in ("user", "assistant", "system", "developer"):
            raise ConversationError(
                400, f"invalid message role '{role}'", "invalid_value", "items"
            )
        it["role"] = role
        it["content"] = _norm_content(it.get("content"), role)
        it.setdefault("status", "completed")
    if not it.get("id"):
        it["id"] = item_id_for(kind)
    return it


class ConversationStore:
    def __init__(self, root: Path, max_items: int = 10000):
        self.root = Path(root)
        self.max_items = int(max_items)
        self.root.mkdir(parents=True, exist_ok=True)
        self.gc()

    def gc(self, grace_seconds: float = 600.0) -> int:
        """Delete ``.tmp`` leftovers from a crash mid-save (older than the grace)."""
        removed = 0
        cutoff = time.time() - grace_seconds
        for p in self.root.glob(".*.tmp"):
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
                    removed += 1
            except OSError:
                pass
        return removed

    # -- persistence ---------------------------------------------------------
    def _path(self, conv_id: str) -> Path:
        if not valid_id(conv_id):
            raise ConversationNotFound(conv_id)
        return self.root / f"{conv_id}.json"

    def _load(self, conv_id: str) -> dict:
        path = self._path(conv_id)
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            raise ConversationNotFound(conv_id) from None

    def _save(self, rec: dict) -> None:
        path = self._path(rec["id"])
        tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    @staticmethod
    def public(rec: dict) -> dict:
        return {
            "id": rec["id"],
            "object": "conversation",
            "created_at": rec["created_at"],
            "metadata": rec.get("metadata") or {},
        }

    # -- conversations -------------------------------------------------------
    def create(self, items: list | None = None, metadata: Any = None) -> dict:
        meta = check_metadata(metadata)
        items = items or []
        if not isinstance(items, list):
            raise ConversationError(
                400, "items must be an array", "invalid_type", "items"
            )
        if len(items) > MAX_CREATE_ITEMS:
            raise ConversationError(
                400,
                f"at most {MAX_CREATE_ITEMS} items can be added at once",
                "too_many_items",
                "items",
            )
        norm = [normalize_item(i) for i in items]
        if len(norm) > self.max_items:
            raise ConversationError(
                400, "conversation item limit reached", "too_many_items", "items"
            )
        rec = {
            "id": new_id(),
            "object": "conversation",
            "created_at": int(time.time()),
            "metadata": meta,
            "items": norm,
        }
        with _lock:
            self._save(rec)
        return self.public(rec)

    def get(self, conv_id: str) -> dict:
        with _lock:
            return self.public(self._load(conv_id))

    def exists(self, conv_id: str) -> bool:
        try:
            self.get(conv_id)
            return True
        except ConversationNotFound:
            return False

    def update_metadata(self, conv_id: str, metadata: Any) -> dict:
        meta = check_metadata(metadata)
        with _lock:
            rec = self._load(conv_id)
            rec["metadata"] = meta
            self._save(rec)
            return self.public(rec)

    def delete(self, conv_id: str) -> None:
        with _lock:
            path = self._path(conv_id)
            if not path.exists():
                raise ConversationNotFound(conv_id)
            path.unlink()

    # -- items ---------------------------------------------------------------
    def add_items(
        self, conv_id: str, items: list, *, cap: int | None = None
    ) -> list[dict]:
        """Append items; ``cap`` (API: 20) bounds one call, the store limit bounds the total."""
        if not isinstance(items, list):
            raise ConversationError(
                400, "items must be an array", "invalid_type", "items"
            )
        if cap is not None and len(items) > cap:
            raise ConversationError(
                400,
                f"at most {cap} items can be added at once",
                "too_many_items",
                "items",
            )
        norm = [normalize_item(i) for i in items]
        with _lock:
            rec = self._load(conv_id)
            if len(rec["items"]) + len(norm) > self.max_items:
                raise ConversationError(
                    400,
                    f"a conversation can hold at most {self.max_items} items",
                    "too_many_items",
                    "items",
                )
            rec["items"].extend(norm)
            self._save(rec)
        return norm

    def all_items(self, conv_id: str) -> list[dict]:
        """Every item, oldest first (a copy)."""
        with _lock:
            return copy.deepcopy(self._load(conv_id)["items"])

    def list_items(
        self,
        conv_id: str,
        *,
        limit: int = 20,
        order: str = "desc",
        after: str | None = None,
    ) -> dict:
        items = self.all_items(conv_id)
        if order == "desc":
            items.reverse()
        if after:
            for i, it in enumerate(items):
                if it["id"] == after:
                    items = items[i + 1 :]
                    break
            else:
                raise ConversationError(
                    404, f"Item '{after[:80]}' not found", "item_not_found", "after"
                )
        page = items[:limit]
        return {
            "object": "list",
            "data": page,
            "first_id": page[0]["id"] if page else None,
            "last_id": page[-1]["id"] if page else None,
            "has_more": len(items) > limit,
        }

    def get_item(self, conv_id: str, item_id: str) -> dict:
        for it in self.all_items(conv_id):
            if it["id"] == item_id:
                return it
        raise ItemNotFound(item_id)

    def delete_item(self, conv_id: str, item_id: str) -> dict:
        with _lock:
            rec = self._load(conv_id)
            kept = [it for it in rec["items"] if it["id"] != item_id]
            if len(kept) == len(rec["items"]):
                raise ItemNotFound(item_id)
            rec["items"] = kept
            self._save(rec)
            return self.public(rec)


_store: ConversationStore | None = None


def get_store() -> ConversationStore:
    """Process-wide store; rebuilt when the configured directory changes."""
    global _store
    root = settings.get("YUNSHU_CONVERSATIONS_DIR") or str(
        Path.home() / ".yunshu" / "conversations"
    )
    root = str(Path(str(root)).expanduser())
    with _lock:
        max_items = int(settings.get("YUNSHU_CONVERSATION_MAX_ITEMS"))
        if _store is None or str(_store.root) != root:
            _store = ConversationStore(Path(root), max_items)
        else:
            _store.max_items = max_items
        return _store


def reset_store() -> None:
    global _store
    _store = None
