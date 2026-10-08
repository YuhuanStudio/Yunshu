"""Local store behind stored chat completions (``store=true``).

Layout under the store root (default ``~/.yunshu/chat_completions``)::

    <created>_<id>.json   {"completion": {...ChatCompletion...}, "messages": [...input messages...]}

One JSON file per completion, atomic writes, the oldest completions are evicted past
``YUNSHU_CHAT_COMPLETIONS_MAX``. Ids are validated before they touch a path.
"""

from __future__ import annotations

import copy
import json
import os
import re
import secrets
import threading
from pathlib import Path
from typing import Any, cast

from yunshu_engine import settings

_ID_RE = re.compile(r"^chatcmpl-[A-Za-z0-9_-]{1,64}$")
_lock = threading.RLock()

MAX_METADATA_KEYS = 16


class ChatStoreError(Exception):
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


def check_metadata(metadata: Any) -> dict[str, str]:
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ChatStoreError(
            400, "metadata must be an object of strings", "invalid_type", "metadata"
        )
    if len(metadata) > MAX_METADATA_KEYS:
        raise ChatStoreError(
            400,
            f"metadata can have at most {MAX_METADATA_KEYS} keys",
            "metadata_too_large",
            "metadata",
        )
    for k, v in metadata.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ChatStoreError(
                400,
                "metadata keys and values must be strings",
                "invalid_type",
                "metadata",
            )
        if len(k) > 64 or len(v) > 512:
            raise ChatStoreError(
                400,
                "metadata keys are at most 64 and values at most 512 characters",
                "metadata_too_large",
                "metadata",
            )
    return dict(metadata)


def _not_found(cid: str) -> ChatStoreError:
    return ChatStoreError(
        404, f"No stored chat completion found with id '{str(cid)[:80]}'", "not_found"
    )


def input_messages(messages: list[dict]) -> list[dict]:
    """The request messages in the stored shape: an id each, plus ``content_parts`` for array content."""
    out: list[dict] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        q = {k: copy.deepcopy(v) for k, v in m.items() if v is not None}
        q["id"] = f"msg_{secrets.token_hex(12)}"
        content = m.get("content")
        if isinstance(content, list):
            parts = [
                p
                for p in content
                if isinstance(p, dict) and p.get("type") in ("text", "image_url")
            ]
            q["content_parts"] = copy.deepcopy(parts)
            q["content"] = (
                "".join(p.get("text", "") for p in parts if p.get("type") == "text")
                or None
            )
        else:
            q["content_parts"] = None
        out.append(q)
    return out


class ChatCompletionStore:
    def __init__(self, root: Path, max_items: int):
        self.root = root
        self.max_items = max_items

    # -- files --
    def _files(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(p for p in self.root.glob("*.json") if "_" in p.name)

    def _path_of(self, cid: str) -> Path | None:
        if not _ID_RE.match(cid or ""):
            return None
        for p in self._files():
            if p.name.split("_", 1)[1] == f"{cid}.json":
                return p
        return None

    def _load(self, p: Path) -> dict:
        return cast(dict, json.loads(p.read_text()))

    def _write(self, p: Path, rec: dict) -> None:
        tmp = p.with_name(p.name + f".{secrets.token_hex(4)}.tmp")
        try:
            tmp.write_text(json.dumps(rec, ensure_ascii=False))
            os.replace(tmp, p)
        finally:
            tmp.unlink(missing_ok=True)

    # -- operations --
    def save(
        self, completion: dict, messages: list[dict], metadata: dict | None
    ) -> None:
        cid = completion.get("id")
        if not isinstance(cid, str) or not _ID_RE.match(cid):
            return
        comp = copy.deepcopy(completion)
        comp["metadata"] = check_metadata(metadata)
        created = int(comp.get("created") or 0)
        with _lock:
            self.root.mkdir(parents=True, exist_ok=True)
            self._write(
                self.root / f"{created:012d}_{cid}.json",
                {"completion": comp, "messages": input_messages(messages)},
            )
            files = self._files()
            for old in files[: max(0, len(files) - self.max_items)]:
                old.unlink(missing_ok=True)

    def get(self, cid: str) -> dict:
        with _lock:
            p = self._path_of(cid)
            if p is None:
                raise _not_found(cid)
            return cast(dict, self._load(p)["completion"])

    def update_metadata(self, cid: str, metadata: Any) -> dict:
        meta = check_metadata(metadata)
        with _lock:
            p = self._path_of(cid)
            if p is None:
                raise _not_found(cid)
            rec = self._load(p)
            rec["completion"]["metadata"] = meta
            self._write(p, rec)
            return cast(dict, rec["completion"])

    def delete(self, cid: str) -> None:
        with _lock:
            p = self._path_of(cid)
            if p is None:
                raise _not_found(cid)
            p.unlink(missing_ok=True)

    @staticmethod
    def _page(
        rows: list[dict], ids: list[str], *, limit: int, order: str, after: str | None
    ) -> dict:
        if order == "desc":
            rows = rows[::-1]
            ids = ids[::-1]
        if after:
            if after not in ids:
                raise ChatStoreError(
                    400, f"after id '{after[:80]}' not found", "invalid_value", "after"
                )
            i = ids.index(after) + 1
            rows, ids = rows[i:], ids[i:]
        more = len(rows) > limit
        rows = rows[:limit]
        return {
            "object": "list",
            "data": rows,
            "first_id": rows[0]["id"] if rows else None,
            "last_id": rows[-1]["id"] if rows else None,
            "has_more": more,
        }

    def list(
        self,
        *,
        limit: int = 20,
        order: str = "asc",
        after: str | None = None,
        model: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict:
        with _lock:
            comps = [self._load(p)["completion"] for p in self._files()]
        if model:
            comps = [c for c in comps if c.get("model") == model]
        if metadata:
            comps = [
                c
                for c in comps
                if all(
                    (c.get("metadata") or {}).get(k) == v for k, v in metadata.items()
                )
            ]
        return self._page(
            comps, [c["id"] for c in comps], limit=limit, order=order, after=after
        )

    def list_messages(
        self, cid: str, *, limit: int = 20, order: str = "asc", after: str | None = None
    ) -> dict:
        with _lock:
            p = self._path_of(cid)
            if p is None:
                raise _not_found(cid)
            msgs = self._load(p)["messages"]
        return self._page(
            msgs, [m["id"] for m in msgs], limit=limit, order=order, after=after
        )


_store: ChatCompletionStore | None = None


def get_store() -> ChatCompletionStore:
    global _store
    root = settings.get("YUNSHU_CHAT_COMPLETIONS_DIR") or str(
        Path.home() / ".yunshu" / "chat_completions"
    )
    root = str(Path(str(root)).expanduser())
    with _lock:
        mx = int(settings.get("YUNSHU_CHAT_COMPLETIONS_MAX"))
        if _store is None or str(_store.root) != root:
            _store = ChatCompletionStore(Path(root), mx)
        else:
            _store.max_items = mx
        return _store


def reset_store() -> None:
    global _store
    _store = None


# -- stream reassembly --


def completion_from_chunks(chunks: list[dict]) -> dict | None:
    """Fold chat.completion.chunk objects of a finished stream into one chat.completion."""
    if not chunks:
        return None
    first = chunks[0]
    choices: dict[int, dict] = {}
    usage = None
    for ch in chunks:
        if isinstance(ch.get("usage"), dict):
            usage = ch["usage"]
        for c in ch.get("choices") or []:
            i = int(c.get("index", 0))
            slot = choices.setdefault(
                i,
                {
                    "index": i,
                    "message": {"role": "assistant", "content": None, "refusal": None},
                    "finish_reason": None,
                    "logprobs": None,
                },
            )
            d = c.get("delta") or {}
            msg = slot["message"]
            if d.get("content"):
                msg["content"] = (msg["content"] or "") + d["content"]
            for k in ("reasoning_content", "reasoning"):
                if d.get(k):
                    msg[k] = (msg.get(k) or "") + d[k]
            for tc in d.get("tool_calls") or []:
                calls = msg.setdefault("tool_calls", [])
                idx = int(tc.get("index", len(calls)))
                while len(calls) <= idx:
                    calls.append(
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    )
                cur = calls[idx]
                if tc.get("id"):
                    cur["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    cur["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    cur["function"]["arguments"] += fn["arguments"]
            if c.get("finish_reason"):
                slot["finish_reason"] = c["finish_reason"]
            lp = c.get("logprobs")
            if isinstance(lp, dict) and lp.get("content"):
                cur_lp = slot["logprobs"] or {"content": [], "refusal": None}
                cur_lp["content"].extend(lp["content"])
                slot["logprobs"] = cur_lp
    if not choices:
        return None
    out = {
        "id": first.get("id"),
        "object": "chat.completion",
        "created": first.get("created"),
        "model": first.get("model"),
        "choices": [choices[i] for i in sorted(choices)],
    }
    if first.get("system_fingerprint"):
        out["system_fingerprint"] = first["system_fingerprint"]
    if usage:
        out["usage"] = usage
    return out
