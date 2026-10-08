"""Atomic local JSON persistence for Evals, following chat_store's storage protocol."""

from __future__ import annotations

import copy
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import cast

from yunshu_engine import settings

from .conversations_store import ConversationError, check_metadata

_lock = threading.RLock()
_ID = re.compile(r"^(eval|evalrun)_[0-9a-f]{24}$")


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(12)}"


class EvalStore:
    def __init__(self, root: Path):
        self.root = root

    def path(self, id: str) -> Path:
        if not _ID.fullmatch(id):
            raise ConversationError(404, "Eval or run not found", "not_found")
        return self.root / f"{id}.json"

    def get(self, id: str) -> dict:
        with _lock:
            try:
                return cast(dict, json.loads(self.path(id).read_text()))
            except (FileNotFoundError, json.JSONDecodeError):
                raise ConversationError(
                    404, "Eval or run not found", "not_found"
                ) from None

    def save(self, rec: dict) -> dict:
        with _lock:
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.path(rec["id"])
            tmp = path.with_name(path.name + f".{secrets.token_hex(4)}.tmp")
            try:
                with tmp.open("w") as f:
                    json.dump(rec, f, ensure_ascii=False, allow_nan=False)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
        return copy.deepcopy(rec)

    def change(self, id: str, **changes) -> dict:
        with _lock:
            rec = self.get(id)
            rec.update(changes)
            return self.save(rec)

    def rows(self, prefix: str) -> list[dict]:
        with _lock:
            rows = [self.get(p.stem) for p in self.root.glob(f"{prefix}_*.json")]
        return sorted(rows, key=lambda r: (r["created_at"], r["id"]))

    def delete(self, id: str) -> None:
        with _lock:
            self.get(id)
            self.path(id).unlink()

    def get_eval(self, id: str) -> dict:
        if not id.startswith("eval_"):
            raise ConversationError(404, "Eval not found", "not_found")
        return self.get(id)

    def create_eval(self, body: dict) -> dict:
        config = body["data_source_config"]
        schema: dict = {
            "type": "object",
            "properties": {"item": config.get("item_schema", {"type": "object"})},
            "required": ["item"],
        }
        if config.get("include_sample_schema"):
            schema["properties"]["sample"] = {
                "type": "object",
                "properties": {"output_text": {"type": "string"}},
            }
        public_config = {"type": config["type"], "schema": schema}
        if "metadata" in config:
            public_config["metadata"] = config["metadata"]
        return self.save(
            {
                **body,
                "_config": config,
                "data_source_config": public_config,
                "id": new_id("eval"),
                "object": "eval",
                "created_at": int(time.time()),
                "updated_at": int(time.time()),
                "name": body.get("name") or "",
                "metadata": check_metadata(body.get("metadata")),
            }
        )

    def run(self, eid: str, rid: str) -> dict:
        self.get_eval(eid)
        rec = self.get(rid)
        if rec.get("eval_id") != eid:
            raise ConversationError(404, "Run not found", "not_found")
        return rec


def get_store() -> EvalStore:
    root = settings.get("YUNSHU_EVALS_DIR") or str(Path.home() / ".yunshu" / "evals")
    return EvalStore(Path(root).expanduser())


def public(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def page(rows: list[dict], *, limit=20, order="asc", after=None) -> dict:
    if order == "desc":
        rows = rows[::-1]
    if after:
        ids = [r["id"] for r in rows]
        if after not in ids:
            raise ConversationError(
                400, "Unknown pagination cursor", "invalid_value", "after"
            )
        rows = rows[ids.index(after) + 1 :]
    data = [public(r) for r in rows[:limit]]
    return dict(
        object="list",
        data=data,
        has_more=len(rows) > limit,
        first_id=data[0]["id"] if data else None,
        last_id=data[-1]["id"] if data else None,
    )


def stored_completions() -> list[dict]:
    """Read apiplanned fc28350d's atomic <created>_<chatcmpl-id>.json format.

    No duplicate chat_store module: either branch can land first without an
    add/add conflict. Atomic replacement makes each record a consistent snapshot.
    """
    root = settings.get("YUNSHU_CHAT_COMPLETIONS_DIR") or str(
        Path.home() / ".yunshu" / "chat_completions"
    )
    rows = []
    for path in sorted(Path(root).expanduser().glob("*.json")):
        if not re.fullmatch(r"[0-9]+_chatcmpl-[A-Za-z0-9_-]{1,64}\.json", path.name):
            continue
        try:
            rec = json.loads(path.read_text())
        except FileNotFoundError:
            continue  # concurrent deletion or retention eviction
        if (
            isinstance(rec, dict)
            and isinstance(rec.get("completion"), dict)
            and isinstance(rec.get("messages"), list)
        ):
            rows.append(rec)
    return rows
