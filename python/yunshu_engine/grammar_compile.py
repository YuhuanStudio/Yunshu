"""Bounded CPU grammar artifacts; mutable matchers always belong to requests.

Schema keys preserve object order: llguidance emits properties in schema order.
Tokenizer-bound templates are keyed by the retained LLTokenizer identity (including
its vocabulary width), grammar text and parser options. No MLX work runs here.
"""

from __future__ import annotations

import asyncio
import copy
import json
import threading
import weakref
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, cast


class ArtifactCache:
    """LRU plus single-flight construction. Failed artifacts are never retained.

    Bound entries and key/source bytes, rather than claiming a native parser's
    opaque memory size. Oversized inputs still work but bypass retention.
    """

    def __init__(self, max_entries: int = 128, max_key_bytes: int = 4 << 20):
        self.max_entries = max_entries
        self.max_key_bytes = max_key_bytes
        self._entries: OrderedDict[tuple, tuple[Any, int]] = OrderedDict()
        self._pending: dict[tuple, Future] = {}
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key: tuple, build: Callable[[], Any]) -> Any:
        size = len(repr(key).encode("utf-8"))
        if size > self.max_key_bytes // 4:
            return build()
        with self._lock:
            hit = self._entries.get(key)
            if hit is not None:
                self._entries.move_to_end(key)
                return hit[0]
            future = self._pending.get(key)
            owner = future is None
            if owner:
                future = Future()
                self._pending[key] = future
        assert future is not None
        if not owner:
            return future.result()
        try:
            result = build()
        except BaseException as exc:
            with self._lock:
                self._pending.pop(key, None)
                future.set_exception(exc)
            raise
        with self._lock:
            self._entries[key] = (result, size)
            self._bytes += size
            while (
                len(self._entries) > self.max_entries
                or self._bytes > self.max_key_bytes
            ):
                _, (_, evicted_size) = self._entries.popitem(last=False)
                self._bytes -= evicted_size
            self._pending.pop(key, None)
            future.set_result(result)
        return result

    def contains(self, key: tuple) -> bool:
        with self._lock:
            if key not in self._entries:
                return False
            self._entries.move_to_end(key)
            return True

    def clear(self) -> None:
        """Clear completed artifacts (for CPU measurements, never cancel owners)."""
        with self._lock:
            self._entries.clear()
            self._bytes = 0


ARTIFACTS = ArtifactCache()
_WORKERS = ThreadPoolExecutor(max_workers=2, thread_name_prefix="yunshu-grammar")
_ADMISSION: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def schema_source(schema: Any) -> str:
    from .grammar_constraint import _llg_schema_json

    raw = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    return cast(
        str, ARTIFACTS.get(("schema-source", raw), lambda: _llg_schema_json(schema))
    )


def llg_grammar(kind: str, source: str, *, compact: bool = False) -> str:
    from llguidance import LLMatcher

    from .grammar_constraint import _LLG_COMPACT, _LLG_WHITESPACE

    def build():
        grammar = (
            LLMatcher.grammar_from_lark(source)
            if kind == "cfg"
            else LLMatcher.grammar_from_json_schema(
                source, defaults=_LLG_COMPACT if compact else _LLG_WHITESPACE
            )
        )
        error = LLMatcher.validate_grammar(grammar)
        if error:
            raise ValueError(error)
        return grammar

    return cast(str, ARTIFACTS.get(("llg", kind, source, compact), build))


def new_llg_matcher(llt: Any, grammar: str) -> Any:
    from llguidance import LLMatcher

    def build():
        error = LLMatcher.validate_grammar(grammar, llt)
        if error:
            raise ValueError(error)
        template = LLMatcher(llt, grammar)
        error = template.get_error()
        if error:
            raise ValueError(error)
        # Retaining llt prevents an id from being recycled under a cached key.
        return llt, template, threading.Lock()

    _, template, lock = ARTIFACTS.get(("matcher", id(llt), grammar), build)
    with lock:
        return template.deep_copy()


def regex_dfa(pattern: str) -> Any:
    from .grammar_constraint import _RegexDFA

    template = ARTIFACTS.get(("regex", pattern), lambda: _RegexDFA(pattern))
    # NFA construction is complete; only lazy DFA transition/closure caches mutate.
    # Give each request independent caches and traversal state.
    dfa = copy.copy(template)
    dfa._closure_cache = dict(template._closure_cache)
    dfa._trans = {}
    dfa._nfa_transitions = {}
    for state, edges in template._nfa_transitions.items():
        cloned = []
        for label, target in edges:
            private_label = copy.copy(label)
            private_label.cache = {}
            cloned.append((private_label, target))
        dfa._nfa_transitions[state] = cloned
    return dfa


def _prepare_spec(spec: Any) -> None:
    from . import settings
    from .grammar_constraint import validate_constraint_spec, validate_llg_json_schema
    from .json_schema import UnsupportedSchemaError, validate_supported_schema

    if isinstance(spec, str) and spec != "json_object":
        spec = json.loads(spec)
    validate_constraint_spec(spec)
    if isinstance(spec, dict) and spec.get("type") in ("regex", "choice", "cfg"):
        return
    if settings.get("YUNSHU_JSON_SCHEMA_ENGINE") == "llguidance":
        schema = {"type": "object"} if spec == "json_object" else spec
        try:
            validate_llg_json_schema(schema)
        except UnsupportedSchemaError:
            # Preserve the existing lossless in-house fallback policy.
            validate_supported_schema(schema)


async def prepare_constraint(spec: Any) -> None:
    """Compile/validate off the event loop before admitting constrained decode.

    At most 32 queued submissions per event loop and two active CPU workers.
    Cancellation never exposes a partially built artifact or emits a token.
    Tokenizer binding and any MLX token-table work remain on existing threads.
    """
    if spec is None:
        return
    from . import settings

    raw = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
    key = ("prepared", settings.get("YUNSHU_JSON_SCHEMA_ENGINE"), raw)
    # Repeated requests need no executor round trip. Only success gets this key.
    if ARTIFACTS.contains(key):
        return
    snapshot = copy.deepcopy(spec)
    loop = asyncio.get_running_loop()
    gate = _ADMISSION.get(loop)
    if gate is None:
        gate = asyncio.Semaphore(32)
        _ADMISSION[loop] = gate
    async with gate:

        def build():
            _prepare_spec(snapshot)
            return True

        await loop.run_in_executor(_WORKERS, ARTIFACTS.get, key, build)
