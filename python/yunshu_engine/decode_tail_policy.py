"""Whether a finished reply's decode cache can ever be reused by the next turn.

A decode cache holds the reply's reasoning as well as its content. The next prompt carries that
reasoning only when the client sends it back (``reasoning_content`` / ``reasoning`` on chat, reasoning
items on Responses, ``thinking`` blocks on Anthropic); a client that sends only the content leaves the
reasoning out, and then the cache can never match. Such replies are not stored; every other reply keeps
its decode-tail store.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
_LOGGED: set[str] = set()


def _field(item, key):
    return item.get(key) if isinstance(item, dict) else getattr(item, key, None)


def _echo(turns: list[bool]) -> bool | None:
    """None: no prior assistant turn. True: some prior turn carries reasoning. False: none does."""
    if not turns:
        return None
    return any(turns)


def chat_echoes_reasoning(messages) -> bool | None:
    """Chat Completions history: an assistant message with ``reasoning_content`` or ``reasoning``."""
    return _echo(
        [
            bool(_field(m, "reasoning_content") or _field(m, "reasoning"))
            for m in messages
            if _field(m, "role") == "assistant"
        ]
    )


def anthropic_echoes_reasoning(messages) -> bool | None:
    """Messages history: an assistant turn with a ``thinking`` or ``redacted_thinking`` block."""
    turns = []
    for m in messages:
        if _field(m, "role") != "assistant":
            continue
        content = _field(m, "content")
        blocks = content if isinstance(content, list) else []
        turns.append(
            any(_field(b, "type") in ("thinking", "redacted_thinking") for b in blocks)
        )
    return _echo(turns)


def set_request_echo(value: bool | None) -> None:
    """Put this request's reading of the history on its tracker info; the engine copies it to the job."""
    from .request_tracker import current_request_info

    info = current_request_info.get()
    if info is None:
        return
    setattr(info, "client_echoes_reasoning", value)  # noqa: B010 (tracker info is a plain object)
    # the generation may already be registered: its cancel event is what the runner reads
    gen = getattr(info, "gen", None)
    cancel = getattr(gen, "cancel_event", None)
    if cancel is not None:
        cancel.client_echoes_reasoning = value


def reasoning_token_ids(tokenizer) -> tuple[int | None, int | None]:
    """The single-token ids of ``<think>`` and ``</think>``, or None where the vocabulary has none."""
    ids = []
    for tok in ("<think>", "</think>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
        except Exception:  # noqa: BLE001
            tid = None
        unk = getattr(tokenizer, "unk_token_id", None)
        ids.append(None if tid is None or tid == unk else int(tid))
    return ids[0], ids[1]


def reply_is_stranded(prompt_ids, generated, open_id, close_id) -> bool:
    """The reply opened a reasoning block the prompt left open, and its reasoning comes before any
    content (or the reply was cut off inside it)."""
    if open_id is None or close_id is None:
        return False
    tail = [int(t) for t in list(prompt_ids)[-4:]]
    if (
        open_id not in tail or close_id in tail
    ):  # thinking off: the prompt closed the block
        return False
    gen = [int(t) for t in generated]
    return close_id not in gen or gen.index(close_id) > 0


def skip_decode_tail(client_echoes, prompt_ids, generated, open_id, close_id) -> bool:
    """Skip the store only when the client sent the history back without any reasoning (at least one
    prior assistant turn, none with reasoning) and this reply's content sits behind its reasoning.
    Turn 1 (no history) and any client that echoes reasoning keep the store."""
    if client_echoes is not False:
        return False
    return reply_is_stranded(prompt_ids, generated, open_id, close_id)


def log_once(reason: str, text: str, *args) -> None:
    if reason not in _LOGGED:
        _LOGGED.add(reason)
        logger.info(text, *args)
