"""Prompt-length labels: one definition, one helper, one check.  CPU-only; never imports MLX.

DEFINITION.  A benchmark cell's length label (1024, 8192, "32K", ...) is the server-reported
``usage.prompt_tokens`` of the cell's FIRST turn: chat template, system prompt, and media (image/audio/video)
tokens all included.  Later turns add the previous reply and are not held to the label.  A label is therefore
never "content tokens", "characters / 4" or "filler tokens"; a harness that cannot hit the label within
``tol`` (default max(16, 1%)) fails the cell instead of recording a mislabelled number.

Use ``build_text`` to make text whose rendered chat prompt is exactly the target (given the tokenizer, its chat
template and a known media token count) and ``check_served_length`` on every turn-1 response.
"""

from __future__ import annotations

import math

DEFAULT_MODEL = "/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
FILLER = (
    "The quick brown fox jumps over the lazy dog while the committee reviews "
    "quarterly figures, notes on weather, travel schedules and unrelated trivia. "
)


class PromptLengthMismatchError(AssertionError):
    """The served prompt_tokens differ from the cell's label by more than the tolerance."""


def default_tol(label: int) -> int:
    return max(16, math.ceil(0.01 * label))


def check_served_length(label, usage, turn=1, tol=None):
    """Raise PromptLengthMismatchError unless usage['prompt_tokens'] is within tol of label (turn 1 only).

    Returns the served prompt_tokens (or None for turn > 1, where the reply is legitimately part of the prompt).
    A missing usage / prompt_tokens fails closed."""
    if turn != 1:
        return None
    served = (usage or {}).get("prompt_tokens")
    if not isinstance(served, int) or isinstance(served, bool):
        raise PromptLengthMismatchError(
            f"no usage.prompt_tokens to check against label {label}: {usage!r}"
        )
    tol = default_tol(label) if tol is None else tol
    if abs(served - label) > tol:
        raise PromptLengthMismatchError(
            f"length label {label} but the server received prompt_tokens={served} (|diff|={abs(served - label)} > tol {tol})"
        )
    return served


def load_tokenizer(path=DEFAULT_MODEL):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)


def _ids(tok, text):
    return tok.encode(text, add_special_tokens=False)


def chat_tokens(tok, text, system=None, **template_kwargs):
    """Token count of the rendered chat prompt (template + generation prompt) for one user message."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": text}
    ]
    kw = {"enable_thinking": False}
    kw.update(template_kwargs)
    out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **kw)
    ids = out["input_ids"] if hasattr(out, "keys") else out
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


def served_tokens(tok, text, media_tokens=0, system=None, **kw):
    """What a server using this tokenizer + template will report for text (+ known media tokens)."""
    return chat_tokens(tok, text, system, **kw) + media_tokens


def build_text(tok, source, target, suffix="", media_tokens=0, system=None, **kw):
    """Text (prefix of source + suffix) whose served length is exactly target = template + system + media + text.

    source is repeated if shorter than needed.  Raises ValueError when the target is below the fixed overhead
    or the length cannot be hit exactly (BPE joins make a +-1 wobble; padding fixes the short side)."""
    fixed = served_tokens(tok, "", media_tokens, system, **kw)
    if target - fixed - len(_ids(tok, suffix)) < 1:
        raise ValueError(f"target {target} is below fixed overhead {fixed} + suffix")
    budget = target - fixed - len(_ids(tok, suffix))
    ids = _ids(tok, source)
    if not _ids(tok, "\n" + source):
        raise ValueError("empty source text")  # the repeat loop below would never end
    while len(ids) < budget + 64:
        ids = ids + _ids(tok, "\n" + source)
    text = tok.decode(ids[:budget]) + suffix
    for _ in range(64):
        actual = served_tokens(tok, text, media_tokens, system, **kw)
        if actual == target:
            return text
        if actual > target:
            budget -= actual - target
            if budget < 1:
                break
            text = tok.decode(ids[:budget]) + suffix
        else:
            text = tok.decode(ids[:budget]) + " a" * (target - actual) + suffix
            if served_tokens(tok, text, media_tokens, system, **kw) > target:
                budget += 1
                text = tok.decode(ids[:budget]) + suffix
    raise ValueError(f"cannot construct served length {target}")


def filler_source(n_chars):
    return (FILLER * (n_chars // len(FILLER) + 1))[:n_chars]


def tokens_from_chars(tok, n_chars):
    """Deprecated-alias helper: the token length of n_chars of filler (what --ctx-chars used to mean)."""
    return len(_ids(tok, filler_source(n_chars)))
