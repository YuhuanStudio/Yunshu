"""CPU-only, exact-token snapshot corpus; never imports MLX."""

from functools import lru_cache
from pathlib import Path

MODEL = Path("/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp")


@lru_cache(maxsize=1)
def tokenizer():
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(MODEL / "tokenizer.json"))


def token_count(text, tok=None):
    return len((tok or tokenizer()).encode(text, add_special_tokens=False).ids)


def exact_prompt(source, target, suffix="", tok=None):
    """Preserve an instruction suffix and fill exactly target content tokens.

    Content tokens exclude engine-specific chat-template overhead, which is
    separately recorded as the server's prompt_tokens. Round-trip verification
    catches BPE joins and incomplete Unicode tokens after truncation.
    """
    tok = tok or tokenizer()
    budget = target - token_count(suffix, tok) - 8
    if budget < 1:
        raise ValueError("token budget too small for instruction")
    ids = tok.encode(source, add_special_tokens=False).ids
    if len(ids) < budget:
        raise ValueError("source shorter than token budget")
    prefix = tok.decode(ids[:budget], skip_special_tokens=False)
    text = prefix + suffix
    for _ in range(64):
        actual = token_count(text, tok)
        if actual == target:
            assert token_count(text, tok) == target
            return text
        if actual > target:
            budget -= actual - target
            prefix = tok.decode(ids[:budget], skip_special_tokens=False)
            text = prefix + suffix
        else:
            text += " a" * (target - actual)
    raise ValueError("cannot construct an exact token budget")


def assert_prompt(text, target, tok=None):
    actual = token_count(text, tok)
    if actual != target:
        raise AssertionError(f"prompt content tokens={actual}, expected={target}")
    return actual
