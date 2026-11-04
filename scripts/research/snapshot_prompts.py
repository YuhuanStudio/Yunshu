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


@lru_cache(maxsize=1)
def chat_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)


def chat_token_count(text, tok=None):
    encoded = (tok or chat_tokenizer()).apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # Transformers 5 returns BatchEncoding, while older versions return IDs.
    ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


@lru_cache(maxsize=1)
def template_prefix_tokens(tok=None):
    """Tokens every chat prompt of this model starts with before its content: the common prefix of the rendered
    prompts of two contents that differ at their first token. A server may legitimately report this many cached tokens
    for a brand-new prompt (the shared template header), so the cold-cache guard allows it."""
    t = tok or chat_tokenizer()

    def ids(text):
        encoded = t.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        out = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        return out[0] if out and isinstance(out[0], list) else out

    a, b = ids("Alpha beta gamma."), ids("Zulu yankee xray.")
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def assert_chat_prompt(text, target, tok=None):
    actual = chat_token_count(text, tok)
    if actual != target:
        raise AssertionError(f"rendered prompt tokens={actual}, expected={target}")
    return actual


def exact_chat_prompt(source, target, suffix=""):
    """Fit the entire reference model input, including its chat template."""
    budget = target - chat_token_count("")
    for _ in range(8):
        text = exact_prompt(source, budget, suffix)
        actual = chat_token_count(text)
        if actual == target:
            assert_chat_prompt(text, target)
            return text
        budget += target - actual
    raise ValueError("cannot construct exact rendered chat budget")
