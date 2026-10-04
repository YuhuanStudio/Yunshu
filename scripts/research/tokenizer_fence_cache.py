"""Research-only exact prefix reuse at a non-normalized Qwen added-token fence.

BPE and NFC process separate segments around non-normalized added tokens. Keep
only prefixes ending at <|im_end|>, never an arbitrary text/token boundary.
Unknown tokenizers, processors, special-token policies and encoder overrides
fall back to full encoding. Entries are bounded by text and token storage.
"""

import json
import sys
from collections import OrderedDict


class FenceCache:
    def __init__(self, max_bytes=8 << 20, max_entries=16):
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.bytes = 0
        self.identity = None
        self.hits = 0
        self.reused_tokens = 0

    def clear(self):
        self.entries.clear()
        self.bytes = 0
        self.identity = None

    def qualify(self, tokenizer):
        from mlx_lm.tokenizer_utils import TokenizerWrapper
        from tokenizers import normalizers, processors
        from transformers.models.qwen3_5.tokenization_qwen3_5 import Qwen3_5Tokenizer

        raw = (
            tokenizer._tokenizer
            if isinstance(tokenizer, TokenizerWrapper)
            else tokenizer
        )
        if (
            type(raw) is not Qwen3_5Tokenizer
            or "encode" in vars(raw)
            or raw.split_special_tokens
            or raw.bos_token is not None
        ):
            return None
        backend = raw.backend_tokenizer
        if backend.truncation or backend.padding:
            return None
        if (
            backend.normalizer is not None
            and type(backend.normalizer) is not normalizers.NFC
        ):
            return None
        processor = backend.post_processor
        if processor is not None and type(processor) is not processors.ByteLevel:
            return None
        fence = "<|im_end|>"
        fence_id = raw.convert_tokens_to_ids(fence)
        added = backend.get_added_tokens_decoder()
        if any(fence in str(value) and str(value) != fence for value in added.values()):
            return None
        token = added.get(fence_id)
        if (
            token is None
            or str(token) != fence
            or token.normalized
            or token.lstrip
            or token.rstrip
            or token.single_word
            or not token.special
        ):
            return None

        def state(obj):
            return obj.__getstate__() if obj is not None else None

        # Invalidate if the loaded tokenizer's configuration changes in place.
        identity = (
            raw,
            backend,
            backend.get_vocab_size(),
            state(backend.normalizer),
            state(backend.pre_tokenizer),
            state(processor),
            tuple((key, value.__getstate__()) for key, value in sorted(added.items())),
        )
        return identity, fence, fence_id

    def encode(self, tokenizer, text, *, add_special_tokens=True):
        def full(text):
            try:
                return tokenizer.encode(text, add_special_tokens=add_special_tokens)
            except TypeError:
                return tokenizer.encode(text)

        qualification = self.qualify(tokenizer)
        if qualification is None:
            self.clear()
            return full(text)
        identity, fence, fence_id = qualification
        identity += (add_special_tokens,)
        if self.identity != identity:
            self.clear()
            self.identity = identity
        match = max(
            (prefix for prefix in self.entries if text.startswith(prefix)),
            key=len,
            default=None,
        )
        if match is None:
            ids = full(text)
        else:
            prefix_ids, _ = self.entries[match]
            ids = list(prefix_ids) + full(text[len(match) :])
            self.entries.move_to_end(match)
            self.hits += 1
            self.reused_tokens += len(prefix_ids)
        end = text.rfind(fence)
        if end >= 0:
            end += len(fence)
            # With normalized=False, no occurrence can merge into ordinary BPE.
            # Match the final literal occurrence to its added-token position.
            count = text[:end].count(fence)
            positions = [i for i, value in enumerate(ids) if value == fence_id]
            if len(positions) == count:
                prefix = text[:end]
                prefix_ids = tuple(ids[: positions[-1] + 1])
                # Account conservatively for Python ints as well as references.
                size = sys.getsizeof(prefix) + 36 * len(prefix_ids)
                if size <= self.max_bytes and self.max_entries > 0:
                    old = self.entries.pop(prefix, None)
                    if old:
                        self.bytes -= old[1]
                    self.entries[prefix] = (prefix_ids, size)
                    self.bytes += size
                    while (
                        self.bytes > self.max_bytes
                        or len(self.entries) > self.max_entries
                    ):
                        _, (_, removed) = self.entries.popitem(last=False)
                        self.bytes -= removed
        return ids


def install():
    import mlx.core as mx

    from yunshu_engine.vlm_engine import VLMEngine, _VLMTextPromptCache

    original = VLMEngine._tokenize_with_cache
    original_clear = _VLMTextPromptCache.clear
    counts = dict(enabled=True, hits=0, reused_tokens=0)

    def tokenize(self, messages, enable_thinking=None, template_extra=None):
        if not counts["enabled"]:
            return original(self, messages, enable_thinking, template_extra)
        cache_key = _VLMTextPromptCache._compute_messages_hash(
            messages, enable_thinking
        )
        if template_extra:
            cache_key += "|" + json.dumps(template_extra, sort_keys=True)
        cached = self._text_prompt_cache.get_token_ids(cache_key)
        if cached is not None:
            return mx.array(cached)
        text = self._format_prompt(
            messages, enable_thinking=enable_thinking, template_extra=template_extra
        )
        cache = self._text_prompt_cache.__dict__.setdefault(
            "_research_fence_cache", FenceCache()
        )
        bos = getattr(self._tokenizer, "bos_token", None)
        add_special = not (isinstance(bos, str) and bos and text.startswith(bos))
        before = cache.hits
        ids = cache.encode(self._tokenizer, text, add_special_tokens=add_special)
        if cache.hits > before:
            if not counts["hits"]:
                import logging

                logging.getLogger("yunshu_engine.research.fence_cache").info(
                    "Exact tokenizer fence reuse engaged"
                )
            counts["hits"] += cache.hits - before
            counts["reused_tokens"] = cache.reused_tokens
        self._text_prompt_cache.put_token_ids(cache_key, ids)
        return mx.array(ids)

    def clear(cache):
        original_clear(cache)
        prefix = cache.__dict__.get("_research_fence_cache")
        if prefix is not None:
            prefix.clear()

    _VLMTextPromptCache.clear = clear
    VLMEngine._tokenize_with_cache = tokenize

    def uninstall():
        VLMEngine._tokenize_with_cache = original
        _VLMTextPromptCache.clear = original_clear

    return counts, uninstall
