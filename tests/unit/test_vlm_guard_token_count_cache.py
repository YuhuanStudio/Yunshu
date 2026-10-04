"""Regressions for the engine-owned context guard token-count cache."""

import threading

import pytest

from yunshu_control.token_counter import (
    TokenCountCache,
    count_message_tokens,
    count_tokens,
)


class Tokenizer:
    def __init__(self, extra=0):
        self.calls = 0
        self.extra = extra

    def encode(self, text):
        self.calls += 1
        return range(len(text) + self.extra)


def test_repeated_exact_text_and_zero_count():
    cache = TokenCountCache()
    tok = Tokenizer()
    assert cache.count("abc", tok) == cache.count("abc", tok) == 3
    assert tok.calls == 1
    tok.extra = -3
    cache.clear()
    assert cache.count("abc", tok) == cache.count("abc", tok) == 0
    assert tok.calls == 2


def test_tokenizer_identity_not_equality():
    class EqualTokenizer(Tokenizer):
        def __eq__(self, other):
            return True

    a, b = EqualTokenizer(), EqualTokenizer(1)
    cache = TokenCountCache()
    assert cache.count("abc", a) == 3
    assert cache.count("abc", b) == 4
    assert cache.count("abc", a) == 3
    assert a.calls == 2 and b.calls == 1


def test_entry_and_text_byte_limits():
    cache = TokenCountCache(max_entries=2, max_text_bytes=12)
    tok = Tokenizer()
    for text in ("aa", "b", "c"):
        cache.count(text, tok)
    assert len(cache._counts) <= 2 and cache._text_bytes <= 12
    cache.count("aa", tok)
    assert tok.calls == 4
    cache.count("oversize", tok)
    cache.count("oversize", tok)
    assert tok.calls == 6 and "oversize" not in cache._counts
    assert cache._text_bytes <= 12


def test_failed_encode_is_not_cached_or_retried():
    class FailsOnce(Tokenizer):
        def encode(self, text):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("transient")
            return range(10)

    cache, tok = TokenCountCache(), FailsOnce()
    assert cache.count("abc def", tok) == count_tokens("abc def")
    assert tok.calls == 1
    assert cache.count("abc def", tok) == cache.count("abc def", tok) == 10
    assert tok.calls == 2


@pytest.mark.parametrize("replace_tokenizer", [False, True])
def test_late_encode_cannot_publish_after_clear_or_replacement(replace_tokenizer):
    started, release = threading.Event(), threading.Event()

    class Delayed(Tokenizer):
        def encode(self, text):
            self.calls += 1
            size = 3 + self.extra
            if self.calls == 1:
                started.set()
                assert release.wait(5)
            return range(size)

    cache, old = TokenCountCache(), Delayed()
    result = []
    worker = threading.Thread(target=lambda: result.append(cache.count("abc", old)))
    worker.start()
    try:
        assert started.wait(5)
        if replace_tokenizer:
            current = Tokenizer(1)
        else:
            cache.clear()
            old.extra = 1
            current = old
        assert cache.count("abc", current) == 4
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and result == [3]
    assert cache.count("abc", current) == 4


def test_clear_releases_tokenizer_and_text():
    cache, tok = TokenCountCache(), Tokenizer()
    cache.count("abc", tok)
    cache.clear()
    assert not cache._counts and cache._tokenizer is None and cache._text_bytes == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("qualified", [False, True])
async def test_vlm_guard_reuses_exact_counts_without_changing_budget(
    monkeypatch, qualified
):
    from yunshu_engine.vlm_engine import VLMEngine
    from yunshu_gateway.routers import chat

    engine = VLMEngine.__new__(VLMEngine)
    engine._tokenizer = Tokenizer()
    engine._processor = None
    engine._config = {"max_position_embeddings": 256}
    engine._guard_token_counts = TokenCountCache()
    engine._prefix_invariant_dispatch = qualified
    monkeypatch.setattr(chat, "get_engine", lambda: engine)
    monkeypatch.setattr(chat, "get_model_manager", lambda: None)
    budgets = []
    monkeypatch.setattr(
        chat, "_apply_token_budget", lambda req, count, e: budgets.append(count)
    )
    for _ in range(2):
        req = chat.ChatCompletionRequest(
            model="i8-test",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
            max_tokens=8,
        )
        response = await chat._handle_vlm_chat(
            req, [{"role": "user", "content": "hello"}], None
        )
        await response.body_iterator.aclose()
    assert budgets == [11, 11] and engine._tokenizer.calls == (1 if qualified else 2)


def test_tool_fields_and_media_are_recounted_with_cached_text():
    cache, tok = TokenCountCache(), Tokenizer()
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "abc"},
                {"type": "image_url", "image_url": {"url": "x"}},
            ],
            "tool_calls": [{"function": {"name": "echo", "arguments": {"v": 1}}}],
            "name": "echo",
            "tool_call_id": "id",
        }
    ]
    first = count_message_tokens(
        messages, tok, media_counter=lambda part: 7, text_counter=cache.count
    )
    calls = tok.calls
    second = count_message_tokens(
        messages, tok, media_counter=lambda part: 8, text_counter=cache.count
    )
    assert second == first + 1 and tok.calls == calls
    assert second == count_message_tokens(messages, tok, media_counter=lambda part: 8)


@pytest.mark.asyncio
async def test_engine_owns_and_clears_guard_counts(monkeypatch):
    from yunshu_engine import mlx_executor
    from yunshu_engine.vlm_engine import VLMEngine

    engine = VLMEngine("not-loaded")
    tokenizer = Tokenizer()
    assert engine._guard_token_counts.count("abc", tokenizer) == 3
    engine._running = True
    monkeypatch.setattr(mlx_executor, "sync_and_clear_cache", lambda: None)
    await engine.stop()
    assert not engine._guard_token_counts._counts
    assert engine._guard_token_counts._tokenizer is None
