"""Gateway prompt guards count tokens off the event loop, with a per-text memo."""

import asyncio
import threading
import time

from yunshu_control.token_counter import count_message_tokens
from yunshu_gateway.token_budget import count_prompt_tokens, count_prompt_tokens_async


class SlowTokenizer:
    """Encodes one token per word; each encode blocks like a long real prompt would."""

    def __init__(self, delay=0.0):
        self.delay = delay
        self.calls = 0
        self.threads = set()

    def encode(self, text):
        self.calls += 1
        self.threads.add(threading.get_ident())
        if self.delay:
            time.sleep(self.delay)
        return text.split()


def _history(turns):
    msgs = [{"role": "system", "content": "you are an agent"}]
    for i in range(turns):
        msgs.append({"role": "user", "content": f"step {i} " + "word " * 20})
        msgs.append({"role": "assistant", "content": f"answer {i} " + "token " * 20})
    return msgs


def test_counts_match_the_uncached_counter():
    msgs = _history(5)
    tok = SlowTokenizer()
    assert count_prompt_tokens(msgs, tok) == count_message_tokens(msgs, SlowTokenizer())


def test_multi_turn_history_is_encoded_once_per_text():
    tok = SlowTokenizer()
    count_prompt_tokens(_history(10), tok)
    first = tok.calls
    # the next turn repeats the whole history and adds one message
    count_prompt_tokens(_history(10) + [{"role": "user", "content": "new turn"}], tok)
    assert tok.calls - first == 1


def test_long_prompt_does_not_block_the_event_loop():
    tok = SlowTokenizer(delay=0.1)
    msgs = _history(2)
    ticks = 0

    async def ticker(stop):
        nonlocal ticks
        while not stop.is_set():
            await asyncio.sleep(0.01)
            ticks += 1

    async def main():
        stop = asyncio.Event()
        task = asyncio.create_task(ticker(stop))
        total = await count_prompt_tokens_async(msgs, tok)
        stop.set()
        await task
        return total

    total = asyncio.run(main())
    assert total == count_message_tokens(msgs, SlowTokenizer())
    # the slow encode ran on a worker thread while the loop kept serving other work
    assert threading.get_ident() not in tok.threads
    assert ticks >= 5
