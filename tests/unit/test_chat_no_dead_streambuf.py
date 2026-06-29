"""the chat streaming path acquired a per-request StreamingResponseBuffer and
wrote every chunk into it, but the buffer was never read or flushed (the bytes actually
sent are `encoded`). Once the 64KB ring filled it logged a truncation WARNING on every
subsequent chunk — dead work + log spam on the hot path. Removed from chat.py.
"""

from __future__ import annotations

import inspect

from yunshu_gateway.routers import chat


def test_chat_no_longer_uses_streaming_response_buffer():
    src = inspect.getsource(chat)
    assert "_stream_buf" not in src
    assert "get_streaming_buffer" not in src
    assert "return_streaming_buffer" not in src
