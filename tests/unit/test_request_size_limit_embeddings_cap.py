"""+ 955: request size-limit byte enforcement + embeddings per-input length cap.

the request-size middleware byte-counted the stream only for `transfer-encoding:
  chunked`, so a request with NO Content-Length AND not chunked (HTTP/2 DATA frames, or any
  other framing) bypassed the limit entirely and was read whole into memory. Enforce on
  every no-Content-Length request.
the embeddings per-input length cap (_MAX_INPUT_TEXT_LENGTH) was checked only for a
  single str input; a list of N (up to 2048) arbitrarily long strings bypassed it
  (uncapped forward-pass / memory-pressure DoS). Enforce per list element.
"""

from __future__ import annotations

import inspect

import pydantic
import pytest


def test_byte_count_on_any_no_content_length():
    from yunshu_gateway import main

    src = inspect.getsource(main)
    # the byte-count branch fires on ANY missing Content-Length (not only chunked)
    assert "if content_length is None:" in src
    assert 'if transfer_encoding == "chunked" and content_length is None:' not in src


def test_list_str_length_cap_enforced():
    from yunshu_gateway.routers.embeddings import (
        _MAX_INPUT_TEXT_LENGTH,
        EmbeddingRequest,
    )

    long_s = "x" * (_MAX_INPUT_TEXT_LENGTH + 1)
    # a single over-long string is rejected (pre-existing)
    with pytest.raises(pydantic.ValidationError):
        EmbeddingRequest(model="m", input=long_s)
    # a LIST containing an over-long string is now rejected too
    with pytest.raises(pydantic.ValidationError):
        EmbeddingRequest(model="m", input=["ok", long_s, "fine"])
    # a normal list passes
    r = EmbeddingRequest(model="m", input=["hello", "world"])
    assert r.input == ["hello", "world"]
