"""A memory-admission refusal is the expected answer: one WARNING line, no traceback.

The refusal is driven through the real streaming runner (`_stream_vlm_runner_text`), with only
the batch runner's event generator replaced by one that raises the admission error.
"""

import logging

import mlx.core as mx

from yunshu_engine.memory_admission import MemoryAdmissionError, PromptTooLongError
from yunshu_engine.vlm_engine import VLMEngine


class _Queue:
    def __init__(self) -> None:
        self.items: list = []

    def put_nowait(self, item) -> None:
        self.items.append(item)


def _refuse_through_runner(caplog, exc: BaseException):
    caplog.set_level(logging.DEBUG, logger="yunshu_engine")
    engine = VLMEngine.__new__(VLMEngine)

    def refused(input_ids, stats=None, **params):
        raise exc
        yield  # makes this a generator

    engine._runner_events = refused
    queue = _Queue()
    engine._stream_vlm_runner_text(mx.array([1, 2, 3]), "req_abc123", queue)
    return [r for r in caplog.records if r.name.startswith("yunshu_engine")], queue


def test_prompt_too_long_is_one_warning_with_request_id_and_fits(caplog):
    records, queue = _refuse_through_runner(
        caplog, PromptTooLongError(200010, 183258, "needs 8.58 GiB")
    )
    assert not [r for r in records if r.levelno >= logging.ERROR]
    assert not [r for r in records if r.exc_info]
    warnings = [r for r in records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, [r.getMessage() for r in records]
    msg = warnings[0].getMessage()
    assert "req_abc123" in msg and "183258" in msg
    # the client still gets the refusal, unchanged
    assert queue.items[-1].finish_reason == "error"
    assert "prompt_too_long" in queue.items[-1].error


def test_plain_memory_refusal_is_a_warning_without_traceback(caplog):
    records, _ = _refuse_through_runner(caplog, MemoryAdmissionError("too big"))
    assert not [r for r in records if r.levelno >= logging.ERROR]
    assert not [r for r in records if r.exc_info]


def test_other_failures_still_log_error_with_traceback(caplog):
    records, _ = _refuse_through_runner(caplog, RuntimeError("boom"))
    errors = [r for r in records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and errors[0].exc_info is not None
