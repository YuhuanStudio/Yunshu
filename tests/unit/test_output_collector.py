"""Tests for Output Collector — low-latency streaming with smart buffering."""

import pytest

from yunshu_engine.output_collector import RequestOutputCollector, RequestStreamState
from yunshu_engine.request import RequestOutput


class TestRequestOutputCollector:
    def test_put_and_get_nowait(self):
        collector = RequestOutputCollector()
        out = RequestOutput(request_id="test", new_text="hello")
        collector.put(out)
        result = collector.get_nowait()
        assert result is not None
        assert result.new_text == "hello"

    def test_get_nowait_empty(self):
        collector = RequestOutputCollector()
        assert collector.get_nowait() is None

    def test_sentinel(self):
        collector = RequestOutputCollector()
        collector.put(None)
        # After sentinel, get_nowait returns None (stream end)
        assert collector.get_nowait() is None

    def test_aggregation_merges_outputs(self):
        collector = RequestOutputCollector(aggregate=True)
        collector.put(
            RequestOutput(request_id="test", new_text="hel", new_token_ids=[1, 2])
        )
        collector.put(
            RequestOutput(request_id="test", new_text="lo", new_token_ids=[3])
        )
        result = collector.get_nowait()
        assert result is not None
        assert result.new_text == "hello"
        assert result.new_token_ids == [1, 2, 3]

    def test_no_aggregation_overwrites(self):
        collector = RequestOutputCollector(aggregate=False)
        collector.put(RequestOutput(request_id="test", new_text="first"))
        collector.put(RequestOutput(request_id="test", new_text="second"))
        result = collector.get_nowait()
        assert result.new_text == "second"

    def test_clear(self):
        collector = RequestOutputCollector()
        collector.put(RequestOutput(request_id="test", new_text="data"))
        collector.clear()
        assert collector.output is None
        assert collector.get_nowait() is None

    def test_clear_resets_sentinel(self):
        collector = RequestOutputCollector()
        collector.put(None)
        collector.clear()
        assert not collector._sentinel

    def test_merge_preserves_cumulative(self):
        collector = RequestOutputCollector()
        out1 = RequestOutput(
            request_id="test", new_text="a", output_token_ids=[1], completion_tokens=1
        )
        out2 = RequestOutput(
            request_id="test",
            new_text="b",
            output_token_ids=[1, 2],
            completion_tokens=2,
            finished=True,
            finish_reason="stop",
        )
        collector.put(out1)
        collector.put(out2)
        result = collector.get_nowait()
        assert result.finished is True
        assert result.finish_reason == "stop"
        assert result.completion_tokens == 2

    def test_merge_preserves_error(self):
        collector = RequestOutputCollector()
        out1 = RequestOutput(request_id="test", new_text="a", error="err1")
        out2 = RequestOutput(request_id="test", new_text="b")
        collector.put(out1)
        collector.put(out2)
        result = collector.get_nowait()
        assert result.error == "err1"

    def test_has_waiting_consumers_default_false(self):
        RequestOutputCollector._waiting_consumers = 0
        assert not RequestOutputCollector.has_waiting_consumers()


class TestRequestOutputCollectorAsync:
    @pytest.mark.asyncio
    async def test_async_get(self):
        collector = RequestOutputCollector()
        collector.put(RequestOutput(request_id="test", new_text="async"))
        result = await collector.get()
        assert result.new_text == "async"

    @pytest.mark.asyncio
    async def test_async_get_sentinel(self):
        collector = RequestOutputCollector()
        collector.put(None)
        result = await collector.get()
        assert result is None


class TestRequestStreamState:
    def test_should_send_first_token(self):
        state = RequestStreamState(stream_interval=3)
        assert state.should_send(total_tokens=1, finished=False) is True

    def test_should_send_at_interval(self):
        state = RequestStreamState(stream_interval=3)
        state.mark_sent(1)
        assert state.should_send(total_tokens=4, finished=False) is True
        assert state.should_send(total_tokens=3, finished=False) is False

    def test_should_send_on_finish(self):
        state = RequestStreamState(stream_interval=10)
        state.mark_sent(1)
        assert state.should_send(total_tokens=2, finished=True) is True

    def test_mark_sent(self):
        state = RequestStreamState()
        state.mark_sent(5)
        assert state.sent_tokens == 5
