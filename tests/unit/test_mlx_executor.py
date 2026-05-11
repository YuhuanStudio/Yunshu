"""Tests for MLX executor — GPU thread safety."""

import asyncio
import threading
import pytest


class TestMLXExecutor:
    """Test the MLX executor singleton."""

    def test_get_executor_returns_same_instance(self):
        from yunshu_engine.mlx_executor import get_mlx_executor
        e1 = get_mlx_executor()
        e2 = get_mlx_executor()
        assert e1 is e2

    def test_executor_is_threadpool(self):
        from concurrent.futures import ThreadPoolExecutor
        from yunshu_engine.mlx_executor import get_mlx_executor
        e = get_mlx_executor()
        assert isinstance(e, ThreadPoolExecutor)

    def test_executor_max_workers_is_one(self):
        from yunshu_engine.mlx_executor import get_mlx_executor
        e = get_mlx_executor()
        assert e._max_workers == 1


class TestRequestOutputCollectorConcurrency:
    """Test output collector under concurrent access."""

    def test_put_and_get_nowait(self):
        from yunshu_engine.output_collector import RequestOutputCollector
        from yunshu_engine.request import RequestOutput
        c = RequestOutputCollector()
        out = RequestOutput(request_id="test", new_text="hello", new_token_ids=[1])
        c.put(out)
        result = c.get_nowait()
        assert result is not None
        assert result.new_text == "hello"

    def test_put_sentinel(self):
        from yunshu_engine.output_collector import RequestOutputCollector
        c = RequestOutputCollector()
        c.put(None)
        assert c._sentinel is True
        result = c.get_nowait()
        assert result is None

    def test_aggregation_merges_text(self):
        from yunshu_engine.output_collector import RequestOutputCollector
        from yunshu_engine.request import RequestOutput
        c = RequestOutputCollector(aggregate=True)
        c.put(RequestOutput(request_id="t", new_text="hel", new_token_ids=[1]))
        c.put(RequestOutput(request_id="t", new_text="lo", new_token_ids=[2]))
        result = c.get_nowait()
        assert result.new_text == "hello"
        assert result.new_token_ids == [1, 2]

    def test_no_aggregation_overwrites(self):
        from yunshu_engine.output_collector import RequestOutputCollector
        from yunshu_engine.request import RequestOutput
        c = RequestOutputCollector(aggregate=False)
        c.put(RequestOutput(request_id="t", new_text="first", new_token_ids=[1]))
        c.put(RequestOutput(request_id="t", new_text="second", new_token_ids=[2]))
        result = c.get_nowait()
        assert result.new_text == "second"

    def test_get_nowait_returns_none_when_empty(self):
        from yunshu_engine.output_collector import RequestOutputCollector
        c = RequestOutputCollector()
        assert c.get_nowait() is None

    def test_clear_resets_state(self):
        from yunshu_engine.output_collector import RequestOutputCollector
        from yunshu_engine.request import RequestOutput
        c = RequestOutputCollector()
        c.put(RequestOutput(request_id="t", new_text="x", new_token_ids=[1]))
        c.clear()
        assert c.get_nowait() is None
        assert c._sentinel is False

    def test_has_waiting_consumers(self):
        from yunshu_engine.output_collector import RequestOutputCollector
        assert RequestOutputCollector.has_waiting_consumers() is False

    @pytest.mark.asyncio
    async def test_async_get(self):
        from yunshu_engine.output_collector import RequestOutputCollector
        from yunshu_engine.request import RequestOutput
        c = RequestOutputCollector()

        async def producer():
            await asyncio.sleep(0.01)
            c.put(RequestOutput(request_id="t", new_text="async", new_token_ids=[1]))

        asyncio.ensure_future(producer())
        result = await c.get()
        assert result.new_text == "async"


class TestRequestStreamState:
    """Test stream_interval batching."""

    def test_should_send_on_first_token(self):
        from yunshu_engine.output_collector import RequestStreamState
        s = RequestStreamState(stream_interval=5)
        assert s.should_send(total_tokens=1, finished=False) is True

    def test_should_send_at_interval(self):
        from yunshu_engine.output_collector import RequestStreamState
        s = RequestStreamState(stream_interval=5)
        s.mark_sent(1)
        assert s.should_send(total_tokens=6, finished=False) is True

    def test_should_not_send_before_interval(self):
        from yunshu_engine.output_collector import RequestStreamState
        s = RequestStreamState(stream_interval=5)
        s.mark_sent(1)
        assert s.should_send(total_tokens=3, finished=False) is False

    def test_should_send_on_finish(self):
        from yunshu_engine.output_collector import RequestStreamState
        s = RequestStreamState(stream_interval=100)
        s.mark_sent(0)
        assert s.should_send(total_tokens=2, finished=True) is True

    def test_mark_sent(self):
        from yunshu_engine.output_collector import RequestStreamState
        s = RequestStreamState(stream_interval=5)
        s.mark_sent(10)
        assert s.sent_tokens == 10
