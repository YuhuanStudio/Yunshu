"""Tests for MLX executor — GPU thread safety."""


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
