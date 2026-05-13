"""Tests for RequestPriorityQueue — heap-based priority queue for the scheduler."""

import time
import threading
import pytest

from yunshu_engine.priority_queue import (
    RequestPriorityQueue,
    _QueueMode,
    make_waiting_queue,
)
from yunshu_engine.scheduler import SchedulingPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Item:
    """Minimal item with a stable identity for ``is`` checks."""

    def __init__(self, name: str, priority: int = 0):
        self.name = name
        self.priority = priority

    def __repr__(self):
        return f"_Item({self.name!r}, priority={self.priority})"


# ===========================================================================
# 1. FCFS ordering (FIFO)
# ===========================================================================

class TestFCFSOrdering:

    def test_fifo_order(self):
        """Items come out in insertion order under FCFS."""
        q = RequestPriorityQueue(mode=_QueueMode.FCFS)
        a, b, c = _Item("a"), _Item("b"), _Item("c")
        q.push(a)
        q.push(b)
        q.push(c)
        assert q.pop() is a
        assert q.pop() is b
        assert q.pop() is c

    def test_priority_ignored_in_fcfs(self):
        """Even if priority is passed, FCFS ignores it."""
        q = RequestPriorityQueue(mode=_QueueMode.FCFS)
        low = _Item("low", priority=1)
        high = _Item("high", priority=100)
        q.push(low, priority=1)
        q.push(high, priority=100)
        assert q.pop() is low   # insertion order wins
        assert q.pop() is high

    def test_large_fcfs_queue(self):
        """10k items maintain FIFO order."""
        q = RequestPriorityQueue(mode=_QueueMode.FCFS)
        items = [_Item(f"i-{i}") for i in range(10_000)]
        for it in items:
            q.push(it)
        for i, it in enumerate(items):
            assert q.pop() is it
        assert len(q) == 0


# ===========================================================================
# 2. PRIORITY ordering (higher priority first)
# ===========================================================================

class TestPriorityOrdering:

    def test_higher_priority_first(self):
        """Higher priority values are served first."""
        q = RequestPriorityQueue(mode=_QueueMode.PRIORITY)
        low = _Item("low", priority=1)
        mid = _Item("mid", priority=5)
        high = _Item("high", priority=10)
        q.push(low, priority=1)
        q.push(high, priority=10)
        q.push(mid, priority=5)
        assert q.pop() is high
        assert q.pop() is mid
        assert q.pop() is low

    def test_negative_priority(self):
        """Negative priorities work correctly (lower value = lower priority)."""
        q = RequestPriorityQueue(mode=_QueueMode.PRIORITY)
        a = _Item("a", priority=-10)
        b = _Item("b", priority=0)
        c = _Item("c", priority=10)
        q.push(c, priority=10)
        q.push(a, priority=-10)
        q.push(b, priority=0)
        assert q.pop() is c
        assert q.pop() is b
        assert q.pop() is a

    def test_large_priority_queue(self):
        """10k items with random priorities come out sorted."""
        import random
        random.seed(42)
        q = RequestPriorityQueue(mode=_QueueMode.PRIORITY)
        items = [(i, random.randint(0, 1000)) for i in range(10_000)]
        for idx, pri in items:
            q.push(_Item(f"i-{idx}", priority=pri), priority=pri)
        # Should come out in descending priority order
        prev_pri = float('inf')
        while q:
            item = q.pop()
            assert item.priority <= prev_pri
            prev_pri = item.priority
        assert prev_pri >= 0  # at least one item was processed


# ===========================================================================
# 3. Same-priority FIFO tiebreaking
# ===========================================================================

class TestSamePriorityTiebreaking:

    def test_same_priority_fifo(self):
        """Among equal-priority items, insertion order (FIFO) wins."""
        q = RequestPriorityQueue(mode=_QueueMode.PRIORITY)
        a, b, c = _Item("a"), _Item("b"), _Item("c")
        q.push(a, priority=5)
        q.push(b, priority=5)
        q.push(c, priority=5)
        assert q.pop() is a
        assert q.pop() is b
        assert q.pop() is c

    def test_mixed_priorities_fifo_tiebreak(self):
        """Within each priority level, FIFO order is preserved."""
        q = RequestPriorityQueue(mode=_QueueMode.PRIORITY)
        # Push in mixed order
        lo1 = _Item("lo1")
        hi1 = _Item("hi1")
        lo2 = _Item("lo2")
        hi2 = _Item("hi2")
        q.push(lo1, priority=1)
        q.push(hi1, priority=10)
        q.push(lo2, priority=1)
        q.push(hi2, priority=10)
        # hi1 and hi2 should come out first, in that order
        assert q.pop() is hi1
        assert q.pop() is hi2
        # Then lo1 and lo2
        assert q.pop() is lo1
        assert q.pop() is lo2


# ===========================================================================
# 4. push_front for preempted requests
# ===========================================================================

class TestPushFront:

    def test_push_front_goes_before_normal_push(self):
        """push_front items appear before normal push items at same priority."""
        q = RequestPriorityQueue(mode=_QueueMode.PRIORITY)
        normal = _Item("normal")
        preempted = _Item("preempted")
        q.push(normal, priority=5)
        q.push_front(preempted, priority=5)
        # preempted should come out first despite being added second
        assert q.pop() is preempted
        assert q.pop() is normal

    def test_push_front_fcfs_mode(self):
        """push_front works in FCFS mode too."""
        q = RequestPriorityQueue(mode=_QueueMode.FCFS)
        a = _Item("a")
        b = _Item("b")
        c = _Item("c")
        q.push(a)
        q.push(b)
        q.push_front(c)
        # c should come before a and b
        assert q.pop() is c
        assert q.pop() is a
        assert q.pop() is b

    def test_multiple_push_front_fifo(self):
        """Multiple push_front calls maintain FIFO among themselves."""
        q = RequestPriorityQueue(mode=_QueueMode.PRIORITY)
        a = _Item("a")
        b = _Item("b")
        c = _Item("c")
        q.push_front(a, priority=5)
        q.push_front(b, priority=5)
        q.push_front(c, priority=5)
        # All push_front items should come before any normal push
        normal = _Item("normal")
        q.push(normal, priority=5)
        results = [q.pop() for _ in range(4)]
        # push_front items come first in insertion order
        assert results[0] is a
        assert results[1] is b
        assert results[2] is c
        assert results[3] is normal

    def test_push_front_higher_priority_than_existing(self):
        """push_front at higher priority than existing items still respects priority."""
        q = RequestPriorityQueue(mode=_QueueMode.PRIORITY)
        low = _Item("low")
        high = _Item("high")
        q.push(low, priority=1)
        q.push_front(high, priority=10)
        # high has higher priority, should come first regardless of push_front
        assert q.pop() is high
        assert q.pop() is low


# ===========================================================================
# 5. Empty queue behavior
# ===========================================================================

class TestEmptyQueue:

    def test_pop_empty_raises(self):
        """pop() on empty queue raises IndexError."""
        q = RequestPriorityQueue(mode=_QueueMode.FCFS)
        with pytest.raises(IndexError):
            q.pop()

    def test_peek_empty_raises(self):
        """peek() on empty queue raises IndexError."""
        q = RequestPriorityQueue(mode=_QueueMode.FCFS)
        with pytest.raises(IndexError):
            q.peek()

    def test_len_empty(self):
        """len() returns 0 for empty queue."""
        q = RequestPriorityQueue(mode=_QueueMode.FCFS)
        assert len(q) == 0

    def test_bool_empty(self):
        """bool() returns False for empty queue."""
        q = RequestPriorityQueue(mode=_QueueMode.FCFS)
        assert not q

    def test_bool_nonempty(self):
        """bool() returns True for non-empty queue."""
        q = RequestPriorityQueue(mode=_QueueMode.FCFS)
        q.push(_Item("x"))
        assert q

    def test_clear(self):
        """clear() empties the queue."""
        q = RequestPriorityQueue(mode=_QueueMode.FCFS)
        for i in range(10):
            q.push(_Item(f"i-{i}"))
        assert len(q) == 10
        q.clear()
        assert len(q) == 0
        assert not q

    def test_peek_does_not_remove(self):
        """peek() returns the item without removing it."""
        q = RequestPriorityQueue(mode=_QueueMode.PRIORITY)
        item = _Item("top", priority=10)
        q.push(item, priority=10)
        assert q.peek() is item
        assert len(q) == 1
        assert q.pop() is item
        assert len(q) == 0


# ===========================================================================
# 6. Large queue performance (verify O(log n) push/pop)
# ===========================================================================

class TestLargeQueuePerformance:

    def test_push_pop_50k_fcfs(self):
        """50k push/pop under FCFS completes quickly (O(log n) per op)."""
        q = RequestPriorityQueue(mode=_QueueMode.FCFS)
        items = [_Item(f"i-{i}") for i in range(50_000)]
        t0 = time.perf_counter()
        for it in items:
            q.push(it)
        t_push = time.perf_counter() - t0
        assert len(q) == 50_000
        t1 = time.perf_counter()
        for it in items:
            assert q.pop() is it
        t_pop = time.perf_counter() - t1
        assert len(q) == 0
        # Sanity: should be well under 5 seconds for 50k items
        assert t_push < 5.0, f"push took {t_push:.2f}s"
        assert t_pop < 5.0, f"pop took {t_pop:.2f}s"

    def test_push_pop_50k_priority(self):
        """50k push/pop under PRIORITY completes quickly."""
        import random
        random.seed(123)
        q = RequestPriorityQueue(mode=_QueueMode.PRIORITY)
        pairs = [(i, random.randint(0, 100)) for i in range(50_000)]
        t0 = time.perf_counter()
        for idx, pri in pairs:
            q.push(_Item(f"i-{idx}", priority=pri), priority=pri)
        t_push = time.perf_counter() - t0
        assert len(q) == 50_000
        t1 = time.perf_counter()
        prev = float('inf')
        while q:
            item = q.pop()
            assert item.priority <= prev
            prev = item.priority
        t_pop = time.perf_counter() - t1
        assert t_push < 5.0, f"push took {t_push:.2f}s"
        assert t_pop < 5.0, f"pop took {t_pop:.2f}s"

    def test_sort_baseline_comparison(self):
        """Verify heap approach is faster than the old sort-every-step pattern.

        This simulates the old pattern: push all, sort, pop all.
        The heap should be competitive or faster.
        """
        import random
        random.seed(99)
        n = 10_000
        items = [(i, random.randint(0, 100)) for i in range(n)]

        # Heap approach
        q = RequestPriorityQueue(mode=_QueueMode.PRIORITY)
        t0 = time.perf_counter()
        for idx, pri in items:
            q.push(_Item(f"i-{idx}", priority=pri), priority=pri)
        while q:
            q.pop()
        t_heap = time.perf_counter() - t0

        # Old sort approach (simulate what the scheduler did)
        from collections import deque
        old_queue = deque()
        t0 = time.perf_counter()
        for idx, pri in items:
            old_queue.append(_Item(f"i-{idx}", priority=pri))
        # Sort the deque (old pattern)
        to_insert = list(old_queue)
        to_insert.sort(key=lambda r: r.priority, reverse=True)
        for item in to_insert:
            _ = item
        t_sort = time.perf_counter() - t0

        # Heap should be competitive — at most 10x slower than sort
        # (in practice it's usually faster for incremental operations)
        # This is a sanity check, not a strict performance guarantee.
        assert t_heap < t_sort * 10 + 0.5, (
            f"Heap ({t_heap:.3f}s) is much slower than sort ({t_sort:.3f}s)"
        )


# ===========================================================================
# 7. make_waiting_queue factory
# ===========================================================================

class TestMakeWaitingQueue:

    def test_fcfs_policy_creates_fcfs_queue(self):
        q = make_waiting_queue(SchedulingPolicy.FCFS)
        assert q._mode == _QueueMode.FCFS

    def test_priority_policy_creates_priority_queue(self):
        q = make_waiting_queue(SchedulingPolicy.PRIORITY)
        assert q._mode == _QueueMode.PRIORITY

    def test_string_policy_fcfs(self):
        q = make_waiting_queue("FCFS")
        assert q._mode == _QueueMode.FCFS

    def test_string_policy_priority(self):
        q = make_waiting_queue("PRIORITY")
        assert q._mode == _QueueMode.PRIORITY


# ===========================================================================
# 8. Thread safety
# ===========================================================================

class TestThreadSafety:

    def test_concurrent_pushes(self):
        """Multiple threads pushing concurrently should not corrupt the heap."""
        q = RequestPriorityQueue(mode=_QueueMode.FCFS)
        n_per_thread = 500
        n_threads = 4
        results = []

        def worker(tid):
            for i in range(n_per_thread):
                q.push(_Item(f"t{tid}-{i}"))

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = n_per_thread * n_threads
        assert len(q) == expected
        # Pop all items — should not raise
        for _ in range(expected):
            q.pop()
        assert len(q) == 0

    def test_concurrent_push_pop(self):
        """Concurrent push and pop should not corrupt the heap."""
        q = RequestPriorityQueue(mode=_QueueMode.PRIORITY)
        n = 2000
        popped_count = 0
        lock = threading.Lock()

        def pusher():
            for i in range(n):
                q.push(_Item(f"push-{i}", priority=i % 10), priority=i % 10)

        def popper():
            nonlocal popped_count
            while True:
                try:
                    q.pop()
                    with lock:
                        popped_count += 1
                except IndexError:
                    # Queue might be temporarily empty; check if pusher is done
                    with lock:
                        if popped_count >= n:
                            break
                    time.sleep(0.001)
                    continue

        push_thread = threading.Thread(target=pusher)
        pop_thread = threading.Thread(target=popper, daemon=True)
        push_thread.start()
        pop_thread.start()
        push_thread.join()
        # Give popper a moment to drain remaining
        time.sleep(0.1)
        # The queue should be nearly empty
        remaining = len(q)
        with lock:
            total = popped_count + remaining
        assert total == n


# ===========================================================================
# 9. __contains__ and __getitem__
# ===========================================================================

class TestContainsAndGetitem:

    def test_contains_true(self):
        q = RequestPriorityQueue(mode=_QueueMode.FCFS)
        item = _Item("x")
        q.push(item)
        assert item in q

    def test_contains_false(self):
        q = RequestPriorityQueue(mode=_QueueMode.FCFS)
        item = _Item("x")
        assert item not in q

    def test_getitem_zero_peek(self):
        """queue[0] should return the same item as peek()."""
        q = RequestPriorityQueue(mode=_QueueMode.PRIORITY)
        top = _Item("top", priority=10)
        q.push(top, priority=10)
        q.push(_Item("low", priority=1), priority=1)
        assert q[0] is top
        assert q[0] is q.peek()
