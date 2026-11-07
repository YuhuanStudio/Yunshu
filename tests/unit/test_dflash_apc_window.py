import pytest

from yunshu_engine import dflash_apc_window as w


@pytest.fixture(autouse=True)
def _fresh():
    w.clear()
    w.set_budget(1000)
    yield
    w.clear()


def test_key_binds_namespace_and_length():
    assert w.key_for(1, [1, 2]) != w.key_for(2, [1, 2])
    assert w.key_for(1, [1, 2]) != w.key_for(1, [1, 2, 3])


def test_budget_evicts_oldest_and_counts_bytes():
    for i in range(5):
        w.put(w.key_for(0, [i]), [i], 400)
    assert w.resident_bytes() <= 1000 and len(w._STORE) == 2
    assert w.get(w.key_for(0, [0])) is None and w.get(w.key_for(0, [4])) == [4]
    assert w.put(w.key_for(0, [99]), [0], 5000) is False


class _Budget:
    def __init__(self, allow=True):
        self.allow, self.written = allow, []

    def allow_write(self, size):
        return self.allow

    def note_written(self, path, size=None):
        self.written.append(path)


@pytest.fixture
def disk(tmp_path):
    w._DISK.update(dir=None, budget=None, tag=None)
    # the counters are module-global: another test's writes must not leak into this one's assertions
    # (found by the random-order pass: disk_written was 4, not 1)
    for k in w._STATS:
        w._STATS[k] = 0
    w.clear()
    yield tmp_path
    w.flush()
    w._DISK.update(dir=None, budget=None, tag=None)


def _window(seed):
    import mlx.core as mx

    mx.random.seed(seed)
    return [mx.random.normal((1, 4, 8)).astype(mx.bfloat16) for _ in range(3)]


def test_prompt_end_window_survives_a_ram_miss(disk):
    # an SSD-restored checkpoint (another process, or evicted from RAM) had no window
    # and its decode drafted from a handful of positions (tf110 1021h2: -10% at 32K)
    import mlx.core as mx

    budget = _Budget()
    w.set_disk(disk / "win", budget, tag=(5, 19))
    w.set_budget(10_000)
    key = w.key_for(7, [1, 2, 3])
    layers = _window(0)
    w.put(key, layers, sum(int(x.nbytes) for x in layers))
    assert w.persist(key)
    w.flush()
    assert budget.written and w.stats()["disk_written"] == 1
    w.clear()  # a new process: RAM empty
    got = w.get(key)
    assert got is not None and len(got) == 3
    for a, b in zip(layers, got, strict=True):
        assert a.dtype == b.dtype == mx.bfloat16 and mx.array_equal(a, b)
    # another drafter (other target layers) never reads it
    w.clear()
    w.set_tag((1, 2))
    assert w.get(key) is None


def test_no_disk_write_without_tag_or_space(disk):
    w.set_budget(10_000)
    key = w.key_for(7, [1, 2])
    layers = _window(1)
    w.put(key, layers, sum(int(x.nbytes) for x in layers))
    w.set_disk(disk / "win", _Budget(), tag=None)
    assert not w.persist(key)
    w.set_disk(disk / "win", _Budget(allow=False), tag=(5,))
    assert not w.persist(key) and w.stats()["disk_skipped"] >= 1


def test_disk_store_keeps_the_newest_within_its_cap(disk, monkeypatch):
    import time

    w.set_disk(disk / "win", _Budget(), tag=(5,))
    w.set_budget(10_000)
    sizes = []
    for i in range(3):
        key = w.key_for(7, [i])
        layers = _window(i)
        w.put(key, layers, sum(int(x.nbytes) for x in layers))
        assert w.persist(key)
        w.flush()
        sizes.append(next(iter(sorted((disk / "win").glob("*.npy")))).stat().st_size)
        time.sleep(0.01)
    monkeypatch.setattr(w, "DISK_CAP", sizes[0] * 2)
    w._disk_trim(disk / "win")
    assert len(list((disk / "win").glob("*.npy"))) == 2
    w.clear()
    assert w.get(w.key_for(7, [0])) is None  # the oldest went
    assert w.get(w.key_for(7, [2])) is not None


def test_reply_window_ends_at_the_decode_cache_and_seeds_the_next_turn():
    """Turn 2 resumes from turn 1's decode cache; without the reply's window the drafter
    started from the 26 new positions only (half the acceptance)."""
    mx = pytest.importorskip("mlx.core")
    w.set_budget(1 << 30)
    keep, layers, width = 4, 2, 3
    # prompt window: positions 7..9 (3 rows) of a 10-token prompt; columns = 2 layers x 3
    prompt = mx.arange(3 * layers * width, dtype=mx.float32).reshape(
        1, 3, layers * width
    )
    tail = w.DecodeTail(prompt, 10, keep, layers)
    for step in range(70):  # crosses the 64-round trim
        tail.add(mx.full((1, 2, layers * width), float(step)), 2)
    w.finish_decode(tail)
    ids = list(range(10 + 140))
    assert not w.store_decode_window(ids[:-1], 7)  # the end must match the cache
    w.finish_decode(tail)
    assert w.store_decode_window(ids, 7)
    got = w.get(
        w.key_for(7, ids)
    )  # what the driver's APC restore seeds the next turn with
    assert got is not None
    assert len(got) == layers and got[0].shape == (1, keep, width)
    assert (
        got[1][0, -1].tolist() == [69.0] * width
        and got[0][0, 0].tolist() == [68.0] * width
    )
