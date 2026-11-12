"""PLE prefetch: row ids match the model, prefetched rows equal synchronous reads, cancel stops the thread."""

import json
import threading

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
lang = pytest.importorskip("mlx_vlm.models.qwen4_exp.language")
ps = pytest.importorskip("mlx_vlm.models.qwen4_exp.ple_storage")
from yunshu_engine import ple_prefetch as pp  # noqa: E402

ROW_WIDTH = 160
GROUP = 32
SHARD_ROWS = (60, 50, 90)  # 200 rows: every id below is in range


class _Recorder:
    def __call__(self, ids):
        self.ids = ids
        return mx.zeros((*ids.shape, ROW_WIDTH))


class _Cache:
    """ArraysCache's slot 3: the previous ``context_len`` tokens."""

    def __init__(self, window):
        self.window = window

    def __getitem__(self, index):
        return self.window if index == 3 else None

    def update_window(self, index, history, n):
        self.window = history[:, -n:]


def _model_module():
    mod = lang.Qwen4ExpNGramEmbedding.__new__(lang.Qwen4ExpNGramEmbedding)
    nn.Module.__init__(mod)
    heads, ngram, eos, vocab = 8, 3, 7, 5000
    mod.ngram_size, mod.context_len, mod.heads_per_ngram = ngram, ngram - 1, heads
    mod.ngram_heads = (ngram - 1) * heads
    mod.eos_token_id = eos
    mod.layer_multipliers = lang._build_layer_multipliers(vocab, ngram, 0, 1234)
    sizes = [
        lang._find_nth_prime_after(1000 - 1, i + 1) for i in range(mod.ngram_heads)
    ]
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    mod.ngram_heads_vocab_sizes = mx.array(sizes, dtype=mx.int64)
    mod.ngram_heads_offsets = mx.array(offsets.tolist(), dtype=mx.int64)
    mod.ngram_embedding = _Recorder()
    return mod


def _tokens(seed, n):
    rng = np.random.default_rng(seed)
    toks = rng.integers(0, 4000, size=n)
    toks[rng.random(n) < 0.05] = 7  # eos inside the span exercises the segment rule
    return toks.astype(np.int64)


def test_span_ids_match_model():
    mod = _model_module()
    params = pp.params_from_module(mod)
    first = _tokens(0, 40)
    mod(mx.array(first[None], dtype=mx.int32), None)
    got = np.array(mod.ngram_embedding.ids[0]).reshape(len(first), -1)
    history = np.concatenate([[7, 7], first])
    assert np.array_equal(pp.span_row_ids(history, params), got)

    second = _tokens(1, 30)
    cache = _Cache(mx.array(first[-2:][None], dtype=mx.int64))
    mod(mx.array(second[None], dtype=mx.int32), cache)
    got = np.array(mod.ngram_embedding.ids[0]).reshape(len(second), -1)
    history = np.concatenate([first[-2:], second])
    assert np.array_equal(pp.span_row_ids(history, params), got)


def _build(tmp_path):
    rng = np.random.default_rng(0)
    entries, start = [], 0
    for index, count in enumerate(SHARD_ROWS):
        entry = {"row_start": start, "row_count": count}
        for name, dtype, width in (
            ("weight", "<u4", ROW_WIDTH // 8),
            ("scales", "<u2", ROW_WIDTH // GROUP),
            ("biases", "<u2", ROW_WIDTH // GROUP),
        ):
            data = rng.integers(0, 2**15, size=(count, width)).astype(dtype)
            fname = f"s{index}_{name}.bin"
            pad = 13
            (tmp_path / fname).write_bytes(b"\xaa" * pad + data.tobytes())
            entry[name] = {
                "dtype": {"<u4": "U32", "<u2": "BF16"}[dtype],
                "file": fname,
                "offset": pad,
                "shape": [count, width],
            }
        entries.append(entry)
        start += count
    manifest = {
        "version": 2,
        "source_root": ".",
        "row_width": ROW_WIDTH,
        "quantization": {"bits": 4, "group_size": GROUP, "mode": "affine"},
        "shards": entries,
        "cache_rows": 0,
    }
    path = tmp_path / "ple.json"
    path.write_text(json.dumps(manifest))
    return path


def _params(eos=0):
    # ids in [0, 192): every head size 12, offsets 12 * head
    rng = np.random.default_rng(3)
    return pp.NGramParams(
        ngram_size=3,
        heads_per_ngram=8,
        context_len=2,
        eos=eos,
        multipliers=rng.integers(1, 2**20, size=3, dtype=np.int64) * 2 + 1,
        sizes=np.full(16, 12, dtype=np.int64),
        offsets=np.arange(16, dtype=np.int64) * 12,
    )


def _pair(tmp_path):
    manifest = _build(tmp_path)
    reference = ps.QuantizedMMapNGramEmbedding(manifest)
    table = ps.QuantizedMMapNGramEmbedding(manifest)
    return reference, table


def test_prefetched_rows_identical_and_counters_match(tmp_path):
    reference, table = _pair(tmp_path)
    params = _params()
    pf = pp.PLEPrefetch(table, params)
    calls = []
    original = pf._original
    pf._original = lambda ids: (calls.append(len(ids)), original(ids))[1]

    ids = _tokens(5, 64)
    later = _tokens(6, 60)
    assert pf.submit_span(ids.tolist(), 0, len(ids))
    assert pf.submit_span(later.tolist(), 0, len(later))
    pf.begin(len(ids), 0, len(ids))
    mine = table._read_rows(
        np.unique(pp.span_row_ids(np.concatenate([[0, 0], ids]), params))
    )
    theirs = reference._read_rows(
        np.unique(pp.span_row_ids(np.concatenate([[0, 0], ids]), params))
    )
    for a, b in zip(mine, theirs, strict=True):
        assert a.dtype == b.dtype and np.array_equal(a, b)
    assert calls == []  # every row came from the buffer
    assert table._misses == reference._misses and table._bytes == reference._bytes

    # the second span: its rows are read in the same way and still match
    later_ids = np.unique(pp.span_row_ids(np.concatenate([[0, 0], later]), params))
    pf.begin(len(later), 0, len(later))
    mine = table._read_rows(later_ids)
    theirs = reference._read_rows(later_ids)
    for a, b in zip(mine, theirs, strict=True):
        assert np.array_equal(a, b)
    assert table._misses == reference._misses and table._bytes == reference._bytes
    pf.close()


def test_rows_missing_from_buffer_are_read_synchronously(tmp_path):
    reference, table = _pair(tmp_path)
    params = _params()
    pf = pp.PLEPrefetch(table, params)
    prefetched = _tokens(8, 16)
    assert pf.submit_span(prefetched.tolist(), 0, len(prefetched))
    pf.begin(len(prefetched), 0, len(prefetched))
    request = np.arange(0, 192, 3, dtype=np.int64)  # mostly not prefetched
    mine = table._read_rows(request)
    theirs = reference._read_rows(request)
    for a, b in zip(mine, theirs, strict=True):
        assert np.array_equal(a, b)
    assert table._misses == reference._misses
    pf.close()


def test_buffer_is_bounded(tmp_path):
    _, table = _pair(tmp_path)
    pf = pp.PLEPrefetch(table, _params())
    for k in range(5):
        pf.submit_span(_tokens(k, 32).tolist(), 0, 32)
    assert len(pf._pending) <= pp.MAX_PENDING
    pf.close()


def test_cancel_stops_background_reads(tmp_path):
    _, table = _pair(tmp_path)
    reader = pp._RowReader(table)
    stop = threading.Event()
    stop.set()
    with pytest.raises(pp.PrefetchCancelledError):
        reader.read(np.arange(0, 192, dtype=np.int64), stop)
    reader.close()

    pf = pp.PLEPrefetch(table, _params())
    pf.submit_span(_tokens(9, 64).tolist(), 0, 64)
    pf.cancel()
    assert not pf._pending
    pf.close()
    assert not [t for t in threading.enumerate() if t.name.startswith("ple-prefetch")]
    assert "_read_rows" not in table.__dict__


def test_read_error_reaches_the_caller(tmp_path):
    manifest = _build(tmp_path)
    table = ps.QuantizedMMapNGramEmbedding(manifest)
    pf = pp.PLEPrefetch(table, _params())
    ids = _tokens(10, 32)
    (tmp_path / "s0_weight.bin").write_bytes(
        b"\xaa" * 4
    )  # truncated: short reads from shard 0
    assert pf.submit_span(ids.tolist(), 0, len(ids))
    pf.begin(len(ids), 0, len(ids))
    request = np.unique(pp.span_row_ids(np.concatenate([[0, 0], ids]), _params()))
    with pytest.raises(OSError):
        table._read_rows(request)
    pf.close()


def test_later_span_entry_survives_the_current_span_and_overlapping_rows(tmp_path):
    reference, table = _pair(tmp_path)
    params = _params()
    pf = pp.PLEPrefetch(table, params)
    ids = _tokens(11, 96).tolist()
    assert pf.submit_span(ids, 0, 48)
    assert pf.submit_span(ids, 48, 96)
    assert not pf.submit_span(ids, 48, 96)  # already pending
    pf.begin(len(ids), 0, 48)
    assert [e.key for e in pf._pending] == [(96, 48, 96)]  # the next span waits
    window = np.concatenate([[0, 0], ids[:48]])
    request = np.unique(pp.span_row_ids(window, params))
    for a, b in zip(
        table._read_rows(request), reference._read_rows(request), strict=True
    ):
        assert np.array_equal(a, b)
    assert [e.key for e in pf._pending] == [
        (96, 48, 96)
    ]  # rows shared with span 0 did not eat it
    pf.begin(len(ids), 48, 96)
    assert not pf._pending and pf._current.key == (96, 48, 96)
    pf.begin(50, 0, 50)  # another prompt: nothing matches, nothing stale is kept
    assert pf._current is None
    pf.close()


def test_parallel_reader_equals_serial(tmp_path):
    _, table = _pair(tmp_path)
    from concurrent.futures import ThreadPoolExecutor

    reader = pp._RowReader(table)
    rows = np.arange(0, 192, dtype=np.int64)
    serial = reader.read(rows, threading.Event())
    with ThreadPoolExecutor(4) as pool:
        parallel = reader.read(rows, threading.Event(), pool, 4)
    for a, b in zip(serial, parallel, strict=True):
        assert np.array_equal(a, b)
    reader.close()


def test_attach_is_none_without_an_ssd_ple_table():
    from types import SimpleNamespace

    no_ple = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(), SimpleNamespace(ple=None)])
    )
    assert pp.attach(no_ple) is None


def test_blocked_time_and_row_counts_are_reported(tmp_path):
    reference, table = _pair(tmp_path)
    params = _params()
    pf = pp.PLEPrefetch(table, params)
    ids = _tokens(12, 32)
    assert pf.submit_span(ids.tolist(), 0, len(ids))
    pf.begin(len(ids), 0, len(ids))
    request = np.unique(pp.span_row_ids(np.concatenate([[0, 0], ids]), params))
    table._read_rows(request)
    blocked, asked, served = pf.take_stats()
    assert asked == len(request) and served == len(request) and blocked > 0
    assert pf.take_stats() == (0.0, 0, 0)  # reset by the read
    table._read_rows(np.asarray([1, 2, 3]))  # not prefetched: counted as asked only
    assert pf.take_stats()[1] == 3
    pf.close()


def test_stats_of_a_span_exclude_reads_made_before_it_began(tmp_path):
    _, table = _pair(tmp_path)
    pf = pp.PLEPrefetch(table, _params())
    table._read_rows(np.asarray([1, 2, 3]))  # decode-time reads between two prompts
    pf.begin(50, 0, 50)
    assert pf.take_stats() == (0.0, 0, 0)
    assert pf.before[1] == 3 and pf.before[0] > 0  # still reported as "before"
    table._read_rows(np.asarray([4, 5]))
    assert pf.take_stats()[1] == 2
    pf.close()
