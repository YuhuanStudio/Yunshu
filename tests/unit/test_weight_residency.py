from yunshu_engine import weight_residency as wr


def test_no_tables_reports_not_engaged(monkeypatch):
    monkeypatch.setattr(wr, "_tables", lambda: [])
    out = wr.ple_lookup_stats()
    assert out["tables"] == 0 and out["engaged"] is False


def test_sums_tables(monkeypatch):
    from types import SimpleNamespace as NS

    def mk(n):
        return NS(
            stats=NS(
                lookups=n,
                rows=2 * n,
                cache_hits=1,
                cache_misses=n,
                bytes_read=100 * n,
                elapsed_seconds=0.5,
            )
        )

    monkeypatch.setattr(wr, "_tables", lambda: [mk(1), mk(3)])
    out = wr.ple_lookup_stats()
    assert out["lookups"] == 4 and out["rows"] == 8 and out["bytes_read"] == 400
    assert out["elapsed_seconds"] == 1.0 and out["engaged"] is True
