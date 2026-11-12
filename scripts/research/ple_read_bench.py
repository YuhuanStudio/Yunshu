"""Where the first span's PLE read time goes: Python planning vs the reads themselves.

Opens the real Flash-Next PLE table, draws random unique row ids (a 2048-token span reads ~24K), and
times ``ple_parallel.parallel_read_rows`` (a) as is, (b) with the reads stubbed out (planning only),
and checks the rows equal upstream's ``_read_rows``. Output: JSON lines.

Usage: ple_read_bench.py MANIFEST [--rows 24000] [--threads 16,32,64] [--trials 3] [--check-rows 2000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return time.perf_counter() - t0, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--rows", type=int, default=24000)
    ap.add_argument("--threads", default="16", help="comma list")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--check-rows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)

    from mlx_vlm.models.qwen4_exp import ple_storage as ps

    from yunshu_engine import ple_parallel as pp

    table = ps.QuantizedMMapNGramEmbedding(a.manifest)
    rng = np.random.default_rng(a.seed)
    dtypes = ps._DTYPES

    def emit(**row):
        print(json.dumps(row), flush=True)

    real_chunk = pp._read_chunk
    for threads in [int(x) for x in a.threads.split(",")]:
        old = getattr(table, "_yunshu_pool", None)
        if old is not None:
            old.shutdown(wait=True)
        table._yunshu_pool = None  # a pool sized for this thread count
        for trial in range(a.trials):
            ids = np.unique(rng.integers(0, table.row_count, size=a.rows))
            pp._read_chunk = lambda chunk: None  # planning only
            plan_s, _ = timed(pp.parallel_read_rows, table, ids, threads, dtypes)
            pp._read_chunk = real_chunk
            first_s, _ = timed(pp.parallel_read_rows, table, ids, threads, dtypes)
            again_s, _ = timed(pp.parallel_read_rows, table, ids, threads, dtypes)
            emit(
                ev="parallel_read_rows",
                trial=trial,
                rows=len(ids),
                threads=threads,
                plan_only_s=plan_s,
                first_touch_s=first_s,  # fresh random rows: cold unless cached before
                second_touch_s=again_s,  # the same rows again: page cache + Python floor
            )
    check = np.unique(rng.integers(0, table.row_count, size=a.check_rows))
    mine = pp.parallel_read_rows(table, check, 16, dtypes)
    theirs = ps.QuantizedMMapNGramEmbedding(a.manifest)
    ref = type(theirs)._read_rows(theirs, check)
    equal = all(np.array_equal(x, y) for x, y in zip(mine, ref, strict=True))
    emit(ev="rows_equal_upstream", rows=len(check), equal=bool(equal))
    return 0 if equal else 1


if __name__ == "__main__":
    sys.exit(main())
