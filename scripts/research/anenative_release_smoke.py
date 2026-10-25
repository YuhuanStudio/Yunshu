"""Idle release of the ANE program: unload, reload, and the split still verifies."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from anenative_splash_micro import load_layers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp")
    p.add_argument("--cache", type=Path, required=True)
    a = p.parse_args()
    from yunshu_engine.ane.ffn_split import AneFfn
    from yunshu_engine.kernels import nax_prefill

    if not nax_prefill.enable():
        raise SystemExit("NAX prefill unavailable")
    layers = load_layers(a.model, 40, 3, True)
    split = AneFfn(layers, 8, a.cache, function_rows=[512, 1024])
    before = split.verify()
    t = time.perf_counter()
    assert split.pool.unload(), "unload failed"
    unload_s = time.perf_counter() - t
    time.sleep(2.0)
    t = time.perf_counter()
    assert split.pool.load(), "load failed"
    load_s = time.perf_counter() - t
    after = split.verify()
    assert not split.retired()
    print(
        json.dumps(
            {
                "unload_s": unload_s,
                "load_s": load_s,
                "error_before": before,
                "error_after": after,
                "complete": True,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
