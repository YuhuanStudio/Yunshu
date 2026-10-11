"""Op census and ms of one DFlash drafter round (random weights, real config): stacked vs not."""

import json
import re
import sys
import tempfile
import time
from pathlib import Path

DRAFTER = "/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2"


def nodes(path: str) -> int:
    return len(re.findall(r"\[label=", Path(path).read_text()))


def main() -> int:
    import mlx.core as mx
    import mlx.nn as nn

    sys.path.insert(0, "python")
    from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import (
        DFlashConfig,
        DFlashDraftModel,
    )

    from yunshu_engine.dflash_tree import quantize_drafter, stack_projections

    cfg = DFlashConfig.from_dict(
        json.loads((Path(DRAFTER) / "config.json").read_text())
    )
    out = {}
    for stacked in (False, True):
        m = DFlashDraftModel(cfg)
        m.embed_tokens = nn.Embedding(1000, cfg.hidden_size)
        mx.eval(m.parameters())
        quantize_drafter(m, 8)
        if stacked:
            stack_projections(m)
        cache = m.make_cache()
        ids = mx.array([[5] + [3] * 15], dtype=mx.int32)
        ctx = mx.random.normal(
            (1, 64, len(cfg.target_layer_ids) * cfg.hidden_size)
        ).astype(mx.bfloat16)
        mx.eval(ids, ctx)
        for _ in range(3):
            mx.eval(m._hidden(ids, ctx, cache))
        t0 = time.perf_counter()
        for _ in range(20):
            mx.eval(m._hidden(ids, ctx, cache))
        ms = (time.perf_counter() - t0) * 1000 / 20
        with tempfile.NamedTemporaryFile(suffix=".dot", delete=False) as f:
            mx.export_to_dot(f.name, m._hidden(ids, ctx, cache))
        out["stacked" if stacked else "plain"] = {
            "ms": round(ms, 2),
            "graph_nodes": nodes(f.name),
        }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
