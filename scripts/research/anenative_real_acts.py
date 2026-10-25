"""Real post-RMSNorm activations of the 27B at prefill: their scale, and the ANE split's error on them.

Runs a text prompt through the language model (no ANE), captures the normalized rows each dense MLP
receives, reports their RMS / max / p99.9, then for sampled layers compares the Splash-style W8A8
ANE share against the GPU alone on exactly those rows (RMS error of what the ANE adds, the
quantity the engine's verify bounds at 10%).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from anenative_splash_micro import load_layers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp")
    p.add_argument("--rows", type=int, default=1024)
    p.add_argument("--units", type=int, default=8)
    p.add_argument("--sample", default="0,8,16,24,32,40,48,56,63")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--corpus", type=Path, default=Path("/Volumes/P5Plus/yunshu-build/snapshot014/corpus"))
    a = p.parse_args()

    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.models.qwen3_5.language import Qwen3_5MLP

    from yunshu_engine.ane.ffn_split import AneFfn
    from yunshu_engine.kernels import nax_prefill

    if not nax_prefill.enable():
        raise SystemExit("NAX prefill unavailable")
    model, processor = load(a.model)
    tok = getattr(processor, "tokenizer", processor)
    text = ""
    for path in sorted(a.corpus.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".md", ".txt"):
            text += path.read_text(errors="ignore")[:3000] + "\n\n"
            if len(text) > 12 * a.rows * 4:
                break
    ids = tok.encode(text)[: a.rows]
    sample = [int(v) for v in a.sample.split(",")]
    captured: dict[int, object] = {}
    original = Qwen3_5MLP.__call__
    layers_lm = model.language_model.model.layers
    index = {id(layer.mlp): i for i, layer in enumerate(layers_lm)}

    def capture(self, x):
        i = index.get(id(self))
        if i in sample and i not in captured:
            captured[i] = mx.array(x[0])
        return original(self, x)

    Qwen3_5MLP.__call__ = capture
    inputs = mx.array(ids)[None]
    out = model.language_model(inputs)
    mx.eval(out.logits if hasattr(out, "logits") else out)
    Qwen3_5MLP.__call__ = original

    records = []
    raw = load_layers(a.model, 0, 64, True)
    for layer in sample:
        x = captured[layer].astype(mx.bfloat16)
        xf = np.array(x.astype(mx.float32))
        stats = {
            "layer": layer,
            "rms": float(np.sqrt((xf**2).mean())),
            "max": float(np.abs(xf).max()),
            "p999": float(np.quantile(np.abs(xf), 0.999)),
        }
        pair = [raw[layer], raw[layer]]
        split = AneFfn(pair, a.units, a.cache, function_rows=[((a.rows + 127) // 128) * 128])
        split.set_minimum_rows(split.rows[0])
        rows = x.shape[0]
        split.begin()
        out = split.add(0, x, None, rows)
        mx.eval(out)
        ok = split.finish()
        alone = split.gpu_alone(0, x, None)
        part = split._gpu_part(0, x, None)
        mx.eval(alone, part)
        o, g, pt = (np.array(t.astype(mx.float32)) for t in (out, alone, part))
        stats["ane_ok"] = ok
        stats["rel_err_vs_ane_part"] = float(np.sqrt(((o - g) ** 2).sum() / ((g - pt) ** 2).sum()))
        stats["rel_err_whole_ffn"] = float(np.linalg.norm(o - g) / np.linalg.norm(g))
        records.append(stats)
        print(json.dumps(stats), flush=True)
        split.close()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(records, indent=1) + "\n")
    print(json.dumps({"complete": True}))


if __name__ == "__main__":
    main()
