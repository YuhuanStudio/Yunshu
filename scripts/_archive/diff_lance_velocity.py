"""Lance t2i velocity-divergence harness — pinpoints the flow-matching bug.

Rigorous isolation (VALIDATION_REPORT §133) established:
  • the Wan VAE port is correct (sharp encode→decode roundtrip, L1 0.07);
  • all diffusion projections load (non-zero, names match);
  • the Euler sign is correct (sign=-1 beats +1);
  • yet ``LanceModel.t2i_velocity`` produces a *broken* velocity — it degrades a
    known-good, lightly-noised latent (L1 0.07 → 0.42) instead of denoising it.

So the bug is a structural/computational subtlety in the MLX velocity forward
(attention routing / norm / RoPE), with correct weights. Finding it needs a
layer-by-layer comparison against the reference PyTorch forward — which requires
Lance's flash-attn/CUDA env (a Linux/NVIDIA box).

This harness makes that diff reproducible:

  # 1. On THIS machine (MLX) — export the fixed test vector + my intermediates:
  PYTHONPATH=python uv run python scripts/diff_lance_velocity.py --mlx-dump /tmp/mlx.npz

  # 2. On the CUDA box, in the Lance repo env — load the SAME test vector and
  #    capture the reference per-layer latent hidden states (see _REFERENCE_RECIPE
  #    below) into /tmp/ref.npz with matching keys ("layer_{i}", "velocity").

  # 3. Back here — report the first diverging layer:
  PYTHONPATH=python uv run python scripts/diff_lance_velocity.py --diff /tmp/mlx.npz /tmp/ref.npz

The fixed test vector (seed 0) is written into the MLX dump so both sides operate
on byte-identical inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

LANCE = Path("/Volumes/P5Plus/models/Lance-3B-bf16")

_REFERENCE_RECIPE = """\
Reference capture (run inside the Lance repo, torch env), for ONE velocity step:

  import numpy as np, torch
  tv = np.load('/tmp/mlx.npz')
  x_t   = torch.tensor(tv['x_t'])          # (1, 256, 48)
  t     = torch.tensor(tv['t'])            # (1,)  flow timestep
  ids   = torch.tensor(tv['prompt_ids'])   # (1, L) chat-templated, ends <|vision_start|>
  pos   = torch.tensor(tv['latent_pos'])   # (256,) = h*32+w

  # text prefill (understanding expert, causal) -> per-layer KV  [the model's
  # normal Qwen2.5 forward]; then ONE gen step where the latents are the query:
  #   lat = vae2llm(x_t) + time_embedder(t)[:,None,:] + latent_pos_embed(pos)[None]
  #   for each layer: latents attend bidirectionally to [text_kv ++ latent_kv]
  #                   via the *_moe_gen* projections; record lat hidden AFTER each layer
  #   v = llm2vae(norm_moe_gen(lat))
  # Save: np.savez('/tmp/ref.npz', velocity=v.numpy(), **{f'layer_{i}': h_i.numpy()})

Match the MLX dump keys exactly: 'layer_0'..'layer_{N-1}' (latent hidden after
each decoder layer, shape (1,256,2048)) and 'velocity' (1,256,48).
"""


def _load_mlx_model():
    from yunshu_engine.lance_llm import LanceModel
    cfg = json.loads((LANCE / "llm_config.json").read_text())
    cfg["vocab_size"] = 151936
    m = LanceModel(cfg)
    w = mx.load(str(LANCE / "model.safetensors"))
    m.load_weights([(k, v) for k, v in w.items()], strict=False)
    mx.eval(m.parameters())
    return m


def _test_vector():
    """Fixed, reproducible single-step input (seed 0)."""
    from transformers import AutoTokenizer

    from yunshu_engine.lance_llm import build_t2i_prompt
    tok = AutoTokenizer.from_pretrained(str(LANCE))
    ids = build_t2i_prompt(tok, "a photo of a red apple on a wooden table")
    mx.random.seed(0)
    x_t = mx.random.normal((1, 256, 48))
    t = mx.array([0.5])
    pos = (mx.arange(16)[:, None] * 32 + mx.arange(16)[None, :]).reshape(-1)
    return ids, x_t, t, pos


def mlx_dump(out_path: str) -> int:
    if not LANCE.exists():
        print("SKIP: Lance not mounted")
        return 0
    m = _load_mlx_model()
    ids, x_t, t, pos = _test_vector()

    # Replicate t2i_velocity but record per-layer latent hidden states.
    text_h = m.embed_tokens(ids)
    lat_h = (
        m.vae_in_proj.vae2llm(x_t)
        + m.time_embedder(t)[:, None, :]
        + m.latent_pos_embed(pos)[None]
    )
    dump: dict[str, np.ndarray] = {}
    for i, layer in enumerate(m.layers):
        text_h, lat_h = layer.forward_mixed(text_h, lat_h)
        mx.eval(lat_h)
        dump[f"layer_{i}"] = np.asarray(lat_h)
    velocity = m.llm2vae(m.norm_moe_gen(lat_h))
    mx.eval(velocity)
    dump["velocity"] = np.asarray(velocity)
    # embed the byte-identical test vector so the reference uses the same inputs
    dump["x_t"] = np.asarray(x_t)
    dump["t"] = np.asarray(t)
    dump["prompt_ids"] = np.asarray(ids)
    dump["latent_pos"] = np.asarray(pos)
    np.savez(out_path, **dump)
    print(f"wrote {out_path}: {len(m.layers)} layers + velocity + test vector")
    print(f"  velocity std={velocity.std().item():.4f} (broken forward: see §133)")
    print("\nNext, on the CUDA box:\n" + _REFERENCE_RECIPE)
    return 0


def diff(mlx_path: str, ref_path: str) -> int:
    a = np.load(mlx_path)
    b = np.load(ref_path)
    layers = sorted(
        (k for k in a.files if k.startswith("layer_")),
        key=lambda k: int(k.split("_")[1]),
    )
    print(f"{'key':>12} | {'max|Δ|':>10} | {'rel':>8}")
    first = None
    for k in [*layers, "velocity"]:
        if k not in b.files:
            print(f"{k:>12} | (missing in reference dump)")
            continue
        da = a[k].astype(np.float64)
        db = b[k].astype(np.float64)
        if da.shape != db.shape:
            print(f"{k:>12} | SHAPE {da.shape} vs {db.shape}")
            first = first or k
            continue
        mad = float(np.abs(da - db).max())
        rel = mad / (float(np.abs(db).max()) + 1e-8)
        flag = "  <-- FIRST DIVERGENCE" if (rel > 0.05 and first is None) else ""
        if rel > 0.05 and first is None:
            first = k
        print(f"{k:>12} | {mad:10.5f} | {rel:8.4f}{flag}")
    print(f"\nfirst diverging layer: {first or 'none (match)'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx-dump", metavar="OUT.npz")
    ap.add_argument("--diff", nargs=2, metavar=("MLX.npz", "REF.npz"))
    args = ap.parse_args()
    if args.mlx_dump:
        return mlx_dump(args.mlx_dump)
    if args.diff:
        return diff(*args.diff)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
