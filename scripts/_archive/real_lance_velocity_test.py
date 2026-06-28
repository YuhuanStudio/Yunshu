"""Run the REAL Lance reference full model to test the t2i sequence setup (#176).

The per-layer math is proven faithful (§140); the bug is in generation-level
setup. This builds the FULL real reference stack (36 real Qwen2MoT layers + the
diffusion projections, all on CPU via lance_ref_env) and measures the velocity's
cosine similarity to the ground-truth flow-matching velocity (noise − x0) on a
known latent, comparing two sequence layouts:

  (A) bare:    [text_prompt] [latents]                       (the MLX port's layout)
  (B) bracketed: [text_prompt <vision_start>] [latents] [<vision_end>]
                 (reference: VAE tokens flanked by vision tokens, lance.py L1599)

If (B) gives a markedly higher cosine (→ correct denoising direction), the missing
bracket structure is the bug and porting it fixes Lance t2i. Decisive, on-machine.

Run: PYTHONPATH=. uv run python scripts/real_lance_velocity_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

LANCE = Path("/Volumes/P5Plus/models/Lance-3B-bf16")


def main() -> int:
    if not LANCE.exists():
        print("SKIP: Lance not mounted")
        return 0
    import torch
    from scripts.lance_ref_env import load_reference_qwen2_navit
    nav = load_reference_qwen2_navit("reference/Lance")
    from modeling.qwen2.configuration_qwen2 import Qwen2Config

    cfg_d = json.loads((LANCE / "llm_config.json").read_text())
    cfg = Qwen2Config(
        vocab_size=cfg_d["vocab_size"], hidden_size=cfg_d["hidden_size"],
        num_attention_heads=cfg_d["num_attention_heads"],
        num_key_value_heads=cfg_d["num_key_value_heads"],
        num_hidden_layers=cfg_d["num_hidden_layers"],
        intermediate_size=cfg_d["intermediate_size"], rms_norm_eps=cfg_d["rms_norm_eps"],
        rope_theta=cfg_d["rope_theta"], max_position_embeddings=cfg_d["max_position_embeddings"],
        attention_dropout=0.0, qk_norm=True, layer_module="Qwen2MoTDecoderLayer",
        apply_qwen_2_5_vl_pos_emb=False, rope_scaling=None, _attn_implementation="eager",
    )
    w = mx.load(str(LANCE / "model.safetensors"))

    def tt(name):
        return torch.from_numpy(np.asarray(w[name].astype(mx.float32)))

    # build 36 real layers + load weights
    layers = []
    for i in range(cfg.num_hidden_layers):
        ly = nav.Qwen2MoTDecoderLayer(cfg, layer_idx=i).eval().to(torch.bfloat16)
        sd = {k[len(f"layers.{i}."):]: tt(k) for k in w if k.startswith(f"layers.{i}.")}
        ly.load_state_dict(sd, strict=False)
        layers.append(ly)
    norm_g = tt("norm_moe_gen.weight")
    hd = cfg.hidden_size // cfg.num_attention_heads

    def lin(x, name, bias=True):
        y = x @ tt(f"{name}.weight").T
        if bias and f"{name}.bias" in w:
            y = y + tt(f"{name}.bias")
        return y

    def rms(x, ww, eps=1e-6):
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype) * ww

    def velocity(text_ids, x_t, t_val, bracket: bool):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(LANCE))
        vs = tok.convert_tokens_to_ids("<|vision_start|>")
        ve = tok.convert_tokens_to_ids("<|vision_end|>")
        text = list(text_ids)
        if bracket:
            text = text + [vs]
        text_emb = tt("embed_tokens.weight")[torch.tensor(text)]
        nt = text_emb.shape[0]
        nlat = x_t.shape[0]
        # latent embedding
        half = 128
        fr = torch.exp(-np.log(10000.0) * torch.arange(half).float() / half)
        ts = torch.tensor([t_val])
        a = ts[:, None] * fr[None]
        sinus = torch.cat([torch.cos(a), torch.sin(a)], -1)
        temb = lin(torch.nn.functional.silu(lin(sinus, "time_embedder.proj_in")), "time_embedder.proj_out")
        lpos_ids = (torch.arange(16)[:, None] * 32 + torch.arange(16)[None, :]).reshape(-1)
        lat_emb = lin(x_t, "vae_in_proj.vae2llm") + temb + tt("latent_pos_embed.pos_embed")[lpos_ids]
        end_emb = tt("embed_tokens.weight")[torch.tensor([ve])] if bracket else None

        parts = [text_emb, lat_emb] + ([end_emb] if bracket else [])
        seq = torch.cat(parts, 0).to(torch.bfloat16)
        # positions: text 0..nt-1 ; latents 1000.. ; vision_end after latents
        # REAL reference (validation_gen): ALL latent tokens share ONE RoPE
        # position (the vision_start position); spatial info comes only from the
        # additive latent_pos_embed. (Verified by extracting packed_position_ids
        # from the real ValidationDataset — latents all at a single constant.)
        lat_const = nt - 1  # share the last text (vision_start) position
        positions = [torch.arange(nt), torch.full((nlat,), lat_const)]
        if bracket:
            positions.append(torch.tensor([nt]))
        position_ids = torch.cat(positions)
        inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, hd, 2).float() / hd))
        f2 = torch.outer(position_ids.float(), inv)
        emb = torch.cat([f2, f2], -1)
        pemb = (emb.cos(), emb.sin())
        L = seq.shape[0]
        txt_idx = torch.cat([torch.arange(nt)] + ([torch.tensor([L - 1])] if bracket else []))
        vae_idx = torch.arange(nt, nt + nlat)
        ql = torch.tensor([L], dtype=torch.int32)
        x = seq
        with torch.no_grad():
            for ly in layers:
                x, _ = ly(packed_query_sequence=x, query_lens=ql,
                          packed_query_position_embeddings=pemb,
                          packed_query_indexes=torch.arange(L), past_key_values=None,
                          key_values_lens=None, packed_key_value_indexes=None,
                          update_past_key_values=False, is_causal=False, mode="gen",
                          packed_vae_token_indexes=vae_idx, packed_text_indexes=txt_idx,
                          apply_qwen_2_5_vl_pos_emb=False)
            lat = rms(x[vae_idx], norm_g.to(torch.bfloat16))
            v = lin(lat.float(), "llm2vae")
        return v

    # known latent + ground-truth velocity
    from yunshu_engine.wan_vae import load_wan_vae
    vae = load_wan_vae(str(LANCE / "vae.safetensors"))
    hh = ww_ = 256
    img = np.zeros((hh, ww_, 3), np.float32)
    for i in range(0, hh, 16):
        img[i:i + 8] += [1, 0, 0]
    for j in range(0, ww_, 32):
        img[:, j:j + 16] += [0, 0, 1]
    img = np.clip(img, 0, 1) * 2 - 1
    x0 = np.asarray(vae.encode(mx.array(img)[None, None]).reshape(256, 48).astype(mx.float32))
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(x0.shape).astype(np.float32)
    t_val = 0.5
    x_t = (1 - t_val) * x0 + t_val * noise
    true_v = (noise - x0).ravel()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(LANCE))
    g = tok.convert_tokens_to_ids
    ids = [g("<|im_start|>")] + tok("user\na colorful checkerboard pattern", add_special_tokens=False)["input_ids"] + [g("<|im_end|>")] + tok("\n", add_special_tokens=False)["input_ids"] + [g("<|im_start|>")] + tok("assistant\n", add_special_tokens=False)["input_ids"]

    if "--generate" not in sys.argv:
        for bracket in (False, True):
            v = velocity(ids, torch.tensor(x_t), t_val, bracket).numpy().ravel()
            cos = float(np.dot(v, true_v) / (np.linalg.norm(v) * np.linalg.norm(true_v) + 1e-9))
            print(f"REAL reference velocity cos(v, true_v)  bracket={bracket}: {cos:+.3f}", flush=True)
        return 0

    # --- full generation from pure noise using the REAL reference velocity ---
    steps, shift, cfg_scale = 20, 4.0, 4.0
    rng2 = np.random.default_rng(0)
    x = torch.tensor(rng2.standard_normal((256, 48)).astype(np.float32))
    # proper t2i conditioning: SP1 system prompt (data/common.py generate_system_prompt)
    sysp = ("You are a helpful assistant. Describe the image by detailing the color, "
            "quantity, text, shape, size, texture, spatial relationships of the objects "
            "and background:")
    nl = tok("\n", add_special_tokens=False)["input_ids"]

    def chat(sp, up):
        s = [g("<|im_start|>")] + tok("system\n" + sp, add_special_tokens=False)["input_ids"] + [g("<|im_end|>")] + nl
        s += [g("<|im_start|>")] + tok("user\n" + up, add_special_tokens=False)["input_ids"] + [g("<|im_end|>")] + nl
        s += [g("<|im_start|>")] + tok("assistant\n", add_special_tokens=False)["input_ids"]
        return s
    ids = chat(sysp, "a photo of a red apple on a wooden table")
    uncond = chat(sysp, "")
    tlin = torch.linspace(1, 0, steps + 1)
    tlin = shift * tlin / (1 + (shift - 1) * tlin)
    dts = (tlin[:-1] - tlin[1:]).tolist()
    tl = tlin[:-1].tolist()
    for i in range(steps):
        vc = velocity(ids, x, tl[i], bracket=True)
        vu = velocity(uncond, x, tl[i], bracket=True)
        v = vu + cfg_scale * (vc - vu)
        x = x - v * dts[i]
        print(f"step {i+1}/{steps} t={tl[i]:.3f} |x|={float(x.norm()):.1f}", flush=True)
    lat = mx.array(x.numpy())[None, None].reshape(1, 1, 16, 16, 48)
    img = vae.decode(lat)
    mx.eval(img)
    arr = np.asarray(img)[0, 0]
    from PIL import Image
    Image.fromarray(((np.clip(arr, -1, 1) + 1) * 127.5).astype(np.uint8)).save("/tmp/real_ref_gen.png")
    print(f"REAL reference full generation: img std={arr.std():.3f}, saved /tmp/real_ref_gen.png", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
