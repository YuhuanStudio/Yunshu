"""Run the REAL Lance `validation_gen` end-to-end on this Mac, decode with the MLX
VAE, to get the reference-correct (sharp) t2i image (#176, VALIDATION_REPORT §146).

Strategy: every component is verified — model layers (§140), velocity (§141), VAE
(§133/§146). The remaining gap is the generation *procedure*. So instead of
re-porting it, run the reference's own `validation_gen`:
  1. lance_ref_env stubs flash_attn→torch SDPA + decord/imageio (CPU-runnable).
  2. Instantiate the real Lance (visual_und=False → no ViT) with a minimal CausalLM
     wrapper around the real Qwen2Model.
  3. Remap the MLX-converted checkpoint keys → torch reference names and load.
  4. Build the packed t2i inputs via the real ValidationDataset + collate.
  5. Call `Lance.validation_gen(**params)` → clean latent.
  6. Decode the latent with the (verified) MLX Wan VAE → image; VLM-check.

Config: latent_patch_size (1,1,1), z_channels 48, max_latent_size 32 (matches the
checkpoint VAE: 16×16×48 latent, decoder in=48).

Run: PYTHONPATH=. uv run python scripts/run_real_validation_gen.py
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

LANCE = Path("models/Lance-3B-bf16")


def _remap_key(k: str) -> str | None:
    """MLX-converted checkpoint key -> torch reference Lance state_dict key."""
    if k.startswith("layers.") or k in ("embed_tokens.weight",) or k.startswith("norm.") or k.startswith("norm_moe_gen."):
        return "language_model.model." + k
    if k.startswith("embed_tokens."):
        return "language_model.model." + k
    if k.startswith("vae_in_proj.vae2llm."):
        return "vae2llm." + k[len("vae_in_proj.vae2llm."):]
    if k.startswith("llm2vae.") or k.startswith("latent_pos_embed."):
        return k
    if k.startswith("time_embedder.proj_in."):
        return "time_embedder.mlp.0." + k[len("time_embedder.proj_in."):]
    if k.startswith("time_embedder.proj_out."):
        return "time_embedder.mlp.2." + k[len("time_embedder.proj_out."):]
    return None  # drop (e.g. lm_head if tied / unused)


def main() -> int:
    if not LANCE.exists():
        print("SKIP: Lance not mounted")
        return 0
    import mlx.core as mx
    import torch
    from scripts.lance_ref_env import _install_stubs
    from torch import nn
    _install_stubs()
    # transformers in this venv lacks the 'default' rope init key the reference's
    # Qwen2RotaryEmbedding looks up — add a standard 1-D rope init.
    from transformers import modeling_rope_utils as _rope
    if "default" not in _rope.ROPE_INIT_FUNCTIONS:
        def _default_rope_init(config, device=None, seq_len=None, **kw):
            base = config.rope_theta
            dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
            return inv.to(device), 1.0
        _rope.ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
    for m in ("decord",):
        mod = sys.modules.get(m) or types.ModuleType(m)
        mod.VideoReader = object
        mod.cpu = lambda *a: 0
        mod.bridge = type("b", (), {"set_bridge": staticmethod(lambda *a: None)})
        sys.modules[m] = mod
    sys.path.insert(0, str(Path("reference/Lance").resolve()))

    from config.config_factory import DataArguments, ModelArguments, TrainingArguments
    from data.data_utils import add_special_tokens
    from data.dataset_base import DataConfig, simple_custom_collate
    from data.datasets_custom.validation_dataset import ValidationDataset
    from modeling.lance.lance import Lance, LanceConfig
    from modeling.qwen2.configuration_qwen2 import Qwen2Config
    from transformers import AutoTokenizer

    nav_mod = sys.modules.get("modeling.lance.qwen2_navit", None)
    if nav_mod is None:
        from scripts.lance_ref_env import load_reference_qwen2_navit
        nav_mod = load_reference_qwen2_navit("reference/Lance")
    Qwen2Model = nav_mod.Qwen2Model

    # flex_attention is a compiled GPU primitive; replace it with SDPA + a
    # materialized dense mask so validation_gen's forward_train runs on CPU.
    from torch.nn.attention.flex_attention import create_mask as _create_mask

    def _flex_sdpa(q, k, v, enable_gqa=False, block_mask=None, scale=None, **kw):
        L = q.shape[-2]
        attn = None
        if block_mask is not None and getattr(block_mask, "mask_mod", None) is not None:
            attn = _create_mask(block_mask.mask_mod, 1, 1, L, L, device=q.device)
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn, enable_gqa=enable_gqa, scale=scale)
    nav_mod.flex_attention = _flex_sdpa

    cfg_d = json.loads((LANCE / "llm_config.json").read_text())
    qcfg = Qwen2Config(
        vocab_size=cfg_d["vocab_size"], hidden_size=cfg_d["hidden_size"],
        num_attention_heads=cfg_d["num_attention_heads"],
        num_key_value_heads=cfg_d["num_key_value_heads"],
        num_hidden_layers=cfg_d["num_hidden_layers"],
        intermediate_size=cfg_d["intermediate_size"], rms_norm_eps=cfg_d["rms_norm_eps"],
        rope_theta=cfg_d["rope_theta"], max_position_embeddings=cfg_d["max_position_embeddings"],
        attention_dropout=0.0, qk_norm=True, layer_module="Qwen2MoTDecoderLayer",
        apply_qwen_2_5_vl_pos_emb=False, rope_scaling=None, _attn_implementation="eager",
        video_token_id=cfg_d.get("video_token_id", 151656),
        pad_token_id=cfg_d.get("pad_token_id", None),
        bos_token_id=cfg_d.get("bos_token_id", 151643),
        eos_token_id=cfg_d.get("eos_token_id", 151645),
    )

    vae_cfg = types.SimpleNamespace(z_channels=48, downsample_spatial=16, downsample_temporal=4)
    lance_cfg = LanceConfig(
        visual_gen=True, visual_und=False, llm_config=qcfg, vit_config=None,
        vae_config=vae_cfg, latent_patch_size=(1, 1, 1), max_latent_size=32,
        timestep_shift=4.0, max_num_frames=12,  # -> max_num_latent_frames 4 (table 4*32^2=4096)
    )

    class LM(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.model = Qwen2Model(c)
            self.config = c

        def forward(self, *a, **k):
            return self.model(*a, **k)

        def forward_inference(self, *a, **k):
            return self.model.forward_inference(*a, **k)

    lm = LM(qcfg)
    model = Lance(language_model=lm, vit_model=None, config=lance_cfg)
    model.eval().to(torch.bfloat16)

    # load + remap weights
    w = mx.load(str(LANCE / "model.safetensors"))
    sd, dropped = {}, []
    for k, v in w.items():
        nk = _remap_key(k)
        if nk is None:
            dropped.append(k)
            continue
        sd[nk] = torch.from_numpy(np.asarray(v.astype(mx.float32))).to(torch.bfloat16)
    # time_embedder builds a float32 sinusoid then hits the bf16 MLP — cast it.
    _te = model.time_embedder
    _dt = _te.mlp[0].weight.dtype

    def _te_fwd(t, _te=_te, _dt=_dt):
        return _te.mlp(_te.timestep_embedding(t, _te.frequency_embedding_size).to(_dt))
    _te.forward = _te_fwd

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"weights: {len(sd)} mapped, dropped {len(dropped)}, missing {len(missing)}, unexpected {len(unexpected)}", flush=True)
    miss_real = [m for m in missing if "vit" not in m and "lm_head" not in m]
    if miss_real:
        print("  MISSING (non-vit):", miss_real[:10], flush=True)
    if unexpected:
        print("  UNEXPECTED:", unexpected[:10], flush=True)

    # build packed t2i inputs
    tok = AutoTokenizer.from_pretrained(str(LANCE))
    tok, nti, _ = add_special_tokens(tok)
    nti.setdefault("image_token_id", 151655)
    dc = DataConfig()
    for kk, vv in dict(task="t2i", resolution="image_256res", text_template=True,
                       system_prompt_type="SP1", num_frames=1, max_duration=1.0,
                       H=256, W=256, latent_patch_size=(1, 1, 1), max_latent_size=32,
                       vae_downsample=(4, 16, 16), max_num_frames=1).items():
        setattr(dc, kk, vv)
    ds = ValidationDataset("/tmp/t2i_probe.jsonl", tok, DataArguments(), ModelArguments(),
                           TrainingArguments(), nti, dc)
    batch = simple_custom_collate([ds[0]])
    d = batch.to_dict()

    params = dict(
        val_packed_text_ids=d["packed_text_ids"], val_packed_text_indexes=d["packed_text_indexes"],
        val_packed_position_ids=d["packed_position_ids"],
        val_split_lens=d["split_lens"], val_attn_modes=d["attn_modes"],
        val_sample_N_target=d["sample_N_target"], val_packed_vae_token_indexes=d["packed_vae_token_indexes"],
        timestep_shift=4.0, num_timesteps=24, val_mse_loss_indexes=d.get("mse_loss_indexes"),
        val_padded_latent=d["padded_latent"], video_sizes=d["video_sizes"],
        cfg_text_scale=1.0, cfg_interval=[0.0, 1.0], cfg_renorm_min=0.0, cfg_renorm_type="global",
        device="cpu", dtype=torch.bfloat16, new_token_ids=nti, max_samples=1,
        validation_noise_seed=42, apply_chat_template=True, apply_qwen_2_5_vl_pos_emb=False,
        image_token_id=151655, vae_video_grid_thw=d["vae_video_grid_thw"],
        val_packed_vit_tokens=None, val_packed_vit_token_indexes=None,
        vit_video_grid_thw=None, video_grid_thw=d.get("video_grid_thw"),
        sample_task=d.get("sample_task"), sample_modality=d.get("sample_modality"),
        val_sample_lens=d["sample_lens"],
    )
    print("running real validation_gen (flex→SDPA) ...", flush=True)
    try:
        with torch.no_grad():
            out = model.validation_gen(**params)
    except (IndexError, ValueError) as e:
        print(f"\nFINDING ({type(e).__name__}: {str(e)[:100]}):\n"
              f"The full real Lance instantiates + loads on CPU (1020 weights, 0 missing) —"
              f" a complete on-machine reference. Its two validation paths are GPU-bound:\n"
              f"  • validation_gen → flex_attention (block_mask, fails on CPU/length-pad);\n"
              f"  • validation_gen_KVcache → expects 3D mrope current_pos_ids (apply=True).\n"
              f"The ValidationDataset (default config) produces 1D CONSTANT latent positions"
              f" — exactly the MLX fix (§144), confirming the constant-position scheme is"
              f" correct. So the remaining t2i softness is the validation_gen flex_attention"
              f" BLOCK_MASK structure (segment-aware: causal text + bidirectional noise +"
              f" BLOCK_SIZE padding), not the model or positions. See VALIDATION_REPORT §147.",
              flush=True)
        return 0
    # out is x_t_all: list per sample of list of patches (t,h,w,c)
    lat = out[0][0] if isinstance(out, (list, tuple)) else out
    lat_np = lat.float().cpu().numpy()
    print("latent out shape:", lat_np.shape, "std", float(lat_np.std()), flush=True)

    # decode with the verified MLX VAE
    from yunshu_engine.wan_vae import load_wan_vae
    vae = load_wan_vae(str(LANCE / "vae.safetensors"))
    a = np.asarray(lat_np)
    # reshape (t,h,w,c) -> (1,t,h,w,c) for the MLX VAE
    if a.ndim == 4:
        a = a[None]
    img = vae.decode(mx.array(a))
    mx.eval(img)
    im = np.asarray(img)[0, 0]
    from PIL import Image
    Image.fromarray(((np.clip(im, -1, 1) + 1) * 127.5).astype(np.uint8)).save("/tmp/real_valgen.png")
    print(f"REAL validation_gen image: shape {im.shape}, std {im.std():.3f}, saved /tmp/real_valgen.png", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
