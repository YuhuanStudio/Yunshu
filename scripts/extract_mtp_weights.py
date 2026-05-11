"""Extract MTP weights from HuggingFace Qwen3.5 safetensors.

Downloads the original (non-MLX) safetensors, finds MTP keys, extracts
them, shifts norm weights (+1 for RMSNorm), and saves as MLX-compatible
mtp-weights.safetensors in the target model directory.

Usage:
    .venv/bin/python3 scripts/extract_mtp_weights.py --all
    .venv/bin/python3 scripts/extract_mtp_weights.py --model Qwen3.5-0.8B-MLX-bf16
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

# Map MLX model dir name -> HF source repo
MLX_TO_HF = {
    "Qwen3.5-0.8B-MLX-bf16": "Qwen/Qwen3.5-0.8B",
    "Qwen3.5-2B-MLX-bf16": "Qwen/Qwen3.5-2B",
    "Qwen3.5-4B-MLX-bf16": "Qwen/Qwen3.5-4B",
    "Qwen3.5-9B-MLX-bf16": "Qwen/Qwen3.5-9B",
    "Qwen3.5-9B-MLX-4bit": "Qwen/Qwen3.5-9B",
}

# RMSNorm weight suffixes that need +1 shift
NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "mtp.norm.weight",
    ".pre_fc_norm_hidden.weight",
    ".pre_fc_norm_embedding.weight",
)


def _scan_safetensors_keys(path: Path) -> list[str]:
    """Read safetensors header and return tensor key names."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return [k for k in header if k != "__metadata__"]


def _find_mtp_keys(safetensors_path: Path) -> dict[str, dict]:
    """Return {key: {offset, length, dtype, shape}} for MTP keys."""
    with open(safetensors_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))

    mtp = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        # Original HF keys: language_model.model.mtp.*
        if "mtp." in k and ("language_model.model.mtp." in k or k.startswith("mtp.")):
            mtp[k] = v
    return mtp


def extract_mtp_weights(model_name: str, force: bool = False) -> bool:
    """Extract MTP weights for a single model."""
    model_dir = ROOT / "models" / model_name
    if not model_dir.exists():
        print(f"  SKIP: {model_dir} not found")
        return False

    out_path = model_dir / "mtp-weights.safetensors"
    if out_path.exists() and not force:
        print(f"  EXISTS: {out_path} (use --force to overwrite)")
        return True

    hf_repo = MLX_TO_HF.get(model_name)
    if not hf_repo:
        print(f"  SKIP: No HF repo mapping for {model_name}")
        return False

    # Check config has MTP
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    text_config = config.get("text_config", config)
    n_mtp = int(text_config.get("mtp_num_hidden_layers", 0) or 0)
    if n_mtp == 0:
        print(f"  SKIP: {model_name} has mtp_num_hidden_layers=0")
        return False

    print(f"  Downloading from {hf_repo}...")

    try:
        from huggingface_hub import hf_hub_download
        import numpy as np
        import mlx.core as mx
    except ImportError as e:
        print(f"  ERROR: {e}")
        return False

    # List safetensors files in the repo
    from huggingface_hub import list_repo_files
    repo_files = list_repo_files(hf_repo)
    sf_files = [f for f in repo_files if f.endswith(".safetensors")]

    all_mtp_weights = {}

    for sf_name in sf_files:
        print(f"    Scanning {sf_name}...")
        local_path = hf_hub_download(hf_repo, sf_name)
        mtp_keys = _find_mtp_keys(Path(local_path))

        if not mtp_keys:
            continue

        print(f"    Found {len(mtp_keys)} MTP keys in {sf_name}")
        # Load with mx.load and extract MTP keys
        data = mx.load(str(local_path))
        for key in mtp_keys:
            # Strip the HF prefix: language_model.model.mtp.xxx -> mtp.xxx
            short_key = key
            if key.startswith("language_model.model.mtp."):
                short_key = key[len("language_model.model."):]
            all_mtp_weights[short_key] = data[key]
        del data

    if not all_mtp_weights:
        print(f"  WARNING: No MTP keys found in {hf_repo}")
        return False

    # Apply RMSNorm +1 shift
    norm_keys_shifted = []
    for k, v in all_mtp_weights.items():
        if any(k.endswith(s) for s in NORM_SUFFIXES):
            if v.ndim == 1:
                all_mtp_weights[k] = v + mx.array(1.0)
                norm_keys_shifted.append(k)

    print(f"  Extracted {len(all_mtp_weights)} MTP weight tensors")
    if norm_keys_shifted:
        print(f"  Shifted {len(norm_keys_shifted)} norm weights (+1)")

    # Save as MLX safetensors
    mx.save_safetensors(str(out_path), all_mtp_weights)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")

    # Verify: reload and check
    verify = mx.load(str(out_path))
    assert len(verify) == len(all_mtp_weights), "Key count mismatch!"
    print(f"  Verified: {len(verify)} tensors loaded correctly")

    return True


def main():
    parser = argparse.ArgumentParser(description="Extract MTP weights from HuggingFace")
    parser.add_argument("--all", action="store_true", help="Extract for all Qwen3.5 models")
    parser.add_argument("--model", help="Specific model dir name (e.g. Qwen3.5-0.8B-MLX-bf16)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing mtp-weights.safetensors")
    args = parser.parse_args()

    if args.all:
        models = sorted(MLX_TO_HF.keys())
    elif args.model:
        models = [args.model]
    else:
        parser.error("Specify --all or --model")

    print("=== MTP Weight Extraction ===")
    results = {}
    for model_name in models:
        print(f"\n[{model_name}]")
        ok = extract_mtp_weights(model_name, force=args.force)
        results[model_name] = "OK" if ok else "SKIP/FAIL"

    print("\n=== Summary ===")
    for name, status in results.items():
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
