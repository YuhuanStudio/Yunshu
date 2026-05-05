"""Model download script — downloads all Yunshu test models.

Run: PYTHONPATH=. uv run python scripts/download_models.py
"""

import os
import sys
from pathlib import Path

MODELS = [
    ("mlx-community/Qwen3.5-9B-MLX-4bit", "LLM"),
    ("mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16", "TTS"),
    ("mlx-community/Qwen3-ASR-1.7B-bf16", "ASR"),
    ("mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit", "VLM"),
    ("andrevp/Z-Image-Turbo-MLX-4bit", "IMAGE_GEN"),
]

MODELS_DIR = Path(__file__).parent.parent / "models"


def download_model(model_id: str, target_dir: Path):
    """Download a model from HuggingFace."""
    local_name = model_id.rsplit("/", 1)[-1]
    dest = target_dir / local_name

    if dest.exists():
        print(f"  [SKIP] {local_name} already exists")
        return

    print(f"  [DOWNLOAD] {model_id} → {dest}")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=model_id,
        local_dir=str(dest),
    )
    print(f"  [DONE] {local_name}")


def main():
    print("=== Yunshu Model Downloader ===")
    print(f"Target: {MODELS_DIR}")
    print()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Filter by argument if provided
    filter_type = sys.argv[1] if len(sys.argv) > 1 else None

    for model_id, model_type in MODELS:
        if filter_type and filter_type.lower() != model_type.lower():
            continue
        print(f"[{model_type}] {model_id}")
        download_model(model_id, MODELS_DIR)
        print()

    print("Done! Run 'just dev-multi' to start the multi-model server.")


if __name__ == "__main__":
    main()
