"""Fetch the pinned research tokenizer/config snapshot, never model tensors.

Usage: uv run python scripts/research/fetch_qwen38_metadata.py /tmp/qwen38-metadata
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

REPO = "mlx-community/Qwen3.8-27B-4bit"
REVISION = "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"
FILES = (
    "chat_template.jinja",
    "tokenizer_config.json",
    "tokenizer.json",
    "config.json",
    "model.safetensors.index.json",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in FILES:
        url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        partial = args.directory / (name + ".part")
        partial.write_bytes(data)
        partial.replace(args.directory / name)
        hashes[name] = hashlib.sha256(data).hexdigest()
    print(json.dumps(dict(repo=REPO, revision=REVISION, sha256=hashes), indent=2))


if __name__ == "__main__":
    main()
