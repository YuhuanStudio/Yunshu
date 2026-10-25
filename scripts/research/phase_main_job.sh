#!/bin/bash
# gpuq job: phase_probe against a detached checkout of main (the tree to compare). phase_main_job.sh TREE OUTDIR
set -euo pipefail
T=$(cd "$(dirname "$0")/../.." && pwd)
exec /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python "$T/scripts/research/phase_probe.py" \
  /Volumes/P5Plus/models/Qwen3.5-9B-MLX-4bit --scenario burst --out "$2" --ctx-chars 30000 --tokens 300 \
  --tree "$1" --port 18996
