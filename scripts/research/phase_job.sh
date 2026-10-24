#!/bin/bash
# gpuq job: phase_probe on a 9B model. phase_job.sh SCENARIO OUTDIR
set -euo pipefail
T=$(cd "$(dirname "$0")/../.." && pwd)
exec /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python "$T/scripts/research/phase_probe.py" \
  /Volumes/P5Plus/models/Qwen3.5-9B-MLX-4bit --scenario "$1" --out "$2" --ctx-chars 30000 --tokens 300
