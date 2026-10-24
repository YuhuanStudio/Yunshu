#!/bin/bash
# gpuq short job: the retry scenario on a 2B model (adoption wiring proof). agentload_small.sh GRACE OUTDIR
set -euo pipefail
T=$(cd "$(dirname "$0")/../.." && pwd)
exec /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python "$T/scripts/research/agentload_repro.py" \
  /Volumes/P5Plus/models/Qwen3.5-2B-MLX-bf16 --scenario retry --grace "$1" --out "$2" \
  --port 18993 --home /Volumes/P5Plus/yunshu-build/agentload-home-small \
  --max-tokens 2500 --r2-at 3 --cut-at 6
