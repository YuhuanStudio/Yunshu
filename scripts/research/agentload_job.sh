#!/bin/bash
# gpuq job body: scripts/research/agentload_job.sh SCENARIO GRACE OUTDIR [extra repro args]
set -euo pipefail
T=$(cd "$(dirname "$0")/../.." && pwd)
SC=$1; GR=$2; OUT=$3; shift 3
exec /Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python "$T/scripts/research/agentload_repro.py" \
  /Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp --scenario "$SC" --grace "$GR" --out "$OUT" \
  --port 18992 --home /Volumes/P5Plus/yunshu-build/agentload-home \
  --draft /Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2 --shots-dir "$OUT/shots-$SC-g$GR" "$@"
