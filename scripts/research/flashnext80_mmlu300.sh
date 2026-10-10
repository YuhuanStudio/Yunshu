#!/bin/bash
# Queue the thinking-protocol MMLU-Pro-300 for Flash-Next oQ4e-MTP (same harness as the 27B gate number).
# Resumable: every round continues the same jsonl, finished ids are skipped.
# Usage: flashnext80_mmlu300.sh VIEW_DIR [ROUNDS]
set -eu
V=${1:?view dir}
ROUNDS=${2:-4}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
export GPUQ_DIR=${GPUQ_DIR:-/Volumes/P5Plus/yunshu-gpuq} GPUQ_OWNER=${GPUQ_OWNER:-sonnet-flashnext80}
export PAIRED_OUT=${PAIRED_OUT:-/Volumes/P5Plus/yunshu-build/flashnext80/mmlu300}
export PAIRED_TREE=$ROOT PAIRED_PORT_LAST=18999
mkdir -p "$PAIRED_OUT"
G=/Users/yuhuan/Documents/YuhuanStudio/Yunshu/scripts/dev/gpuq
# The paired-eval env is pinned to an older mlx-vlm without qwen4_exp (and lxml): serve with the main env.
PY=/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python
export PAIRED_PY=$PY
for r in $(seq ${FIRST:-0} $((ROUNDS - 1))); do
  "$G" submit --label "flashnext80-mmlu300-r$r" --timeout 20 --stall 12 --priority 0 --mem-gb 95 --quiet -- \
    "$PY" "$ROOT/scripts/research/accuracy/paired_eval.py" run --bench mmlu_pro --arm fn80 --model "$V" \
    --model-name Qwen3.8-Flash-Next --mmlu-n 300 --budget-min 15 --concurrency 4 \
    --env YUNSHU_VLM_DRAFT=mtp --env YUNSHU_VLM_APC_MEMORY_GB=1 --env YUNSHU_VLM_APC_DISK=0 \
    --env YUNSHU_MAX_MEMORY_GB=74.5
done
