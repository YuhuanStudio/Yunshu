#!/bin/bash
# Queue the identity + speed sweep of the fused path: spec off and MTP block 2/3/4/5 at 1K and 8K, 1000 tokens.
# Usage: flashnext80_queue_sweep.sh TAG [EXTRA harness args...]   (labels flashnext80-sweepTAG-*)
set -eu
cd "$(dirname "$0")/../.."
TAG=$1; shift
S=scripts/research/fn80-gpuq
O=/Volumes/P5Plus/yunshu-build/flashnext80
V=/Volumes/P5Plus/models/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp-ple-ssd
PYB=/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python
COMMON=(--tree "$PWD" --model "$V" --sizes 1024 8192 --max-tokens 1000
        --env YUNSHU_VLM_APC_MEMORY_GB=1 --env YUNSHU_VLM_APC_DISK=0 --env YUNSHU_MAX_MEMORY_GB=74.5 --require-ple)
run() {  # name, extra args
  local n=$1; shift
  $S --label "flashnext80-sweep$TAG-$n" --priority 0 --timeout 18 --stall 10 --mem-gb 95 --expect-complete \
    --out "$O/sweep$TAG-$n.jsonl" -- $PYB scripts/research/flashnext80_footprint.py "${COMMON[@]}" "$@" --out "$O/sweep$TAG-$n.jsonl"
}
run off --env YUNSHU_VLM_DRAFT=off "$@"
for b in 2 3 4 5; do run b$b --env YUNSHU_VLM_DRAFT=mtp --env YUNSHU_MTP_BLOCK_SIZE=$b --require-spec "$@"; done
