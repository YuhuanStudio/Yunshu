#!/bin/bash
# Queue (via the argv-checking submit helper): PLE-rollback identity runs and the TensorFold-vs-Yunshu ceiling.
set -eu
cd "$(dirname "$0")/../.."
S="python3 scripts/research/flashnext80_submit.py"
O=/Volumes/P5Plus/yunshu-build/flashnext80
V=/Volumes/Micron/models/Jundot/Qwen3.8-Flash-Next-oQ3e-mtp
P=/Volumes/Micron/models/Jundot/Qwen3.8-Flash-Next-oQ3e-mtp
PYB=/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python
COMMON=(--tree "$PWD" --model "$V" --sizes 1024 --max-tokens 1000 --env YUNSHU_VLM_APC_MEMORY_GB=1
        --env YUNSHU_VLM_APC_DISK=0 --env YUNSHU_MAX_MEMORY_GB=74.5 --require-ple)
roll() {  # name, extra harness args...
  local n=$1; shift
  $S --label "flashnext80-rollback-$n" --priority 0 --timeout 15 --stall 10 --mem-gb 95 --expect-complete \
    --out "$O/rollback-$n.jsonl" -- $PYB scripts/research/flashnext80_footprint.py "${COMMON[@]}" "$@" --out "$O/rollback-$n.jsonl"
}
roll off --env YUNSHU_VLM_DRAFT=off
roll b2 --env YUNSHU_VLM_DRAFT=mtp --env YUNSHU_MTP_BLOCK_SIZE=2
roll b4 --env YUNSHU_VLM_DRAFT=mtp --env YUNSHU_MTP_BLOCK_SIZE=4
roll offstock --env YUNSHU_VLM_DRAFT=off
roll b4stock --env YUNSHU_VLM_DRAFT=mtp --env YUNSHU_MTP_BLOCK_SIZE=4
for arm in tf-nodraft tf-mtp; do
  $S --label "flashnext80-ceiling-$arm" --priority 0 --timeout 15 --stall 10 --mem-gb 115 --quiet --expect-complete \
    --out "$O/ceiling-$arm.jsonl" -- $PYB scripts/research/flashnext80_engines.py --arm $arm --model "$P" --ctx 1024 \
    --reps 3 --tokens 256 --tf-arg=--ple-on-ssd --env=TENSORFOLD_MEMORY_LIMIT_GB=107 --out "$O/ceiling-$arm.jsonl"
done
for arm in yunshu-off yunshu-mtp; do
  $S --label "flashnext80-ceiling-$arm" --priority 0 --timeout 15 --stall 10 --mem-gb 95 --quiet --expect-complete \
    --out "$O/ceiling-$arm.jsonl" -- $PYB scripts/research/flashnext80_engines.py --arm $arm --model "$V" --ctx 1024 \
    --reps 3 --tokens 256 --env=YUNSHU_VLM_APC_DISK=0 --env=YUNSHU_MAX_MEMORY_GB=74.5 \
    --out "$O/ceiling-$arm.jsonl"
done
