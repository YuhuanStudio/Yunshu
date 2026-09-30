#!/bin/bash
# Queue APC audit GPU jobs (one server per job).   queue_jobs.sh JOB [SRC]
#   JOB: smoke | multi-base | multi-new | multi-ssd | tiers-ram | tiers-ssd | tiers-cold | single60
#   SRC: a python/ dir to serve (default: this worktree); baseline = the pre-policy copy.
# Results (private) land in docs/research/runs/<day>-apcaudit of the main checkout.
set -e
MAIN=/Users/yuhuan/Documents/YuhuanStudio/Yunshu
WT=$(cd "$(dirname "$0")/../../.." && pwd)
JOB=$1
SRC=${2:-$WT/python}
BASELINE=/Volumes/P5Plus/yunshu-build/apcaudit-baseline/python
OUT=$MAIN/docs/research/runs/2026-09-30-apcaudit
M=/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp
TEXT_M=/Volumes/P5Plus/models/Qwen2.5-3B-Instruct-4bit
TEMPLATE=$MAIN/docs/research/runs/2026-09-30-agtraffic/artifacts/cap-opencode-fix-cart-discount-r1/bodies/0002-req.json
SSD_DIR=/Volumes/P5Plus/tmp/apcaudit-ssd
PY=$MAIN/.venv/bin/python
SCRIPT=$WT/scripts/research/apc_audit/session_replay.py
mkdir -p "$OUT"

submit() { # label timeout -- args...
  local label=$1 timeout=$2
  shift 3
  "$MAIN/scripts/dev/gpuq" submit --label "apcaudit-$label" --timeout "$timeout" --stall 8 --priority 0 -- \
    /usr/bin/env AGENTIC_YUNSHU_SRC="$SRC" "$PY" "$SCRIPT" --checkpoint "$M" --template "$TEMPLATE" \
    --label "$label" --out "$OUT/$label.jsonl" --log "$OUT/$label.server.log" "$@"
}

case $JOB in
  smoke)      submit smoke 12 -- --scenario multi --sessions 2 --target 12000 --step 2000 ;;
  multi-base) SRC=$BASELINE submit multi-base 25 -- --scenario multi --sessions 3 --target 30000 --step 3000 --gap-s 20 --deadline-min 21 ;;
  multi-new)  submit multi-new 25 -- --scenario multi --sessions 3 --target 30000 --step 3000 --gap-s 20 --deadline-min 21 ;;
  multi-ssd)  submit multi-ssd 25 -- --scenario multi --sessions 3 --target 30000 --step 3000 --gap-s 20 --deadline-min 21 \
                --env YUNSHU_VLM_APC_DISK_DIR=$SSD_DIR/multi --env YUNSHU_VLM_APC_MEMORY_GB=8 ;;
  single60)   submit single60 25 -- --scenario multi --sessions 1 --target 62000 --step 4000 --deadline-min 21 ;;
  tiers-cold) submit tiers-cold 25 -- --scenario tiers --lengths 10000,30000,60000 --env YUNSHU_VLM_APC_MEMORY_GB=0 ;;
  tiers-ram)  submit tiers-ram 25 -- --scenario tiers --lengths 10000,30000,60000 --env YUNSHU_VLM_APC_MEMORY_GB=32 ;;
  tiers-ssd)  submit tiers-ssd 25 -- --scenario tiers --lengths 10000,30000,60000 \
                --env YUNSHU_VLM_APC_DISK_DIR=$SSD_DIR/tiers --env YUNSHU_VLM_APC_MEMORY_GB=0.3 ;;
  # text-only mlx-lm path (the four-tier HOT / WARM / SSD / COLD cache), Qwen2.5-3B
  text-default) M=$TEXT_M submit text-default 20 -- --scenario multi --sessions 3 --target 20000 --step 2500 --gap-s 5 ;;
  text-ssd)     M=$TEXT_M submit text-ssd 20 -- --scenario multi --sessions 3 --target 20000 --step 2500 --gap-s 5 \
                  --env YUNSHU_SSD_CACHE=1 --env YUNSHU_SSD_CACHE_DIR=$SSD_DIR/text --env YUNSHU_PREFIX_MAX_ENTRIES=2 ;;
  text-warm)    M=$TEXT_M submit text-warm 20 -- --scenario multi --sessions 3 --target 20000 --step 2500 --gap-s 5 \
                  --env YUNSHU_PREFIX_HOT_LIMIT=2 ;;
  *) echo "unknown job $JOB"; exit 1 ;;
esac
