#!/bin/bash
# Queue the APC cache-hierarchy GPU jobs (one server per job, Qwen3.8-27B).   queue_tier4.sh JOB
#   JOB: smoke | smoke-base | small-base | small-nodisk | small-warm | small-warm-int8 | large-base
#        | small-tiers | small-tiers-warm | hdd-only | mid-base | mid-warm | mid-warm-int8
# 3 agent sessions grow to 30K tokens round-robin with idle gaps, then each is revisited; the APC
# RAM budget is small (4 GiB: about two 30K checkpoints) or large (32 GiB). Results (private) land in
# docs/research/runs/<day>-tier4 of the main checkout; the replay is session_replay.py.
set -e
MAIN=/Users/yuhuan/Documents/YuhuanStudio/Yunshu
WT=$(cd "$(dirname "$0")/../../.." && pwd)
SRC=${SRC:-$WT/python}
OUT=$MAIN/docs/research/runs/${TIER4_DAY:-2026-10-02}-tier4
M=/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp
TEMPLATE=$MAIN/docs/research/runs/2026-09-30-agtraffic/artifacts/cap-opencode-fix-cart-discount-r1/bodies/0002-req.json
SCRATCH=/Volumes/P5Plus/yunshu-scratch/tier4/run${SUFFIX:-}   # a fresh tree per run label (no pre-warmed caches)
INTERNAL=/Users/yuhuan/.yunshu/cache/tier4-apc${SUFFIX:-}   # the real internal SSD; removed after the run
PY=$MAIN/.venv/bin/python
SCRIPT=$WT/scripts/research/apc_audit/session_replay.py
mkdir -p "$OUT"
JOB=$1
SMALL=4
MULTI="--scenario multi --sessions 3 --target 30000 --step 3000 --gap-s 20 --deadline-min 48"

submit() { # label timeout -- args...
  local label=$1${SUFFIX:-} timeout=$2
  shift 3
  rm -f "$OUT/$label.jsonl" "$OUT/$label.server.log"
  "$MAIN/scripts/dev/gpuq" submit --label "tier4-$label" --timeout "$timeout" --stall 10 --priority 0 -- \
    /usr/bin/env AGENTIC_YUNSHU_SRC="$SRC" "$PY" "$SCRIPT" --checkpoint "$M" --template "$TEMPLATE" \
    --label "$label" --out "$OUT/$label.jsonl" --log "$OUT/$label.server.log" --env YUNSHU_DEBUG_ROUTES=1 --env YUNSHU_AUTH_DISABLED=1 "$@"
}

case $JOB in
  smoke) submit smoke 15 -- --scenario multi --sessions 2 --target 12000 --step 2000 --gap-s 5 --deadline-min 12 \
           --env YUNSHU_VLM_APC_MEMORY_GB=1.5 --env YUNSHU_VLM_APC_DISK_DIR=$SCRATCH/smoke-t0 --env YUNSHU_VLM_APC_DISK_GB=1.2 \
           --env "YUNSHU_VLM_APC_DISK_TIERS=$SCRATCH/smoke-t1@4,$SCRATCH/smoke-t2@8!sim=150/12" ;;
  smoke-base) submit smoke-base 15 -- --scenario multi --sessions 2 --target 12000 --step 2000 --gap-s 5 --deadline-min 12 \
           --env YUNSHU_VLM_APC_MEMORY_GB=1.5 --env YUNSHU_VLM_APC_DISK_DIR=$SCRATCH/smoke-b0 ;;
  small-base)     submit small-base 55 -- $MULTI --env YUNSHU_VLM_APC_MEMORY_GB=$SMALL --env YUNSHU_VLM_APC_DISK_DIR=$SCRATCH/small-base ;;
  small-nodisk)   submit small-nodisk 55 -- $MULTI --env YUNSHU_VLM_APC_MEMORY_GB=$SMALL --env YUNSHU_VLM_APC_DISK=0 ;;
  small-warm)     submit small-warm 55 -- $MULTI --env YUNSHU_VLM_APC_MEMORY_GB=$SMALL --env YUNSHU_VLM_APC_DISK_DIR=$SCRATCH/small-warm \
                    --env YUNSHU_VLM_APC_WARM=lossless ;;
  small-warm-int8) submit small-warm-int8 55 -- $MULTI --env YUNSHU_VLM_APC_MEMORY_GB=$SMALL --env YUNSHU_VLM_APC_DISK_DIR=$SCRATCH/small-warm8 \
                    --env YUNSHU_VLM_APC_WARM=int8 ;;
  large-base)     submit large-base 55 -- $MULTI --env YUNSHU_VLM_APC_MEMORY_GB=32 --env YUNSHU_VLM_APC_DISK_DIR=$SCRATCH/large-base ;;
  small-tiers)    submit small-tiers 55 -- $MULTI --env YUNSHU_VLM_APC_MEMORY_GB=$SMALL --env YUNSHU_VLM_APC_DISK_DIR=$INTERNAL \
                    --env YUNSHU_VLM_APC_DISK_GB=3 \
                    --env "YUNSHU_VLM_APC_DISK_TIERS=$SCRATCH/tb4-tier@6,$SCRATCH/hdd-tier@40!sim=150/12" ;;
  small-tiers-warm) submit small-tiers-warm 55 -- $MULTI --env YUNSHU_VLM_APC_MEMORY_GB=$SMALL --env YUNSHU_VLM_APC_DISK_DIR=$INTERNAL \
                    --env YUNSHU_VLM_APC_DISK_GB=3 --env YUNSHU_VLM_APC_WARM=lossless \
                    --env "YUNSHU_VLM_APC_DISK_TIERS=$SCRATCH/tb4-tier@6,$SCRATCH/hdd-tier@40!sim=150/12" ;;
  hdd-only)       submit hdd-only 55 -- $MULTI --env YUNSHU_VLM_APC_MEMORY_GB=$SMALL --env YUNSHU_VLM_APC_DISK_DIR=$SCRATCH/hdd-t0 \
                    --env YUNSHU_VLM_APC_DISK_GB=3 --env "YUNSHU_VLM_APC_DISK_TIERS=$SCRATCH/hdd-only-tier@40!sim=150/12" ;;
  # 6 GiB: the three sessions' newest checkpoints (6.6 GB at 30K) are about HOT + WARM, where WARM can matter
  mid-base)       submit mid-base 55 -- $MULTI --env YUNSHU_VLM_APC_MEMORY_GB=6 --env YUNSHU_VLM_APC_DISK_DIR=$SCRATCH/mid-base ;;
  mid-warm)       submit mid-warm 55 -- $MULTI --env YUNSHU_VLM_APC_MEMORY_GB=6 --env YUNSHU_VLM_APC_DISK_DIR=$SCRATCH/mid-warm \
                    --env YUNSHU_VLM_APC_WARM=lossless ;;
  mid-warm-int8)  submit mid-warm-int8 55 -- $MULTI --env YUNSHU_VLM_APC_MEMORY_GB=6 --env YUNSHU_VLM_APC_DISK_DIR=$SCRATCH/mid-warm8 \
                    --env YUNSHU_VLM_APC_WARM=int8 ;;
  *) echo "unknown job $JOB"; exit 1 ;;
esac
