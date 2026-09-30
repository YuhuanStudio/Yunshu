#!/bin/bash
# Queue one real agent session capture per agent (bodies + per-request cache counters).
#   submit_capture.sh LABEL [extra env KEY=VAL ...]
# Output lands in docs/research/runs/<day>-apcaudit of the main checkout (private data).
set -e
LABEL=$1
shift
MAIN=/Users/yuhuan/Documents/YuhuanStudio/Yunshu
WT=$(cd "$(dirname "$0")/../../.." && pwd)
SRC=${SRC:-$WT/python}
OUT=$MAIN/docs/research/runs/2026-09-30-apcaudit
M=/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp
mkdir -p "$OUT"
for agent in opencode claude codex; do
  "$MAIN/scripts/dev/gpuq" submit --label "apcaudit-$LABEL-$agent" --timeout 25 --stall 10 --priority 0 -- \
    /usr/bin/env AGENTIC_YUNSHU_SRC="$SRC" "$@" \
    "$MAIN/.venv/bin/python" "$WT/scripts/research/agentic/run_agentic.py" run \
    --serve yunshu --checkpoint "$M" --engine-label "$LABEL" --agent "$agent" \
    --tasks fix-cart-discount,polyglot-wordy --repeat 1 --timeout-min 9 --save-bodies \
    --output "$OUT/$LABEL-$agent.jsonl"
done
