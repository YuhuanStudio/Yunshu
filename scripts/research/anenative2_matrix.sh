#!/bin/bash
# Submit the cold-TTFT matrix: reps interleaved (rep, ctx, arm), --quiet, one gpuq job each.
# arms: off | cal (calibrated, no hand-set values) | fix (share 0.24, min rows 512)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
O=${O:-/Volumes/P5Plus/yunshu-build/anenative2}
G=/Users/yuhuan/Documents/YuhuanStudio/Yunshu/scripts/dev/gpuq
export GPUQ_DIR=/Volumes/P5Plus/yunshu-gpuq
REPS=${REPS:-3}
CTXS=${CTXS:-"1024 8192 32768"}
ARMS=${ARMS:-"off cal fix"}
ids=()
for rep in $(seq 0 $((REPS - 1))); do
  for ctx in $CTXS; do
    for arm in $ARMS; do
      extra=()
      case $arm in
        off) a=off ;;
        cal) a=splash ;;
        fix) a=splash; extra=(--share 0.24 --min-rows 512) ;;
      esac
      id=$($G submit --quiet --label "ane2-$arm-$ctx-r$rep" --timeout 20 -- bash "$ROOT/scripts/research/anenative_gate1.sh" \
        --arm $a --ctx $ctx --rep $rep --cache "$O/cache-gate" --out "$O/gate/$arm-$ctx-r$rep.jsonl" \
        --scratch "$O/scratch/$arm-$ctx-r$rep" ${extra[@]+"${extra[@]}"})
      ids+=("$id")
    done
  done
done
echo "${ids[@]}" | tee "$O/gate-ids.txt"
