#!/bin/bash
# Flash-Next MTP matrix: depth {2,3,4} x policy {fixed, strata-p0.5}, 1K/32K, 3 reps.
# Runs only after the real-pack smoke run has PASSED (fails closed otherwise).
# Usage: flashnextmtp_matrix.sh SMOKE_RUN_DIR CAND_SHA BASE_SHA PACK
set -u
SMOKE=$1; CAND=$2; BASE=$3; PACK=$4
export GPUQ_DIR=${GPUQ_DIR:-/Volumes/P5Plus/yunshu-gpuq} GPUQ_OWNER=sonnet-flashnextmtp
YV="$(dirname "$0")/../dev/yv"
"$YV" wait "$SMOKE" || { echo "smoke did not pass; matrix not submitted" >&2; exit 1; }
"$YV" status "$SMOKE" | grep -q "verdict PASS" || { echo "smoke verdict is not PASS" >&2; exit 1; }
for depth in 2 3 4; do
  for policy in fixed p0.5; do
    prob=0; [ "$policy" = p0.5 ] && prob=0.5
    "$YV" ab --base "$BASE" --cand "$CAND" --suite preflight,identity,apc,speed \
      --ctx 1024,32768 --reps 3 --spec-off --reuse-base-speed \
      --label "flashnextmtp-matrix-d${depth}-${policy}" --model "$PACK" --model-name flashnext \
      --base-env YUNSHU_VLM_DRAFT=off --cand-env YUNSHU_VLM_DRAFT=mtp \
      --cand-env "YUNSHU_MTP_BLOCK_SIZE=$((depth + 1))" \
      --cand-env "YUNSHU_QWEN4_DRAFT_MIN_PROB=$prob" \
      --engaged spec:mtp --priority -2 --mem-gb 90 --detach || exit 1
  done
done
