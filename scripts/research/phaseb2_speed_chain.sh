#!/bin/bash
# Sequential phaseb2 speed suites (at most 3 jobs queued at a time). Usage: chain.sh <first-run-dir-to-wait-for>
export GPUQ_DIR=/Volumes/P5Plus/yunshu-gpuq TMPDIR=/Volumes/P5Plus/yunshu-build/tmp-phaseb2/
cd /Volumes/P5Plus/yunshu-build/wt-phaseb2 || exit 1
[ -n "$1" ] && scripts/dev/yv wait "$1" >/dev/null 2>&1
for s in 16k-burst4 85k-burst2 85k-stagger 85k-burst4; do
  scripts/dev/yv ab --base /Volumes/P5Plus/yunshu-build/wt-phaseb2-base --cand . \
    --model /Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp \
    --env YUNSHU_VLM_DRAFT=/Volumes/P5Plus/models/incoai/Qwen3.8-27B-DFlash2 \
    --suite phaseb2-speed-$s --label phaseb2-sp-$s --priority 1 >/dev/null 2>&1
  echo "done $s"
done
