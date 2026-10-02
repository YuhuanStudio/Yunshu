#!/bin/sh
# One round of the A/B (arm 0 = stock ordering, arm 1 = auxiliary + uncached-work scheduling);
# usage: i8_ab.sh RUN [first-arm]. Each arm continues after a failure of the other.
set -u
root="$(git rev-parse --show-toplevel)"
main=/Users/yuhuan/Documents/YuhuanStudio/Yunshu
bodies="$main/docs/research/runs/2026-09-30-agtraffic/artifacts/cap-opencode-fix-cart-discount-r1/bodies"
out="$root/docs/research/runs/2026-10-03-i8c"
mkdir -p "$out"
export AGENTIC_YUNSHU_SRC="$root/python"
rc=0
run=$1
first=${2:-0}
for run in $run; do
  for arm in $first $((1 - first)); do
    YUNSHU_AUXILIARY_SCHEDULING=$arm YUNSHU_UNCACHED_SCHEDULING=$arm \
    "$main/.venv/bin/python" "$root/scripts/research/agentic/i8_session_replay.py" \
      --checkpoint /Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp \
      --bodies "$bodies" \
      --cache-dir "/Volumes/P5Plus/yunshu-build/codex-i8-apc/c-$arm-$run" \
      --label "i8c-$arm-$run" --out "$out/$arm-$run.jsonl" --log "$out/$arm-$run.server.log"
    r=$?; echo "arm=$arm run=$run rc=$r"; [ $r -ne 0 ] && rc=1
  done
done
exit $rc
