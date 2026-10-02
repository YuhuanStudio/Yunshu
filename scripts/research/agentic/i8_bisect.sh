#!/bin/sh
# Single-feature arms: a = auxiliary scheduling only, u = uncached-work ordering only. usage: i8_bisect.sh RUN
set -u
root="$(git rev-parse --show-toplevel)"
main=/Users/yuhuan/Documents/YuhuanStudio/Yunshu
bodies="$main/docs/research/runs/2026-09-30-agtraffic/artifacts/cap-opencode-fix-cart-discount-r1/bodies"
out="$root/docs/research/runs/2026-10-03-i8c"
mkdir -p "$out"
export AGENTIC_YUNSHU_SRC="$root/python"
run=$1
rc=0
for arm in a u; do
  if [ $arm = a ]; then aux=1; unc=0; else aux=0; unc=1; fi
  YUNSHU_AUXILIARY_SCHEDULING=$aux YUNSHU_UNCACHED_SCHEDULING=$unc \
  "$main/.venv/bin/python" "$root/scripts/research/agentic/i8_session_replay.py" \
    --checkpoint /Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp \
    --bodies "$bodies" \
    --cache-dir "/Volumes/P5Plus/yunshu-build/codex-i8-apc/c-$arm-$run" \
    --label "i8c-$arm-$run" --out "$out/$arm-$run.jsonl" --log "$out/$arm-$run.server.log"
  r=$?; echo "arm=$arm run=$run rc=$r"; [ $r -ne 0 ] && rc=1
done
exit $rc
