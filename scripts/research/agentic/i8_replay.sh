#!/bin/sh
# Three fresh-server replays, captured bodies unchanged except seeded sampling.
set -eu
mode="$1"
root="$(git rev-parse --show-toplevel)"
main=/Users/yuhuan/Documents/YuhuanStudio/Yunshu
bodies="$main/docs/research/runs/2026-09-30-agtraffic/artifacts/cap-opencode-fix-cart-discount-r1/bodies"
out="$root/docs/research/runs/2026-10-02-i8"
mkdir -p "$out"
export AGENTIC_YUNSHU_SRC="$root/python"
export YUNSHU_AUXILIARY_SCHEDULING="$mode"
for run in 1 2 3; do
  "$main/.venv/bin/python" "$root/scripts/research/agentic/i8_session_replay.py" \
    --checkpoint /Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp \
    --bodies "$bodies" \
    --cache-dir "/Volumes/P5Plus/yunshu-build/codex-i8-apc/$mode-$run" \
    --label "i8-$mode-$run" --out "$out/$mode-$run.jsonl" --log "$out/$mode-$run.server.log"
done
