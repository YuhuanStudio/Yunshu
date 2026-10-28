#!/bin/bash
# gpuq job body: capture one shard of one arm (pilot). pilot.sh ARM CACHE_DIR SHARD
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
ARM=$1; CACHE=$2; SHARD=$3
export TFB_WORK=/Volumes/P5Plus/yunshu-build/tfnew TFB_YUNSHU_SRC="$ROOT/python" TFB_OUT="$CACHE/tfb-$ARM" TFB_PORT_LAST=18996
export PYTHONPATH="$ROOT/python:$ROOT/scripts" TMPDIR="$CACHE/tmp"
mkdir -p "$TMPDIR"
cd "$ROOT"
EXTRA=()
[ "$ARM" = cand ] && EXTRA=(--env YUNSHU_QSA_SPARSE_PREFILL=1)
exec "$ROOT/.venv/bin/python" -m verify.kld.capture run --arm "$ARM" --model /Volumes/Micron/models/Jundot/Qwen3.8-Flash-Next-oQ2.5e-mtp-ple-ssd --cache-dir "$CACHE/$ARM" --shard "$SHARD" ${EXTRA[@]+"${EXTRA[@]}"} --out "$CACHE/$ARM-pilot.jsonl"
