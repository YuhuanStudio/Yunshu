#!/bin/bash
# gpuq job body: one arm (shard) of the fidelity run. fnfid_job.sh OUTDIR ARM SHARD(k/n)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$1; ARM=$2; SHARD=${3:-0/1}
export TFB_WORK=/Volumes/P5Plus/yunshu-build/tfnew TFB_YUNSHU_SRC="$ROOT/python" TFB_OUT="$OUT/tfb-$ARM-${SHARD/\//of}" TFB_PORT_LAST=18996
export PYTHONPATH="$ROOT/python" TMPDIR="$OUT/tmp"
mkdir -p "$TMPDIR"
cd "$ROOT/scripts/research"
exec "$ROOT/.venv/bin/python" fnfid.py run --arm "$ARM" --shard "$SHARD" --model /Volumes/Micron/models/Jundot/Qwen3.8-Flash-Next-oQ2.5e-mtp --out "$OUT/$ARM-${SHARD/\//of}.jsonl"
