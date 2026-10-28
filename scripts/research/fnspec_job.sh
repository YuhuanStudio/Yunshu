#!/bin/bash
# One Flash-Next sampled-spec arm (gpuq job body): fnspec_job.sh ARM OUTDIR [probe args...]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ARM=$1; OUT=$2; shift 2
mkdir -p "$OUT"
export TFB_WORK=/Volumes/P5Plus/yunshu-build/snapshot014/corpus TFB_YUNSHU_SRC="$ROOT/python" TFB_OUT="$OUT/tfb" TFB_PORT_LAST=18996
export PYTHONPATH="$ROOT/python" TMPDIR="$OUT/tmp"
mkdir -p "$TMPDIR"
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/research/fnspec_probe.py" --arm "$ARM" \
  --model /Volumes/Micron/models/Jundot/Qwen3.8-Flash-Next-oQ2.5e-mtp-ple-ssd --out "$OUT/$ARM.json" "$@"
