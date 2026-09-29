#!/bin/zsh
# 27B server decode speed by context on an idle GPU.
#
#   zsh scripts/research/validate_server_context.sh
#
# One server (port 18990), then bench_context_batch at 1K / 8K / 32K / 131K on
# the code_python and novel_en corpora (single requests, no batches). To compare
# two builds, run it from each checkout with a different OUT. The server log
# must show "mtp_lane": True in the verify kernels line.
#
# M (the Qwen3.8-27B checkpoint dir) comes from the environment or
# scripts/research/local.env (gitignored).
set -u
cd "$(dirname "$0")/../.."
[ -f scripts/research/local.env ] && source scripts/research/local.env
M=${M:?set M to the Qwen3.8-27B checkpoint directory}
PORT=${PORT:-18990}
OUT=${OUT:-docs/research/runs/$(date +%Y-%m-%d)-server-context}
LENGTHS=${LENGTHS:-"1024 8192 32768 131072"}
CORPORA=${CORPORA:-"code_python novel_en"}
URL=http://127.0.0.1:$PORT
PY=${PY:-.venv/bin/python}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=python
mkdir -p $OUT
log(){ echo "$(date +%H:%M:%S) $*"; }
wait_ready(){ for i in $(seq 1 300); do curl -s -m 2 $URL/health/ready 2>/dev/null | grep -q '"ready":true' && return 0; kill -0 $1 2>/dev/null || return 1; sleep 2; done; return 1; }
stop(){ kill -INT $1 2>/dev/null; for i in $(seq 1 30); do kill -0 $1 2>/dev/null || return; sleep 1; done; kill $1 2>/dev/null; }

YUNSHU_MODEL=$M YUNSHU_AUTH_DISABLED=1 \
  $PY -m uvicorn yunshu_gateway.main:app --host 127.0.0.1 --port $PORT > $OUT/server.log 2>&1 &
YP=$!
if wait_ready $YP; then
  log "server up"
  for corpus in ${=CORPORA}; do
    $PY scripts/research/bench_context_batch.py --url $URL --model Qwen3.8-27B --tokenizer $M --pid $YP \
      --corpus $corpus --lengths ${=LENGTHS} --batches --note "server context" \
      --output $OUT/context-$corpus.jsonl > $OUT/context-$corpus.log 2>&1 || log "context $corpus FAILED"
  done
else
  log "server failed"
fi
stop $YP
log "server context validation done"
