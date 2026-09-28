#!/bin/zsh
# 27B validation of fused chunked prefill (YUNSHU_FUSED_PREFILL_TOKENS) on an
# idle GPU. Phase 1 picks the budget: bench_mixed_load (4 streams decoding, a
# 16K / 32K prompt arrives) at 0 (separate prefill forwards) and each budget;
# phase 2 runs the regression set at the chosen budget ($WIN, default 128).
#
#   zsh scripts/research/validate_fused_prefill.sh            # both phases
#   PHASE=2 WIN=256 zsh scripts/research/validate_fused_prefill.sh
#
# Every server log must show "Fused prefill engaged" for budgets > 0.
set -u
cd "$(dirname "$0")/../.."
M=${M:-/Volumes/P5Plus/models/Jundot/Qwen3.8-27B-oQ4e-mtp}
PORT=${PORT:-18764}
OUT=${OUT:-docs/research/runs/$(date +%Y-%m-%d)-fused-prefill}
PHASE=${PHASE:-all}
WIN=${WIN:-128}
URL=http://127.0.0.1:$PORT
PY=.venv/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p $OUT
log(){ echo "$(date +%H:%M:%S) $*"; }
wait_ready(){ for i in $(seq 1 300); do curl -s -m 2 $URL/health/ready 2>/dev/null | grep -q '"ready":true' && return 0; kill -0 $1 2>/dev/null || return 1; sleep 2; done; return 1; }
stop(){ kill -INT $1 2>/dev/null; for i in $(seq 1 30); do kill -0 $1 2>/dev/null || return; sleep 1; done; kill $1 2>/dev/null; }
serve(){  # $1 = budget; sets YP
  YUNSHU_FUSED_PREFILL_TOKENS=$1 YUNSHU_MODEL=$M YUNSHU_AUTH_DISABLED=1 \
    $PY -m uvicorn yunshu_gateway.main:app --host 127.0.0.1 --port $PORT > $OUT/server-$1.log 2>&1 &
  YP=$!
  wait_ready $YP
}
engaged(){ [ $1 = 0 ] || grep -q "Fused prefill engaged" $OUT/server-$1.log && log "budget $1 engaged" || log "BUDGET $1 NOT ENGAGED"; }

if [ $PHASE = all -o $PHASE = 1 ]; then
  for b in 0 64 128 256 512; do
    if serve $b; then
      log "mixed load budget=$b"
      for pp in 16384 32768; do
        $PY scripts/research/bench_mixed_load.py --url $URL --model Qwen3.8-27B --tokenizer $M \
          --streams 4 --pp $pp --label fused$b --output $OUT/mixed-load.jsonl > /dev/null 2>&1 \
          || log "mixed load budget=$b pp=$pp FAILED"
      done
      engaged $b
    else log "server budget=$b failed"; fi
    stop $YP; sleep 5
  done
fi

if [ $PHASE = all -o $PHASE = 2 ]; then
  for b in 0 $WIN; do
    if serve $b; then
      log "regression budget=$b"
      $PY scripts/research/probe_concurrency.py --url $URL --model Qwen3.8-27B --n 8 \
        --note "fused prefill $b" --output $OUT/concurrency.jsonl > /dev/null 2>&1 || log "concurrency $b FAILED"
      $PY scripts/research/bench_context_batch.py --url $URL --model Qwen3.8-27B --tokenizer $M --pid $YP \
        --lengths 1024 32768 --batches 2 4 8 --note "fused prefill $b" \
        --output $OUT/context-batch.jsonl > /dev/null 2>&1 || log "context batch $b FAILED"
      $PY scripts/research/bench_engine_matrix.py --url $URL --model Qwen3.8-27B --engine yunshu-fused$b \
        --checkpoint $M --pid $YP --note "fused prefill $b" --output $OUT/matrix.jsonl > /dev/null 2>&1 \
        || log "matrix $b FAILED"
      if [ $b != 0 ]; then
        log "mmlu budget=$b"
        $PY scripts/research/soak_mmlu_pro.py --url $URL --model Qwen3.8-27B --pid $YP \
          --note "Yunshu fused prefill $b" --output $OUT/mmlu-soak.jsonl > $OUT/mmlu-soak.log 2>&1 \
          || log "mmlu $b FAILED"
      fi
      engaged $b
    else log "server budget=$b failed"; fi
    stop $YP; sleep 5
  done
fi
log "fused prefill validation done"
