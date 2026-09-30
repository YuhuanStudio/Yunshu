#!/bin/zsh
# 27B validation of the round driver (YUNSHU_ROUND_DRIVER), in jobs of at most
# ~20 minutes each so every one can run under a GPU lock.
#
#   zsh scripts/research/validate_round_driver.sh                 # list the phases
#   PHASE=parity-1k zsh scripts/research/validate_round_driver.sh # one phase
#   PHASE=all zsh scripts/research/validate_round_driver.sh       # every phase in order
#
# GPU_RUN: a command that serializes GPU users, prefixed to every phase when
# PHASE=all (e.g. GPU_RUN="gpu_run.sh rd" runs `gpu_run.sh rd-<phase> zsh <this> `).
#
# In-process (scripts/research/sweep_round_driver.py; parity and rows apart):
#   parity-1k      alone / batch / staggered, MTP on and off token for token
#   rows-1k        rows 1 2 4 8, with and without drafts
#   parity-32k-a/b the same at 32K context (one prompt each)
#   rows-32k-a     rows 1 4 at 32K
#   rows-32k-b     rows 8 at 32K
#   decode-anatomy per-step time split at 1 / 8 rows (probe_driver_decode.py)
# Server (port 18990..; YUNSHU_ROUND_DRIVER=0 / 1):
#   server-rd{0,1}-probe    probe_concurrency n=8 + bench_engine_matrix (34/34)
#   server-rd{0,1}-ctx1k / ctx32k   bench_context_batch: b1, then b2 b4 b8 at that length
#   server-rd{0,1}-ctx131k-{1,2,4}  bench_context_batch at 131K (b8 x 131K exceeds a 20 minute job)
#   server-rd{0,1}-mixed16 / mixed32  bench_mixed_load (4 streams + a 16K / 32K prompt)
#   server-rd{0,1}-repeat   probe_repeat_doc: the same 8K / 32K prompt cold, repeated, and with another question
# Accuracy:
#   mmlu-<k> / mmlu0-<k>  MMLU-Pro slice k (0..2 = questions 100k .. 100k+99), b8, driver on / off
#
# M (the Qwen3.8-27B checkpoint dir) comes from the environment or
# scripts/research/local.env (gitignored). Results go to $OUT.
set -u
cd "$(dirname "$0")/../.."
[ -f scripts/research/local.env ] && source scripts/research/local.env
M=${M:?set M to the Qwen3.8-27B checkpoint directory}
PY=${PY:-.venv/bin/python}
BASE_PORT=${BASE_PORT:-18990}
OUT=${OUT:-docs/research/runs/$(date +%Y-%m-%d)-round-driver-packed}
PHASE=${PHASE:-list}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p $OUT
log(){ echo "$(date +%H:%M:%S) $*"; }

PHASES=(parity-1k parity-32k-a parity-32k-b rows-1k rows-32k-a rows-32k-b decode-anatomy
  server-rd0-probe server-rd1-probe
  server-rd0-ctx1k server-rd1-ctx1k server-rd0-ctx32k server-rd1-ctx32k
  server-rd0-ctx131k-1 server-rd1-ctx131k-1 server-rd0-ctx131k-2 server-rd1-ctx131k-2
  server-rd0-ctx131k-4 server-rd1-ctx131k-4
  server-rd0-mixed16 server-rd1-mixed16 server-rd0-mixed32 server-rd1-mixed32
  server-rd0-repeat server-rd1-repeat mmlu-0 mmlu-1 mmlu-2 mmlu0-0 mmlu0-1 mmlu0-2)

wait_ready(){ for i in $(seq 1 300); do curl -s -m 2 $URL/health/ready 2>/dev/null | grep -q '"ready":true' && return 0; kill -0 $1 2>/dev/null || return 1; sleep 2; done; return 1; }
stop(){ kill -INT $1 2>/dev/null; for i in $(seq 1 30); do kill -0 $1 2>/dev/null || return; sleep 1; done; kill $1 2>/dev/null; }
serve(){  # $1 = 0|1 ; sets YP, URL
  URL=http://127.0.0.1:$BASE_PORT
  YUNSHU_ROUND_DRIVER=$1 YUNSHU_MODEL=$M YUNSHU_AUTH_DISABLED=1 \
    $PY -m uvicorn yunshu_gateway.main:app --host 127.0.0.1 --port $BASE_PORT > $OUT/server-rd$1-$PHASE.log 2>&1 &
  YP=$!
  wait_ready $YP || return 1
  if [ $1 = 1 ]; then
    grep -q "Round driver: [0-9]* lane projections" $OUT/server-rd1-$PHASE.log && log "round driver engaged" || log "ROUND DRIVER NOT ENGAGED"
  fi
}
sweep(){ $PY scripts/research/sweep_round_driver.py $M --tokens 256 "$@"; }

run_phase(){
  case $1 in
    parity-1k)  sweep --phase parity --rows 4 --parity-rows 8 --output $OUT/parity-1k.jsonl ;;
    rows-1k)    sweep --phase rows --rows 1 2 4 8 --output $OUT/rows-1k.jsonl ;;
    parity-32k-a) sweep --phase parity --rows 1 --parity-rows 1 --warm-rows 1 --context 32768 --output $OUT/parity-32k.jsonl ;;
    parity-32k-b) sweep --phase parity --rows 1 --parity-rows 1 --warm-rows 1 --skip 1 --context 32768 --output $OUT/parity-32k.jsonl ;;
    rows-32k-a) sweep --phase rows --rows 1 4 --context 32768 --output $OUT/rows-32k.jsonl ;;
    rows-32k-b) sweep --phase rows --rows 8 --context 32768 --output $OUT/rows-32k.jsonl ;;
    decode-anatomy)
      for r in 1 4 8; do
        $PY scripts/research/probe_driver_decode.py $M --rows $r --no-mtp --output $OUT/anatomy.jsonl
        $PY scripts/research/probe_driver_decode.py $M --rows $r --output $OUT/anatomy.jsonl
      done ;;
    server-rd?-*)
      rd=${1#server-rd}; rd=${rd%%-*}; what=${1#server-rd?-}
      PHASE=$1
      serve $rd || { log "server rd=$rd failed"; stop $YP; return 1; }
      case $what in
        probe)
          $PY scripts/research/probe_concurrency.py --url $URL --model Qwen3.8-27B --n 8 \
            --note "round driver=$rd" --output $OUT/concurrency.jsonl > /dev/null 2>&1 || log "concurrency FAILED"
          $PY scripts/research/bench_engine_matrix.py --url $URL --model Qwen3.8-27B --engine yunshu-rd$rd \
            --checkpoint $M --pid $YP --note "round driver=$rd" --output $OUT/matrix.jsonl > /dev/null 2>&1 || log "matrix FAILED" ;;
        ctx1k)
          $PY scripts/research/bench_context_batch.py --url $URL --model Qwen3.8-27B --tokenizer $M --pid $YP \
            --lengths 1024 --batches 2 4 8 --note "round driver=$rd" \
            --output $OUT/context-batch.jsonl > /dev/null 2>&1 || log "context FAILED" ;;
        ctx32k)
          $PY scripts/research/bench_context_batch.py --url $URL --model Qwen3.8-27B --tokenizer $M --pid $YP \
            --lengths 32768 --batches 2 4 8 --batch-pp 32768 --note "round driver=$rd" \
            --output $OUT/context-batch.jsonl > /dev/null 2>&1 || log "context FAILED" ;;
        ctx131k-1)
          $PY scripts/research/bench_context_batch.py --url $URL --model Qwen3.8-27B --tokenizer $M --pid $YP \
            --lengths 131072 --batches --note "round driver=$rd" \
            --output $OUT/context-batch.jsonl > /dev/null 2>&1 || log "context FAILED" ;;
        ctx131k-[24])
          $PY scripts/research/bench_context_batch.py --url $URL --model Qwen3.8-27B --tokenizer $M --pid $YP \
            --lengths --batches ${1##*-} --batch-pp 131072 --note "round driver=$rd" \
            --output $OUT/context-batch.jsonl > /dev/null 2>&1 || log "context FAILED" ;;
        repeat)
          for n in 8192 32768; do
            $PY scripts/research/probe_repeat_doc.py --url $URL --model Qwen3.8-27B --tokenizer $M \
              --tokens $n --output $OUT/repeat-doc-rd$rd.jsonl > /dev/null 2>&1 || log "repeat FAILED"
          done ;;
        mixed16|mixed32)
          $PY scripts/research/bench_mixed_load.py --url $URL --model Qwen3.8-27B --tokenizer $M \
            --streams 4 --pp $((${what#mixed} * 1024)) --label rd$rd-$what --output $OUT/mixed-load.jsonl > /dev/null 2>&1 || log "mixed FAILED" ;;
      esac
      stop $YP ;;
    mmlu-?|mmlu0-?)
      rd=1; [ ${1%%-*} = mmlu0 ] && rd=0
      k=${1##*-}
      PHASE=$1
      serve $rd || { log "mmlu server failed"; stop $YP; return 1; }
      $PY scripts/research/soak_mmlu_pro.py --url $URL --model Qwen3.8-27B --pid $YP \
        --start $((100 * k)) --n 100 --final-idle-s 15 \
        --note "Yunshu round driver=$rd, slice $k" --output $OUT/mmlu-rd$rd-$k.jsonl > $OUT/mmlu-rd$rd-$k.log 2>&1 || log "mmlu FAILED"
      stop $YP ;;
    *) echo "unknown phase $1"; return 2 ;;
  esac
}

case $PHASE in
  list) printf '%s\n' $PHASES ;;
  all)
    for p in $PHASES; do
      log "phase $p"
      if [ -n "${GPU_RUN:-}" ]; then ${=GPU_RUN}-$p env PHASE=$p zsh $0; else run_phase $p; fi
    done ;;
  *) run_phase $PHASE ;;
esac
log "round driver validation: $PHASE done"
