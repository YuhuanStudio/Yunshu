#!/bin/zsh
# Yunshu release gate: is main usable enough to ship? (docs/guides/RELEASE_GATE.md)
#
#   zsh scripts/release/gate.sh                         # install, serve-27b, families, soak
#   STAGE=install,serve-27b zsh scripts/release/gate.sh # any subset, comma-separated
#   STAGE=service GATE_SERVICE=1 zsh scripts/release/gate.sh   # loads a real launchd agent
#
# Stages: install serve-27b families soak service. Model paths come from the environment
# or scripts/research/local.env (gitignored; names in scripts/release/local.env.example).
# Everything the gate installs or writes lives under $GATE_ROOT (HOME, uv tool dirs,
# caches), never in your real ~/.yunshu or ~/.local. Every check prints PASS/FAIL/SKIP
# into $OUT/results-<time>.jsonl; the run ends with a table and exits 1 on any FAIL.
set -u
setopt NULL_GLOB
cd "$(dirname "$0")/../.."
ROOT=$PWD
[ -f scripts/research/local.env ] && source scripts/research/local.env

GATE_ROOT=${GATE_ROOT:-/Volumes/P5Plus/yunshu-build/gate}
OUT=${OUT:-docs/research/runs/$(date +%Y-%m-%d)-release-gate}
STAGE=${STAGE:-install,serve-27b,families,soak}
PORT=${PORT:-18764}
URL=http://127.0.0.1:$PORT
PY=${PY:-$ROOT/.venv/bin/python}           # harness interpreter (the repo venv)
MODELS_DIR=${GATE_MODELS_DIR:-/Volumes/P5Plus/models}
PULL_REPO=${PULL_REPO:-Jundot/Qwen3.8-27B-oQ4e-mtp}   # already in MODELS_DIR: must not download
SOAK_MINUTES=${SOAK_MINUTES:-30}
MMLU_BASELINE=${MMLU_BASELINE:-249}        # 27B MMLU-Pro 300 b8, 2026-09-29 (ragged default)
MMLU_TOLERANCE=${MMLU_TOLERANCE:-3}

mkdir -p $OUT $GATE_ROOT
OUT=${OUT:A}
RESULTS=$OUT/results-$(date +%H%M%S).jsonl
GH=$GATE_ROOT/home                         # HOME for every yunshu command and server
BV=$GATE_ROOT/bin-vision/yunshu            # README quickstart install: yunshu[vision]
BA=$GATE_ROOT/bin-all/yunshu               # yunshu[all]: audio, omni, generation too
export UV_CACHE_DIR=$GATE_ROOT/uv-cache UV_PYTHON_INSTALL_DIR=$GATE_ROOT/uv-python
YENV=(env HOME=$GH HF_HOME=$GH/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
      YUNSHU_MODELS_DIR=$MODELS_DIR)
CLIENT=(uv run --no-project --quiet --python 3.13 --with openai --with anthropic
        --with httpx --with pillow python $ROOT/scripts/release/gate_checks.py
        --results $RESULTS)

log(){ echo "$(date +%H:%M:%S) $*"; }
has(){ [[ ",$STAGE," == *",$1,"* ]]; }
rec(){ $PY $ROOT/scripts/release/gate_checks.py --results $RESULTS record "$@"; }
jget(){ $PY -c "import json,sys; d=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]; s=[x for x in d if x.get('kind')=='summary'][-1]; print(eval(sys.argv[2], {}, s))" "$@" 2>/dev/null; }
port_free(){ ! curl -s -m 2 -o /dev/null $URL/health; }
serve(){  # $1 binary, $2 model, $3 log -> sets YP; 0 when ready
  ${YENV[@]} $1 serve -m $2 -p $PORT > $3 2>&1 &
  YP=$!
  for i in $(seq 1 450); do
    curl -s -m 2 $URL/health/ready 2>/dev/null | grep -q '"ready":true' && return 0
    kill -0 $YP 2>/dev/null || return 1
    sleep 2
  done
  return 1
}
stop(){ kill -INT $1 2>/dev/null; for i in $(seq 1 60); do kill -0 $1 2>/dev/null || return 0; sleep 1; done; kill -9 $1 2>/dev/null; }
log_clean(){  # $1 check name, $2 server log
  local n=$(grep -c "Traceback" $2)
  if [ $n = 0 ]; then rec $1 PASS "no Traceback in $(basename $2)"
  else rec $1 FAIL "$n Traceback(s) in $(basename $2): $(grep -m1 -E '^[A-Za-z_.]+(Error|Exception)' $2 | cut -c1-160)"; fi
}
need_bin(){ [ -x $1 ] && return 0; rec $2 FAIL "$1 missing: run STAGE=install first"; return 1; }

log "gate: stages=$STAGE out=$OUT"
port_free || { rec gate.port FAIL "something already answers on $URL"; STAGE=; }

# ── 1. install: build, clean isolated installs, first-run commands ───────────────────
if has install; then
  log "stage install"
  rm -rf $GATE_ROOT/dist $GATE_ROOT/tool-* $GATE_ROOT/bin-* $GH
  mkdir -p $GATE_ROOT/dist $GH
  if uv build --out-dir $GATE_ROOT/dist > $OUT/install-build.log 2>&1; then
    rec install.build PASS "$(cd $GATE_ROOT/dist && ls | tr '\n' ' ')"
  else rec install.build FAIL "uv build failed (install-build.log)"; fi
  WHEEL=$(ls $GATE_ROOT/dist/*.whl 2>/dev/null | head -1)
  for flavor in vision all; do
    if [ -n "$WHEEL" ] && UV_TOOL_DIR=$GATE_ROOT/tool-$flavor UV_TOOL_BIN_DIR=$GATE_ROOT/bin-$flavor \
        uv tool install --python 3.13 "yunshu[$flavor] @ $WHEEL" > $OUT/install-$flavor.log 2>&1; then
      rec install.uv_tool_$flavor PASS "$GATE_ROOT/bin-$flavor/yunshu"
    else rec install.uv_tool_$flavor FAIL "uv tool install failed (install-$flavor.log)"; fi
  done
  if need_bin $BV install.cli; then
    want=$($PY -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")
    got=$(${YENV[@]} $BV --version 2>&1)
    [[ $got == *$want* ]] && rec install.version PASS "$got" || rec install.version FAIL "want $want, got $got"
    if ${YENV[@]} $BV --json doctor --port $PORT > $OUT/install-doctor.json 2>$OUT/install-doctor.err; then
      rec install.doctor PASS "no failed checks"
    else rec install.doctor FAIL "$($PY -c "import json;d=json.load(open('$OUT/install-doctor.json'));print([c['name']+': '+c['detail'] for c in d['checks'] if c['status']=='fail'])" 2>&1 | cut -c1-300)"; fi
    n=$(${YENV[@]} $BV --json model list 2>/dev/null | $PY -c "import json,sys; print(len(json.load(sys.stdin)['models']))" 2>/dev/null)
    [ "${n:-0}" -gt 0 ] && rec install.model_list PASS "$n models in $MODELS_DIR" || rec install.model_list FAIL "no models listed in $MODELS_DIR"
    st=$(${YENV[@]} $BV --json pull $PULL_REPO 2>/dev/null | $PY -c "import json,sys; print(json.load(sys.stdin).get('status'))" 2>/dev/null)
    [ "$st" = present ] && rec install.pull_refuses_redownload PASS "$PULL_REPO already on disk" \
      || rec install.pull_refuses_redownload FAIL "status=$st (expected present, offline so nothing downloaded)"
    tmpdir=$GATE_ROOT/models-elsewhere
    if ${YENV[@]} $BV config set models_dir $tmpdir > /dev/null 2>&1 \
        && grep -q "$tmpdir" $GH/.yunshu/config.toml 2>/dev/null \
        && ${YENV[@]} $BV config unset models_dir > /dev/null 2>&1; then
      rec install.config_set_models_dir PASS "written to $GH/.yunshu/config.toml and removed"
    else rec install.config_set_models_dir FAIL "config set/unset models_dir did not round-trip"; fi
    if ${YENV[@]} $BV service install -m $PULL_REPO --port $PORT --dry-run > $OUT/install-service.txt 2>&1 \
        && grep -q "plist\|Label" $OUT/install-service.txt && [ ! -e $GH/Library/LaunchAgents ]; then
      rec install.service_dry_run PASS "plist printed, nothing written"
    else rec install.service_dry_run FAIL "see install-service.txt"; fi
  fi
fi

# ── 2. serve-27b: the tier-1 model on the installed quickstart binary ───────────────
if has serve-27b; then
  log "stage serve-27b"
  if [ -z "${M:-}" ] || [ ! -d "$M" ]; then rec serve-27b.model FAIL "M (Qwen3.8-27B) not set or missing"
  elif need_bin $BV serve-27b.boot; then
    if serve $BV $M $OUT/27b-server.log; then
      rec serve-27b.boot PASS "ready on $URL (pid $YP)"
      rm -f $OUT/27b-matrix.jsonl $OUT/27b-concurrency.jsonl
      $PY scripts/research/bench_engine_matrix.py --url $URL --model Qwen3.8-27B --engine gate-27b \
        --checkpoint $M --pid $YP --note "release gate" --output $OUT/27b-matrix.jsonl > $OUT/27b-matrix.log 2>&1
      ok=$(jget $OUT/27b-matrix.jsonl "f\"{ok}/{total} {failed}\"")
      [[ $ok == 34/34* ]] && rec serve-27b.matrix PASS "$ok" || rec serve-27b.matrix FAIL "${ok:-no summary (27b-matrix.log)}"
      ${CLIENT[@]} --prefix serve-27b. sdk --url $URL
      ${CLIENT[@]} --prefix serve-27b. cancel --url $URL
      ${CLIENT[@]} --prefix serve-27b. long --url $URL --tokens 32768
      $PY scripts/research/probe_concurrency.py --url $URL --model Qwen3.8-27B --n 8 \
        --note "release gate" --output $OUT/27b-concurrency.jsonl > /dev/null 2>&1
      qa=$(jget $OUT/27b-concurrency.jsonl "f\"{qa_concurrent_ok}/{qa_n} agg {aggregate_tps} tok/s\"")
      [[ $qa == 8/8* ]] && rec serve-27b.concurrency PASS "$qa" || rec serve-27b.concurrency FAIL "${qa:-no summary}"
      # The server must decode as fast as the engine in-process on the same prompt
      # (same greedy tokens); a per-token serving overhead fails the gate.
      sp=$(${YENV[@]} $PY scripts/release/check_server_path.py --url $URL \
        --model $M --server-log $OUT/27b-server.log --output $OUT/27b-server-path.json 2> $OUT/27b-server-path.log | tail -1)
      spd=$($PY -c "import json,sys; d=json.loads(sys.argv[1]); print(f\"server/in-process worst {d['worst_ratio']} (min {d['min_ratio']}), same text {d['same_text']}, spec {d['spec']}: \" + ', '.join(f\"{c['task']}@{c['context']} {c['server_tps']}/{c['inprocess_tps']}\" for c in d['cases']))" "$sp" 2>/dev/null)
      [[ $sp == *'"status": "PASS"'* ]] && rec serve-27b.server_path PASS "$spd" \
        || rec serve-27b.server_path FAIL "${spd:-could not measure (27b-server-path.log)}"
    else rec serve-27b.boot FAIL "server did not become ready (27b-server.log)"; fi
    stop $YP
    log_clean serve-27b.log_clean $OUT/27b-server.log
  fi
fi

# ── 3. families: one server per model, short smoke per modality ─────────────────────
if has families; then
  log "stage families"
  say -o $OUT/asr.wav --file-format=WAVE --data-format=LEI16@16000 \
    "The quick brown fox jumps over the lazy dog." 2>/dev/null
  FAMILIES=(
    "qwen35-0.8b|M_QWEN35_08B|chat,stream"
    "qwen35-9b-4bit|M_QWEN35_9B|chat,stream,tools,schema,image"
    "text-lm|M_TEXT_LM|chat,stream,tools,schema"
    "gemma4-e4b|M_GEMMA|chat,stream,tools,image"
    "glm-ocr|M_OCR|ocr"
    "asr|M_ASR|asr"
    "tts|M_TTS|tts"
    "qwen3-omni|M_OMNI|chat"
    "image-gen|M_IMAGE|imagegen"
  )
  if need_bin $BA families.binary; then
    for spec in $FAMILIES; do
      label=${spec%%|*}; rest=${spec#*|}; var=${rest%%|*}; kinds=${rest#*|}
      mpath=${(P)var:-}
      if [ -z "$mpath" ] || [ ! -e "$mpath" ]; then
        rec families.$label SKIP "$var not set or missing (scripts/research/local.env)"; continue
      fi
      log "family $label"
      if serve $BA $mpath $OUT/family-$label-server.log; then
        rec families.$label.boot PASS "$(basename $mpath)"
        ${CLIENT[@]} --prefix families.$label. family --url $URL --kinds $kinds --audio $OUT/asr.wav
      else rec families.$label.boot FAIL "not ready (family-$label-server.log)"; fi
      stop $YP
      log_clean families.$label.log_clean $OUT/family-$label-server.log
      sleep 3
    done
  fi
fi

# ── 4. soak: MMLU-Pro 300 b8 accuracy + a mixed realistic soak, memory returns ───────
if has soak; then
  log "stage soak"
  MMLU_DATA=$ROOT/reference/omlx/omlx/eval/data/mmlu_pro_test.jsonl  # soak_mmlu_pro.py inputs
  MMLU_IDS=${MMLU_IDS:-$HOME/Downloads/Qwen3.8-27B-oQ4e-mtp_mmlu_pro.json}
  if [ -z "${M:-}" ] || [ ! -d "$M" ]; then rec soak.model FAIL "M (Qwen3.8-27B) not set or missing"
  elif [ ! -f $MMLU_DATA ] || [ ! -f $MMLU_IDS ]; then
    rec soak.inputs FAIL "need $MMLU_DATA (reference/omlx clone) and the question-id file $MMLU_IDS"
  elif need_bin $BV soak.boot; then
    if serve $BV $M $OUT/soak-server.log; then
      rm -f $OUT/soak-mmlu.jsonl $OUT/soak-realistic.jsonl
      log "soak mmlu"
      $PY scripts/research/soak_mmlu_pro.py --url $URL --model Qwen3.8-27B --pid $YP \
        --ids $MMLU_IDS --note "release gate" --output $OUT/soak-mmlu.jsonl > $OUT/soak-mmlu.log 2>&1
      s=$(jget $OUT/soak-mmlu.jsonl "f\"{correct} {errors} {n} {time_s} {tok_per_s} {start_footprint_gib} {max_footprint_gib} {end_footprint_gib}\"")
      if [ -z "$s" ]; then rec soak.mmlu FAIL "no summary (soak-mmlu.log)"
      else
        read correct errors n secs tps m0 mmax m1 <<< "$s"
        d=$(( correct - MMLU_BASELINE )); d=${d#-}
        detail="$correct/$n, $errors errors, $(( ${secs%.*} / 60 )) min, $tps tok/s, footprint $m0 -> max $mmax -> $m1 GiB"
        [ $n = 300 ] && [ $errors = 0 ] && [ $d -le $MMLU_TOLERANCE ] \
          && rec soak.mmlu PASS "$detail" || rec soak.mmlu FAIL "$detail (baseline $MMLU_BASELINE ±$MMLU_TOLERANCE)"
        $PY -c "import sys; sys.exit(0 if float('$m1') - float('$m0') < 4 else 1)" \
          && rec soak.mmlu_memory_returns PASS "start $m0, end $m1 GiB" \
          || rec soak.mmlu_memory_returns FAIL "start $m0, end $m1 GiB after idle"
      fi
      log "soak realistic $SOAK_MINUTES min"
      $PY scripts/research/soak_realistic.py --url $URL --model Qwen3.8-27B --pid $YP \
        --minutes $SOAK_MINUTES --note "release gate" --output $OUT/soak-realistic.jsonl > $OUT/soak-realistic.log 2>&1
      s=$(jget $OUT/soak-realistic.jsonl "f\"{requests} {ok} {errors} {start_footprint_gib} {max_footprint_gib} {end_footprint_gib}\"")
      if [ -z "$s" ]; then rec soak.realistic FAIL "no summary (soak-realistic.log)"
      else
        read reqs okn errs m0 mmax m1 <<< "$s"
        detail="$okn/$reqs ok, $errs errors, footprint $m0 -> max $mmax -> $m1 GiB"
        [ $errs = 0 ] && [ $(( okn * 100 )) -ge $(( reqs * 95 )) ] \
          && rec soak.realistic PASS "$detail" || rec soak.realistic FAIL "$detail (need 0 errors, >=95% ok)"
        $PY -c "import sys; sys.exit(0 if float('$m1') - float('$m0') < 4 else 1)" \
          && rec soak.realistic_memory_returns PASS "start $m0, end $m1 GiB" \
          || rec soak.realistic_memory_returns FAIL "start $m0, end $m1 GiB after idle"
      fi
    else rec soak.boot FAIL "server did not become ready (soak-server.log)"; fi
    stop $YP
    log_clean soak.log_clean $OUT/soak-server.log
  fi
fi

# ── 5. service: a real launchd agent (only with GATE_SERVICE=1) ──────────────────────
if has service; then
  if [ "${GATE_SERVICE:-0}" != 1 ]; then
    rec service.launchd SKIP "loads a real launchd agent; run with GATE_SERVICE=1 after the user agrees"
  elif need_bin $BV service.launchd && [ -n "${M:-}" ]; then
    log "stage service"
    if ${YENV[@]} $BV service install -m $M --port $PORT --force > $OUT/service.log 2>&1; then
      up=0
      for i in $(seq 1 300); do curl -s -m 2 $URL/health/ready | grep -q '"ready":true' && { up=1; break; }; sleep 2; done
      ${YENV[@]} $BV service status >> $OUT/service.log 2>&1
      [ $up = 1 ] && rec service.launchd PASS "agent started and served $URL" || rec service.launchd FAIL "not ready (service.log)"
    else rec service.launchd FAIL "service install failed (service.log)"; fi
    ${YENV[@]} $BV service uninstall >> $OUT/service.log 2>&1
    sleep 3
    port_free && rec service.uninstall PASS "agent removed, port free" || rec service.uninstall FAIL "port still answering after uninstall"
  fi
fi

log "gate done"
$PY $ROOT/scripts/release/gate_checks.py --results $RESULTS summary | tee $OUT/summary-$(basename $RESULTS .jsonl).txt
exit ${pipestatus[1]}
