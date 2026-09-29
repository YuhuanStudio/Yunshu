#!/bin/zsh
# Start `yunshu serve -m MODEL -p PORT`, wait for /health, run `python SCRIPT PORT [args]`, then
# kill -9 the server whatever happens.
#   usage: serve_and_run.sh MODEL PORT SCRIPT [script args...]
# Run it through the GPU queue:
#   scripts/dev/gpuq run --priority 1 --timeout 5 --stall 2 -- scripts/realmodel/serve_and_run.sh ...
# The ollama SDK is read from $OLLAMA_PYLIBS (a --target install) when it is not in the venv.
M=$1; P=$2; S=$3; shift 3
ROOT=${0:A:h:h:h}
cd "$ROOT" || exit 2
export PYTHONPATH=$ROOT/python${OLLAMA_PYLIBS:+:$OLLAMA_PYLIBS}
PY=${YUNSHU_PY:-/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python}
LOG=${TMPDIR:-/tmp}/serve_and_run_$P.log
$PY -m yunshu_cli serve -m "$M" -p "$P" > "$LOG" 2>&1 &
SP=$!
trap 'kill -9 $SP 2>/dev/null; wait $SP 2>/dev/null' EXIT INT TERM
for i in $(seq 1 240); do
  curl -s "localhost:$P/health" >/dev/null && break
  kill -0 $SP 2>/dev/null || { echo "server died"; tail -30 "$LOG"; exit 3; }
  sleep 1
done
$PY "$S" "$P" "$@"
RC=$?
[ $RC -ne 0 ] && tail -20 "$LOG"
exit $RC
