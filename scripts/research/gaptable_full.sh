#!/bin/bash
# Lead-only full run: final main, >=3 reps; submit/wait one cell at a time, fail closed.
set -euo pipefail
: "${ROOT:?set ROOT to final-main worktree}"
: "${OUT:?set OUT to a new results directory}"
MANIFEST=${MANIFEST:-/Volumes/P5Plus/yunshu-build/competitors/compsync/manifest.json}
PY=/Users/yuhuan/Documents/YuhuanStudio/Yunshu/.venv/bin/python
GPUQ=/Users/yuhuan/Documents/YuhuanStudio/Yunshu/scripts/dev/gpuq
export GPUQ_DIR=/Volumes/P5Plus/yunshu-gpuq GPUQ_OWNER=compsync-final
export TFB_WORK=/Volumes/P5Plus/yunshu-build/snapshot014/corpus TFB_EXACT_PROMPTS=1
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$OUT/tmp"
test ! -e "$OUT" || { echo "OUT already exists; choose a new results directory"; exit 1; }
git -C "$ROOT" diff --quiet HEAD -- python scripts/research || { echo "source tree is dirty"; exit 1; }
mkdir -p "$OUT" "$TMPDIR"
cp "$MANIFEST" "$OUT/manifest.json"
MANIFEST="$OUT/manifest.json"
"$PY" "$ROOT/scripts/research/gaptable.py" plan --manifest "$MANIFEST" --src "$ROOT" --out "$OUT" > "$OUT/plan.json"
# Materialize executable drivers without shell interpolation of arguments.
"$PY" - "$OUT" <<'PY'
import json, pathlib, shlex, sys
out = pathlib.Path(sys.argv[1])
for i, job in enumerate(json.loads((out/'plan.json').read_text())['jobs']):
    driver = out/f'cell-{i:03d}.sh'
    driver.write_text('#!/bin/bash\nset -euo pipefail\nexec ' + shlex.join(job['argv']) + '\n')
    driver.chmod(0o755)
PY
expected=$(git -C "$ROOT" rev-parse HEAD)
for driver in "$OUT"/cell-*.sh; do
  test "$(git -C "$ROOT" rev-parse HEAD)" = "$expected" || { echo 'main changed during run'; exit 1; }
  cell=$(basename "$driver" .sh)
  job=$("$GPUQ" submit --quiet --label "compsync-final-$(basename "$OUT")-$cell" --priority 1 --timeout 30 -- "$driver")
  printf '%s\n' "$job" >> "$OUT/job_ids.txt"
  while ! ls "$GPUQ_DIR/logs/$job.rc" >/dev/null 2>&1; do sleep 60; done
  test "$(cat "$GPUQ_DIR/logs/$job.rc")" = 0 || { tail -80 "$GPUQ_DIR/logs/$job.log"; exit 1; }
done
printf '{"complete":true,"repetitions":3,"yunshu_sha":"%s"}\n' "$expected" > "$OUT/complete.json"

"$PY" "$ROOT/scripts/research/gaptable.py" summary --manifest "$MANIFEST" --src "$ROOT" --out "$OUT" > "$OUT/table.md"
