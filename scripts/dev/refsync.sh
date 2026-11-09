#!/bin/bash
# Full-history, full-blob clones of every reference repo (scripts/dev/refsync_repos.txt),
# fast-forwarded to upstream's default branch. Read competitor/upstream SOURCE here, never only a
# release binary. Partial (--filter) clones are converted to full ones. Never resets or discards
# local work: a repo with local changes or a non-fast-forward branch is reported, not touched.
# Writes $ROOT/REFSYNC.md (repo, head, date, behind-before, state) and $ROOT/logs/refsync-<repo>.log.
set -u
ROOT=${REFSYNC_ROOT:-/Volumes/Micron/yunshu-reference}
LIST="$(cd "$(dirname "$0")" && pwd)/refsync_repos.txt"
mkdir -p "$ROOT/logs"
out="$ROOT/REFSYNC.md.tmp"
printf '# Reference sync %s\n\n| repo | head | date | new commits | state |\n|---|---|---|---|---|\n' "$(date '+%Y-%m-%d %H:%M')" > "$out"
sync_one() {
  local name=$1 url=$2 dir="$ROOT/$1" log="$ROOT/logs/refsync-$1.log"
  : > "$log"
  if [ ! -d "$dir/.git" ]; then
    git clone "$url" "$dir" >>"$log" 2>&1 || { echo "FAIL clone"; return; }
    echo "ok (cloned)"; return
  fi
  if [ -n "$(git -C "$dir" config --get remote.origin.partialclonefilter)" ]; then
    git -C "$dir" config --unset remote.origin.partialclonefilter
    git -C "$dir" config remote.origin.promisor false
    git -C "$dir" fetch --refetch --tags origin >>"$log" 2>&1 || { echo "FAIL refetch (partial->full)"; return; }
  fi
  git -C "$dir" fetch --tags --prune origin >>"$log" 2>&1 || { echo "FAIL fetch"; return; }
  git -C "$dir" remote set-head origin -a >>"$log" 2>&1
  local def; def=$(git -C "$dir" symbolic-ref --short refs/remotes/origin/HEAD 2>>"$log") || { echo "FAIL no origin/HEAD"; return; }
  local br=${def#origin/}
  if [ -n "$(git -C "$dir" status --porcelain --untracked-files=no)" ]; then echo "SKIP local changes"; return; fi
  git -C "$dir" checkout --quiet "$br" >>"$log" 2>&1 || git -C "$dir" checkout --quiet -b "$br" "$def" >>"$log" 2>&1 || { echo "FAIL checkout $br"; return; }
  local behind; behind=$(git -C "$dir" rev-list --count "HEAD..$def")
  git -C "$dir" merge --ff-only "$def" >>"$log" 2>&1 || { echo "FAIL not fast-forward ($behind behind)"; return; }
  echo "ok +$behind"
}
grep -v '^#' "$LIST" | while read -r name url; do
  [ -z "$name" ] && continue
  state=$(sync_one "$name" "$url")
  head=$(git -C "$ROOT/$name" log -1 --format='%h|%cs' 2>/dev/null || echo '-|-')
  n=${state#ok +}; [ "$n" = "$state" ] && n=-
  printf '| %s | %s | %s | %s | %s |\n' "$name" "${head%|*}" "${head#*|}" "$n" "${state%% +*}" >> "$out"
done
mv "$out" "$ROOT/REFSYNC.md"
cat "$ROOT/REFSYNC.md"
