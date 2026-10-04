#!/bin/sh
# Consolidate three fresh-server A/B rounds without losing failed-arm receipts.
set -u
rc=0
for run in 5 6 7; do
  /bin/sh scripts/research/agentic/i8_ab.sh "$run" "$((run % 2))"
  arm_rc=$?
  echo "round=$run rc=$arm_rc"
  [ "$arm_rc" -ne 0 ] && rc=1
done
exit "$rc"
