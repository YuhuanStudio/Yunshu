#!/bin/bash
# Solo sampled spec on/off, 3 reps, one job: fnspec_speed.sh OUTDIR
set -euo pipefail
D=$(dirname "$0")
for arm in on off; do "$D/fnspec_job.sh" $arm "$1" --reps 3; done
