#!/bin/sh
# Interleave two environments in rounds so drift hits both equally.
# usage: http_ttfb_rounds.sh OUT_DIR ROUNDS N_PER_ROUND PYTHON_A PYTHON_B
set -e
out=$1; rounds=$2; n=$3; pa=$4; pb=$5
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
i=0
while [ "$i" -lt "$rounds" ]; do
  "$pa" "$here/http_ttfb_ab.py" run --n "$n" --out "$out/a$i.json" >/dev/null 2>&1
  "$pb" "$here/http_ttfb_ab.py" run --n "$n" --out "$out/b$i.json" >/dev/null 2>&1
  i=$((i + 1))
done
echo "A:"; "$pa" "$here/http_ttfb_ab.py" compare "$out"/a*.json
echo "B:"; "$pb" "$here/http_ttfb_ab.py" compare "$out"/b*.json
