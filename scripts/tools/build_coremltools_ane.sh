#!/usr/bin/env bash
# Build coremltools from GitHub main with its NATIVE C++ extensions, against the repo's
# Python 3.13 venv — the ONLY way to get a working ANE embedding path on this machine.
#
# WHY (Wave 697): the released coremltools 9.0 can't convert real BERT-class embedding
# models under torch 2.12 (its torch frontend mishandles the int-cast op). GitHub main
# fixes that — but a plain `pip install git+…` builds only the pure-Python package, leaving
# the native extensions absent ("BlobWriter not loaded", "No module named libcoremlpython").
# This script cmake-builds milstoragepython + modelpackage + coremlpython against the venv's
# Python 3.13 and drops them into site-packages, so conversion AND predict() both work.
#
# Result (verified, 30-core M3 Max): ANE embedding ~4-5x faster than MLX-GPU; real BERT
# converts + runs end-to-end with torch 2.12 (NO torch downgrade needed).
#
# Usage:  scripts/tools/build_coremltools_ane.sh
# Idempotent-ish: re-running re-clones to /tmp and rebuilds.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="${VENV:-$REPO/.venv}"
PY="$VENV/bin/python"
PYVER="$("$PY" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PYINC="$("$PY" -c 'import sysconfig;print(sysconfig.get_path("include"))')"
PYLIBDIR="$("$PY" -c 'import sysconfig;print(sysconfig.get_config_var("LIBDIR"))')"
PYLIB="$PYLIBDIR/libpython${PYVER}.dylib"
SITE="$("$PY" -c 'import site;print(site.getsitepackages()[0])')"
SRC="/tmp/coremltools_src"

echo ">>> venv=$VENV python=$PYVER inc=$PYINC lib=$PYLIB"
command -v "$VENV/bin/cmake" >/dev/null 2>&1 || { echo "installing cmake into venv"; uv pip install --python "$PY" cmake; }

# 1. pure-python package from GitHub main (fixes the torch-frontend int-cast bug)
PATH="$VENV/bin:$PATH" uv pip install --python "$PY" \
  "coremltools @ git+https://github.com/apple/coremltools.git"

# 2. native extensions via cmake, forced onto THIS python's headers/lib
rm -rf "$SRC"; git clone --depth 1 https://github.com/apple/coremltools.git "$SRC"
cd "$SRC"
PATH="$VENV/bin:$PATH" cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE="$PY" -DPYTHON_INCLUDE_DIR="$PYINC" -DPYTHON_LIBRARY="$PYLIB"
PATH="$VENV/bin:$PATH" cmake --build build --target milstoragepython modelpackage coremlpython -j8

# 3. install the built .dylibs as importable .so into the coremltools package
for lib in milstoragepython modelpackage coremlpython; do
  cp -f "build/lib${lib}.dylib" "$SITE/coremltools/lib${lib}.so"
  echo "    installed lib${lib}.so"
done

echo ">>> verifying…"
"$PY" - <<'PYEOF'
from coremltools.libmilstoragepython import _BlobStorageWriter  # noqa
from coremltools.libmodelpackage import ModelPackage            # noqa
import coremltools as ct
print("OK: coremltools", ct.__version__, "with native extensions (BlobWriter + ModelPackage + predict)")
PYEOF
echo ">>> done. Run: PYTHONPATH=python $PY scripts/bench/bench_ane_real.py --real-bert"
