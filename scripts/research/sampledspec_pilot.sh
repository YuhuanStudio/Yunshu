#!/bin/bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
"$ROOT/scripts/research/sampledspec_unit.sh"
"$ROOT/scripts/research/sampledspec_identity.sh" 1024 0.7 1 0 code
printf '{"complete":true}\n' > "$ROOT/docs/research/sampledspec/pilot.complete.json"
