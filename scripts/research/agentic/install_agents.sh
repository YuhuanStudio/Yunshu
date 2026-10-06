#!/bin/sh
# Install pinned copies of the three coding-agent CLIs under /Volumes/P5Plus, isolated from the user's Mac.
set -e
ROOT=${AGENTIC_CLIS:-/Volumes/P5Plus/yunshu-test-envs/agentic-clis}
HOMEDIR=${AGENTIC_INSTALL_HOME:-/Volumes/P5Plus/yunshu-build/agentic/install-home}
mkdir -p "$ROOT" "$HOMEDIR"
export HOME="$HOMEDIR" npm_config_cache="$ROOT/npm-cache" npm_config_update_notifier=false
npm install --prefix "$ROOT" --no-audit --no-fund \
  opencode-ai@"${OPENCODE_V:-1.18.33}" @anthropic-ai/claude-code@"${CLAUDE_V:-2.1.285}" @openai/codex@"${CODEX_V:-0.157.1}"
# npm 12 blocks install scripts; run the two native-binary fetchers by hand.
(cd "$ROOT/node_modules/@anthropic-ai/claude-code" && node install.cjs)
(cd "$ROOT/node_modules/opencode-ai" && node postinstall.mjs)
for c in claude opencode codex; do "$ROOT/node_modules/.bin/$c" --version; done
