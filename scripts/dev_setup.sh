#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Yunshu Developer Setup Script
# Checks prerequisites, creates venv, installs deps, runs smoke test.
#
# Usage:
#   cd yunshu
#   bash scripts/dev_setup.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }

ERRORS=0

# ─── Section Header ──────────────────────────────────────────
section() {
    echo ""
    echo -e "${BLUE}━━━ $1 ━━━${NC}"
}

# ─── 1. Python version ───────────────────────────────────────
section "Checking Python version"

PYTHON_CMD=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON_CMD="$cmd"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    fail "Python not found. Install Python 3.12+ and try again."
    ERRORS=$((ERRORS + 1))
else
    PY_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PY_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
    PY_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")

    if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 12 ]; }; then
        fail "Python $PY_VERSION found, but 3.12+ is required."
        ERRORS=$((ERRORS + 1))
    else
        pass "Python $PY_VERSION (>= 3.12)"
    fi
fi

# ─── 2. uv ────────────────────────────────────────────────────
section "Checking uv (Python package manager)"

if command -v uv &>/dev/null; then
    UV_VERSION=$(uv --version 2>/dev/null || echo "unknown")
    pass "uv $UV_VERSION"
else
    fail "uv not found."
    echo "  Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
    ERRORS=$((ERRORS + 1))
fi

# ─── 3. just ──────────────────────────────────────────────────
section "Checking just (command runner)"

if command -v just &>/dev/null; then
    JUST_VERSION=$(just --version 2>/dev/null || echo "unknown")
    pass "just $JUST_VERSION"
else
    warn "just not found (optional, but recommended)."
    echo "  Install: brew install just  OR  cargo install just"
    echo "  You can still run commands manually via uv run ..."
fi

# ─── 4. pnpm ──────────────────────────────────────────────────
section "Checking pnpm (Node.js package manager for WebUI)"

if command -v pnpm &>/dev/null; then
    PNPM_VERSION=$(pnpm --version 2>/dev/null || echo "unknown")
    pass "pnpm $PNPM_VERSION"
else
    warn "pnpm not found (only needed for WebUI development)."
    echo "  Install: corepack enable && corepack prepare pnpm@latest --activate"
fi

# ─── 5. Platform check ───────────────────────────────────────
section "Checking platform"

OS_NAME=$(uname -s)
if [ "$OS_NAME" = "Darwin" ]; then
    CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
    # Check for Apple Silicon
    ARCH=$(uname -m)
    if [ "$ARCH" = "arm64" ]; then
        pass "macOS on Apple Silicon (arm64)"
    else
        warn "macOS on Intel (x86_64). MLX requires Apple Silicon."
    fi
else
    warn "Platform: $OS_NAME. Yunshu targets macOS with Apple Silicon."
fi

# ─── 6. Install dependencies ─────────────────────────────────
section "Installing Python dependencies"

if command -v uv &>/dev/null; then
    info "Running: uv sync --all-extras --dev"
    if uv sync --all-extras --dev; then
        pass "Python dependencies installed"
    else
        fail "uv sync failed. Check pyproject.toml and network."
        ERRORS=$((ERRORS + 1))
    fi
else
    warn "Skipping dependency install (uv not available)."
fi

# ─── 7. Install WebUI dependencies ───────────────────────────
section "Setting up WebUI (if applicable)"

if [ -d "$PROJECT_ROOT/webui" ] && command -v pnpm &>/dev/null; then
    info "Installing WebUI dependencies..."
    if (cd "$PROJECT_ROOT/webui" && pnpm install); then
        pass "WebUI dependencies installed"
    else
        warn "WebUI install failed (non-fatal)."
    fi
else
    info "Skipping WebUI setup (no webui/ dir or pnpm not found)."
fi

# ─── 8. Smoke test: import check ─────────────────────────────
section "Running smoke test (import check)"

if command -v uv &>/dev/null; then
    SMOKE_SCRIPT=$(cat <<'PYEOF'
import sys
sys.path.insert(0, "python")

packages = [
    "yunshu_engine",
    "yunshu_gateway",
    "yunshu_control",
    "yunshu_kv",
    "yunshu_mesh",
    "yunshu_api",
    "yunshu_sdk",
    "yunshu_cli",
]

failed = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  [OK] {pkg}")
    except Exception as e:
        print(f"  [FAIL] {pkg}: {e}")
        failed.append(pkg)

if failed:
    print(f"\nFailed imports: {', '.join(failed)}")
    sys.exit(1)
else:
    print(f"\nAll {len(packages)} packages imported successfully.")
PYEOF
)

    if echo "$SMOKE_SCRIPT" | uv run python - 2>/dev/null; then
        pass "All packages importable"
    else
        warn "Some packages failed to import. Run with verbose mode for details."
        ERRORS=$((ERRORS + 1))
    fi
else
    warn "Skipping import smoke test (uv not available)."
fi

# ─── 9. Summary ───────────────────────────────────────────────
section "Setup Summary"

if [ "$ERRORS" -eq 0 ]; then
    echo ""
    pass "All checks passed. Yunshu development environment is ready."
    echo ""
    echo "  Quick start:"
    echo "    cd $PROJECT_ROOT"
    echo "    just dev                # Start dev server (port 8000)"
    echo "    just test               # Run test suite"
    echo "    just lint               # Lint code"
    echo "    just bench-roofline     # Run benchmarks"
    echo ""
    echo "  Environment variables:"
    echo "    YUNSHU_MODEL=path/to/model   # Single-model mode"
    echo "    YUNSHU_MULTI_MODEL=1         # Multi-model mode"
    echo "    YUNSHU_AUTH_TOKEN=secret     # Enable auth"
    echo "    YUNSHU_CORS_ORIGINS=...      # CORS origins"
    echo ""
    echo "  Documentation: docs/index.html"
else
    echo ""
    fail "$ERRORS error(s) found. Fix them before proceeding."
    exit 1
fi
