#!/usr/bin/env bash
# Run the Sensemaking / EM-HSD v2 demo.
# Requirements: Python 3.10+  (no Docker, no other installs needed)
#
# First run:  ./run.sh          — creates .venv, installs deps, starts server
# Re-runs:    ./run.sh          — reuses existing .venv, starts server
# Custom port: PORT=9000 ./run.sh

set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
VENV="$REPO/.venv"
SPINE_DIR="$REPO/Johnny t0-1.03/src"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

# ── Python version check ────────────────────────────────────────────────────
PY=$(command -v python3 || command -v python || true)
if [[ -z "$PY" ]]; then
  echo "ERROR: python3 not found. Install Python 3.10+ and re-run." >&2
  exit 1
fi
PY_VER=$("$PY" -c 'import sys; print(sys.version_info[:2])')
if "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  true
else
  echo "ERROR: Python 3.10+ required (found $PY_VER)." >&2
  exit 1
fi

# ── Virtual environment ─────────────────────────────────────────────────────
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "==> Creating virtual environment..."
  "$PY" -m venv "$VENV"
fi

PIP="$VENV/bin/pip"
PYTHON="$VENV/bin/python"

# ── Dependencies ────────────────────────────────────────────────────────────
echo "==> Installing / updating dependencies..."
"$PIP" install --quiet --upgrade pip
"$PIP" install --quiet -e "$REPO/123[serve,hf]"

# ── Launch ──────────────────────────────────────────────────────────────────
echo ""
echo "  Sensemaking demo → http://$HOST:$PORT/"
echo "  Press Ctrl-C to stop."
echo ""

# Open browser in background after a short delay (Mac/Linux)
if command -v open &>/dev/null; then
  (sleep 2 && open "http://$HOST:$PORT/") &
elif command -v xdg-open &>/dev/null; then
  (sleep 2 && xdg-open "http://$HOST:$PORT/") &
fi

export EM_HSD_SPINE_PATH="$SPINE_DIR"
export EM_HSD_ALLOW_DOWNLOADS=1
exec "$VENV/bin/em-hsd-serve" --host "$HOST" --port "$PORT"
