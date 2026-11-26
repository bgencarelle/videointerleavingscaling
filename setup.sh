#!/usr/bin/env bash
# One-time setup on an Apple Silicon Mac: Homebrew Python + venv + native libwebp.
set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "note: faceloop is made for Apple Silicon Macs; continuing anyway"
fi
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is needed: https://brew.sh" >&2
  exit 1
fi

PYFORMULA="python@3.13"
brew list "$PYFORMULA" >/dev/null 2>&1 || brew install "$PYFORMULA"
brew list webp >/dev/null 2>&1 || brew install webp

PY="$(brew --prefix "$PYFORMULA")/bin/python3.13"
"$PY" - <<'PYCHECK'
import platform, sys
if platform.system() == "Darwin" and platform.machine() != "arm64":
    sys.exit("this Python is not native arm64 (Rosetta). Check your Homebrew install.")
PYCHECK

[[ -d .venv ]] || "$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo
echo "Setup done. Next:"
echo "  .venv/bin/python -m faceloop models fetch --all"
echo "  .venv/bin/python -m faceloop selftest"
