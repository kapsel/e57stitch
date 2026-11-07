#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required (https://docs.astral.sh/uv/) — or install deps manually with pip."
  exit 1
fi

if [ ! -d .venv ]; then
  echo "→ creating virtualenv (Python 3.12)"
  uv venv --python 3.12 .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import e57stitch, fastapi, open3d, pye57" >/dev/null 2>&1; then
  echo "→ installing dependencies"
  uv pip install -e .
fi

if ! python -c "import open3d" >/dev/null 2>&1; then
  echo
  echo "Open3D failed to load. On macOS this usually means libusb is missing:"
  echo "    brew install libusb"
  echo
  exit 1
fi

# Keep a Mac awake for overnight runs; harmless if caffeinate is unavailable.
if command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -imsu python -m e57stitch serve "$@"
fi
exec python -m e57stitch serve "$@"
