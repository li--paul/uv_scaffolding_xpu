#!/usr/bin/env bash
# Launch the LTX-2.5 generation web server (FastAPI, port 8090).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load the Intel oneAPI toolchain if present (needed for the XPU runtime).
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    set +u
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force
    set -u
fi

PORT="${LTX_SERVER_PORT:-8090}"
exec "$ROOT/.venv/bin/python" "$ROOT/ltx_server.py" --port "$PORT" "$@"
