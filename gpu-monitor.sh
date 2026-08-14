#!/usr/bin/env bash
# Wrapper for gpu_monitor.py (real-time Intel Arc B70 utilization monitor).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$ROOT/.venv/bin/python" "$ROOT/gpu_monitor.py" "$@"
