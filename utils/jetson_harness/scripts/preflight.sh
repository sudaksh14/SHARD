#!/usr/bin/env bash
# Report the state of a Jetson (power mode, clocks, software stack, power sensor) before benchmarking.
# Changes nothing on the system.
#
# Usage:  utils/jetson_harness/scripts/preflight.sh [OUT_JSON]
# Env:    PYTHON   python interpreter to use (default: python3)
# Exit:   0 all mandatory checks passed, 1 a mandatory check failed, 2 not a Jetson
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
command -v "$PYTHON" >/dev/null || { echo "error: $PYTHON not found (set PYTHON=...)" >&2; exit 127; }
cd "$ROOT"
args=(--require-jetson)
[[ $# -ge 1 ]] && args+=(--json "$1")
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" exec "$PYTHON" -m utils.jetson_harness.preflight "${args[@]}"
