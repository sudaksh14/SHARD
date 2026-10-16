#!/usr/bin/env bash
# Example D - Jetson evaluation: preflight, energy/latency benchmark, switching benchmark, summary.
# REQUIRES a Jetson (JetPack 5 / Python 3.8 stack, see utils/jetson_harness/README.md).
# Usage: examples/04_jetson_evaluation.sh [OUT_DIR]      (default: jetson_runs/<timestamp>)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/jetson_runs/jetson_run_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
"$ROOT/utils/jetson_harness/scripts/preflight.sh" "$OUT/preflight.json" || { echo "preflight failed - fix the FAIL items first" >&2; exit 1; }
"$ROOT/utils/jetson_harness/scripts/run_energy_benchmark.sh" "$OUT"
"$ROOT/utils/jetson_harness/scripts/run_switching_benchmark.sh" "$OUT"
"$ROOT/utils/jetson_harness/scripts/collect_results.sh" "$OUT"
