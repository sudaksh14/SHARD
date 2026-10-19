#!/usr/bin/env bash
# Summarise a benchmark output directory (from run_energy_benchmark.sh / run_switching_benchmark.sh) into SUMMARY.md and energy.json.
#
# Usage:  utils/jetson_harness/scripts/collect_results.sh OUT_DIR
# Env:    PYTHON   python interpreter (default: python3)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
OUT="${1:?usage: $0 OUT_DIR}"
[[ -d "$OUT" ]] || { echo "error: $OUT is not a directory" >&2; exit 1; }
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
shopt -s nullglob
bench=(); for f in "$OUT"/*.json; do case "$f" in *.meta.json|*switch*|*/energy.json) ;; *) bench+=("$f");; esac; done
{
  echo "# Results summary for $OUT"
  echo; echo '## Energy per inference [mJ], mean ± std (power snapshot × latency)'; echo '```'
  if (( ${#bench[@]} )); then "$PYTHON" -m utils.jetson_harness.energy_stats --table --json "$OUT/energy.json" "${bench[@]}"; else echo "no benchmark JSON files"; fi
  echo '```'
  echo; echo '## Switching latency [ms]'; echo '```'
  "$PYTHON" - "$OUT" <<'PY'
import glob, json, sys
from switching_harness.delta_switching import summarize
for p in sorted(glob.glob(sys.argv[1] + "/switch_*.json")):
    d = json.load(open(p)); s = summarize(d["matrix_ms"])
    print(f"{p}: up {s['mean_up_ms']:.2f}  down {s['mean_down_ms']:.2f}  all {s['mean_all_ms']:.2f}")
PY
  echo '```'
} | tee "$OUT/SUMMARY.md"
