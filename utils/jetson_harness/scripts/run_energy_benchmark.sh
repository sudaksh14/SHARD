#!/usr/bin/env bash
# Latency / power / RAM benchmark of all SHARD levels (+ optional baselines) with the paper's settings, then energy table.
#
# Usage:  utils/jetson_harness/scripts/run_energy_benchmark.sh OUT_DIR [METHODS]
#           OUT_DIR   where the JSON files are written (created if needed)
#           METHODS   space separated subset of: shard shard_eager independent original   (default: "shard shard_eager independent original")
# Env:    PYTHON       python interpreter (default: python3)
#         ARCH         shard_vit (default, as in the reported measurements) | shard_deit_v3 (released checkpoint architecture)
#         CHECKPOINT   optional SHARD state dict (does not change latency/power; only needed to benchmark real weights)
#         EXTRA_ARGS   extra arguments passed to every benchmark_levels invocation (e.g. "--runs 50")
# Notes:  * run preflight.sh first; pin the power mode yourself (sudo nvpmodel -m <id>; the paper used 15W).
#         * `shard` = ONNX Runtime fp32 (the paper's SHARD rows); `shard_eager`, `independent`, `original` = PyTorch eager
#           (how the paper's baseline rows were measured, see utils/jetson_harness/README.md).
#         * needs jtop (pip install jetson-stats); otherwise add EXTRA_ARGS="--allow-missing-power" (latency only).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
OUT="${1:?usage: $0 OUT_DIR [METHODS]}"; shift || true
METHODS="${*:-shard shard_eager independent original}"
ARCH="${ARCH:-shard_vit}"
CKPT_ARG=(); [[ -n "${CHECKPOINT:-}" ]] && CKPT_ARG=(--checkpoint "$CHECKPOINT")
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
cfg_onnx=utils/jetson_harness/configs/paper_15W_onnx_fp32.json
cfg_eager=utils/jetson_harness/configs/paper_15W_pytorch_eager.json
for m in $METHODS; do
  echo "################ $m"
  case "$m" in
    shard)        "$PYTHON" -m utils.jetson_harness.benchmark_levels --config "$cfg_onnx"  --method shard       --arch "$ARCH" "${CKPT_ARG[@]}" --out "$OUT/shard_onnx.json" ${EXTRA_ARGS:-} ;;
    shard_eager)  "$PYTHON" -m utils.jetson_harness.benchmark_levels --config "$cfg_eager" --method shard       --arch "$ARCH" "${CKPT_ARG[@]}" --out "$OUT/shard_eager.json" ${EXTRA_ARGS:-} ;;
    independent)  "$PYTHON" -m utils.jetson_harness.benchmark_levels --config "$cfg_eager" --method independent --arch "$ARCH" --out "$OUT/independent_eager.json" ${EXTRA_ARGS:-} ;;
    original)     "$PYTHON" -m utils.jetson_harness.benchmark_levels --config "$cfg_eager" --method original     --out "$OUT/original_eager.json" ${EXTRA_ARGS:-} ;;
    *) echo "unknown method '$m'" >&2; exit 2 ;;
  esac
done
echo "################ energy per inference (mJ)"
"$PYTHON" -m utils.jetson_harness.energy_stats --table "$OUT"/*.json 2>/dev/null | grep -v '\.meta\.json' || true
