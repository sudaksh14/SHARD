#!/usr/bin/env bash
# Level-switching latency: correctness check, delta switching (warm and cold page cache) and the full-model-load baseline.
#
# Usage:  utils/jetson_harness/scripts/run_switching_benchmark.sh OUT_DIR
# Env:    PYTHON      python interpreter (default: python3)
#         ARCH        shard_vit (default) | shard_deit_v3
#         ITERS       timed switches/loads per (from,to) pair (default 10; the paper results use ITERS=100)
#         MODEL_DIR   where the per-level full-model files for the baseline go (default: OUT_DIR/level_models, ~1 GB)
#         WORKDIR     where the temporary delta file is written (default: system temp dir; use the storage you want to measure)
#         DEVICE      cuda (default) | cpu      EXTRA_ARGS  extra args for the delta/full-model commands (e.g. "--tiny")
#         SKIP_FULL=1 skip the full-model baseline
# Notes:  the cold ("no-cache") run uses posix_fadvise(DONTNEED) after every chunk read (Linux); additionally you may drop the
#         whole page cache beforehand with:  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
#         
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python3}"
OUT="${1:?usage: $0 OUT_DIR}"
# ITERS=100 reproduces the paper switching results; the default is a quicker check
ARCH="${ARCH:-shard_vit}"; ITERS="${ITERS:-10}"; DEVICE="${DEVICE:-cuda}"
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
WD=(); [[ -n "${WORKDIR:-}" ]] && WD=(--workdir "$WORKDIR")
echo "################ correctness (CPU, random weights)"
"$PYTHON" -m utils.switching_harness verify --arch "$ARCH"
echo "################ delta switching, warm page cache"
"$PYTHON" -m utils.switching_harness delta --arch "$ARCH" --device "$DEVICE" --iters "$ITERS" "${WD[@]}" ${EXTRA_ARGS:-} --out "$OUT/switch_delta.json"
echo "################ delta switching, cold page cache (posix_fadvise DONTNEED)"
"$PYTHON" -m utils.switching_harness delta --arch "$ARCH" --device "$DEVICE" --iters "$ITERS" --no-cache "${WD[@]}" ${EXTRA_ARGS:-} --out "$OUT/switch_delta_nocache.json"
if [[ "${SKIP_FULL:-0}" != "1" ]]; then
  echo "################ baseline: independent full models"
  "$PYTHON" -m utils.switching_harness full-model --arch "$ARCH" --device "$DEVICE" --trials "$ITERS" \
      --model-dir "${MODEL_DIR:-$OUT/level_models}" ${EXTRA_ARGS:-} --out "$OUT/switch_full_model.json"
fi
echo "done."
