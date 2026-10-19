# Jetson harness

Minimal tooling to repeat the paper's Jetson Orin experiments (Sec. 4.4-4.6, Fig. 4-5, Table 3). It imports the canonical model/DRM code
from the repository root and adds only: a preflight report, the per-level latency/power benchmark, the energy computation, and shell wrappers.
The switching measurements live in [`../switching_harness`](../switching_harness) and are wrapped by `scripts/run_switching_benchmark.sh`.

## Reference environment of the published results

| | |
|---|---|
| Board | NVIDIA Jetson Orin, 8 GB (6 Cortex-A78AE cores, 7451 MB RAM in tegrastats, 1510 MHz CPU). The exact module (Orin Nano / NX 8 GB) is **not recorded** in the artifacts; run `preflight.sh` and keep its JSON |
| Power mode | `NV Power Mode: 15W` (nvpmodel) |
| L4T | R35 revision 5.0 (`/etc/nv_tegra_release`), kernel 5.10.192-tegra; JetPack 5.1.3 by NVIDIA's numbering (inferred) |
| Python / PyTorch | 3.8.10 / 2.1.0a0+41361538.nv23.06 (NVIDIA wheel), CUDA 11.4, compute capability 8.7 |
| Runtime | ONNX Runtime 1.15.1 GPU build, `CUDAExecutionProvider` only, fp32, batch 1; TensorRT 8.5.2.2 present but **not** used |
| Power sensor | `jtop` (jetson-stats) field `Power TOT` (mW), RAM/GPU/CPU utilisation from the same snapshot |
| Other stacks | other Jetson modules and JetPack versions are not covered by the reference numbers |

The core packages use `from __future__ import annotations` and import on Python 3.8. PyTorch, torchvision and ONNX Runtime must be NVIDIA's JetPack builds (see `requirements.txt`).
No checkpoint is required for latency/power/switching (they do not depend on weight values); `--checkpoint` is optional.

## Setup on the board

```bash
git clone <repo> && cd <repo>
# PyTorch/torchvision/onnxruntime-gpu: install NVIDIA's JetPack wheels first (not on PyPI)
python3 -m pip install -r requirements.txt   # the Jetson-specific lines are commented in the file
sudo pip3 install jetson-stats && sudo systemctl restart jtop.service     # jtop needs its service; log out/in for group permissions
sudo nvpmodel -m <ID_OF_15W_MODE>        # `sudo nvpmodel -q --verbose` lists the modes of your module; the paper used 15W
sudo jetson_clocks                       # optional; the published protocol report does not show it enabled
```

Permissions: `jtop` needs its daemon; `echo 3 > /proc/sys/vm/drop_caches` (cold-cache switching) needs root; `nvpmodel`/`jetson_clocks` need sudo.
The scripts never change these settings.

## Commands

```bash
utils/jetson_harness/scripts/preflight.sh jetson_runs/run1/preflight.json     # 0 = OK, 1 = mandatory item FAIL, 2 = not a Jetson
utils/jetson_harness/scripts/run_energy_benchmark.sh jetson_runs/run1         # shard (ORT fp32), shard_eager, independent, original
utils/jetson_harness/scripts/run_switching_benchmark.sh jetson_runs/run1      # verify + delta (warm/cold cache) + full-model baseline
utils/jetson_harness/scripts/collect_results.sh jetson_runs/run1              # SUMMARY.md + energy.json
# or everything: examples/04_jetson_evaluation.sh
```

Direct use: `python -m utils.jetson_harness.benchmark_levels --config utils/jetson_harness/configs/paper_15W_onnx_fp32.json --method shard --out OUT.json`
(see `--help`). Useful environment variables for the scripts: `PYTHON`, `ARCH` (`shard_vit` default = architecture of the reported measurements,
`shard_deit_v3` = released model, which adds LayerScale), `CHECKPOINT`, `EXTRA_ARGS`, `WORKDIR` (where the delta file is written - use the storage you want to measure), `ITERS`.

Fewer runs for a quick check: `EXTRA_ARGS="--runs 20 --warmup 5"`. Without jtop: `EXTRA_ARGS="--allow-missing-power"` (latency only; power is stored as `null`, never invented).

## What is measured, and what is not

| Quantity | Definition |
|---|---|
| steady-state inference latency | `perf_counter` around one inference with `cuda.synchronize()` before/after; 20 warm-up inferences (not recorded), then 100 runs |
| with ONNX Runtime | the timed region contains `x.cpu().numpy()` and ORT's host->device copy (`--input-copy outside` hoists the copy, not what the published numbers used) |
| power | jtop `Power TOT` snapshot read right after each timed run; 0.1 s sleep between runs (`--interval`) |
| energy per inference [mJ] | `power_mW/1000 * latency_ms` of the same run, mean ± std over the runs (`energy_stats.py`); the JSON field `energy_J` is the *legacy* `sum(power*interval)/1000` and is not the Table-3 quantity |
| RAM | jtop `RAM` fraction (whole system, not per process) |
| model preparation (not timed) | `FileDeltaManager.move_to(level)`, ONNX export (opset 17, in memory, autograd enabled) and ORT session creation |
| switching latency | `switching_harness delta`: time of `manager.move_to(to)` after `move_to(from)`, mean of `--iters` per ordered pair (default 10; the paper results use 100), fresh manager per pair. Includes delta read + deserialise + apply (+ GPU sync). Excludes manager construction (base-model load) |
| baseline switching | `switching_harness full-model`: mean time of `torch.load(level file, map_location=cuda)` |
| not measured | TensorRT, fp16/int8, batch sizes > 1, thermal/throttling behaviour, concurrent workloads |

Do not mix: **initialisation** (model creation, base load), **switching** (delta I/O + apply), **steady-state inference** (above) and any end-to-end
sum - each is reported separately.

## Output files

* `*.json` (benchmark): `{ "Level k": {latency_ms[], cpu_util[], gpu_util[], ram[], power_mW[], energy_J}, ... }` .
* `*.meta.json`: arguments, power sensor availability, nvpmodel output, environment fingerprint.
* `switch_*.json`: `matrix_ms` (row = from, column = to), `matrix_total_ns`, `summary_ms` (up/down/all means), delta file size.
* `preflight.json`, `SUMMARY.md`, `energy.json`.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `jtop is not installed` / preflight FAIL on jtop | `sudo pip3 install jetson-stats`, restart the jtop service, re-login |
| ONNX export error `aten::_native_multi_head_attention ... not supported` | export was done under `torch.no_grad()`; `benchmark_levels.py` exports with autograd enabled on purpose |
| ORT provider missing | the PyPI `onnxruntime` is CPU-only; install the JetPack GPU wheel matching Python 3.8 |
| `--no-cache` fails | needs Linux `os.posix_fadvise`; also drop caches as root for fully cold runs |
| first level is slower / noisy | let the board idle, pin clocks, check `preflight` for throttling; the paper's own protocol notes CV < 5 % as a re-run criterion |
| `CUDA out of memory` while exporting | close other GPU jobs; the 8 GB board is shared CPU/GPU memory (the benchmark frees models between levels and sleeps `--cooldown`) |
 