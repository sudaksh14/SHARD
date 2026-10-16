"""
Preflight / measurement-protocol report for Jetson benchmarks.

Non-interactive system report that *reports* the state that
determines whether benchmark numbers are comparable (board, JetPack/L4T, power mode, clocks, software stack, power
sensor, page cache) and never changes system settings.  It works on non-Jetson machines too (every Jetson-specific item
is then reported as "not available") so the script itself can be smoke-tested anywhere.

    python -m utils.jetson_harness.preflight                    # human readable report
    python -m utils.jetson_harness.preflight --json report.json # + machine readable copy
    python -m utils.jetson_harness.preflight --require-jetson   # exit 2 when not running on a Jetson

Exit status: 0 = all mandatory checks passed, 1 = a mandatory check failed (see "FAIL"), 2 = not a Jetson with --require-jetson.
Mandatory (for the paper protocol): CUDA-enabled PyTorch, ONNX Runtime with CUDAExecutionProvider, jtop with a power reading.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import shutil
import subprocess
import sys

PAPER_PROTOCOL = {
    "board": "NVIDIA Jetson Orin, 8 GB (exact module variant not recorded in the reported results)",
    "power_mode": "15W (nvpmodel)",
    "backend": "ONNX Runtime 1.15.1, CUDAExecutionProvider, fp32, batch size 1, no TensorRT",
    "warmup_runs": 20, "timed_runs": 100,
}


def _run(cmd, timeout=10):
    try:
        return subprocess.check_output(cmd, shell=isinstance(cmd, str), stderr=subprocess.DEVNULL, text=True, timeout=timeout).strip()
    except Exception:
        return None


def _read(path):
    try:
        with open(path, "rb") as f:
            return f.read().decode(errors="replace").strip("\x00\n ")
    except OSError:
        return None


def collect() -> dict:
    r = {"checks": []}

    def check(name, ok, detail, mandatory=False):
        r["checks"].append({"name": name, "ok": bool(ok), "detail": detail, "mandatory": mandatory})

    # --- platform
    model = _read("/proc/device-tree/model")
    tegra = _read("/etc/nv_tegra_release")
    r["platform"] = {"machine": platform.machine(), "system": platform.platform(), "python": sys.version.split()[0],
                     "device_tree_model": model, "nv_tegra_release": tegra}
    r["is_jetson"] = bool(tegra or (model and "NVIDIA" in model))
    check("running on a Jetson", r["is_jetson"], model or "no /proc/device-tree/model, no /etc/nv_tegra_release")

    # --- power mode and clocks
    r["nvpmodel"] = _run("nvpmodel -q 2>/dev/null")
    check("power mode readable (nvpmodel -q)", r["nvpmodel"] is not None,
          (r["nvpmodel"] or "nvpmodel not found").replace("\n", " | "))
    r["jetson_clocks"] = _run("jetson_clocks --show 2>/dev/null")  # may need sudo; informational only
    cpus = {}
    for path in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq")):
        cpus[os.path.basename(os.path.dirname(path))] = {
            "cur_khz": _read(path + "/scaling_cur_freq"), "min_khz": _read(path + "/scaling_min_freq"),
            "max_khz": _read(path + "/scaling_max_freq"), "governor": _read(path + "/scaling_governor")}
    r["cpu_clocks"] = cpus
    gpu = {}
    for path in glob.glob("/sys/devices/platform/*gpu*/devfreq/*") + glob.glob("/sys/devices/gpu.0/devfreq/*"):
        gpu[path] = {k: _read(os.path.join(path, k)) for k in ("cur_freq", "min_freq", "max_freq", "governor")}
    r["gpu_clocks"] = gpu

    # --- software stack
    soft = {}
    try:
        import torch
        soft["torch"] = torch.__version__
        soft["cuda_runtime"] = torch.version.cuda
        soft["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            soft["gpu"] = torch.cuda.get_device_name(0)
            soft["compute_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
        check("PyTorch with CUDA", torch.cuda.is_available(), f"torch {torch.__version__}, cuda {torch.version.cuda}", True)
    except ImportError:
        check("PyTorch with CUDA", False, "torch not installed", True)
    for mod in ("torchvision", "timm", "numpy", "onnx", "psutil"):
        try:
            soft[mod] = __import__(mod).__version__
        except ImportError:
            soft[mod] = None
    try:
        import onnxruntime as ort
        soft["onnxruntime"] = ort.__version__
        soft["ort_providers"] = ort.get_available_providers()
        check("ONNX Runtime CUDAExecutionProvider", "CUDAExecutionProvider" in soft["ort_providers"],
              f"onnxruntime {ort.__version__}, providers {soft['ort_providers']}", True)
    except ImportError:
        soft["onnxruntime"] = None
        check("ONNX Runtime CUDAExecutionProvider", False, "onnxruntime not installed", True)
    soft["tensorrt"] = None
    try:
        import tensorrt
        soft["tensorrt"] = tensorrt.__version__
    except ImportError:
        pass
    r["software"] = soft

    # --- power sensor (jtop)
    try:
        from jtop import jtop
        with jtop() as jetson:
            stats = jetson.stats
            power = stats.get("Power TOT")
        r["power_sensor"] = {"source": "jtop", "Power TOT_mW": power}
        check("jtop power reading ('Power TOT')", power is not None, f"{power} mW", True)
    except ImportError:
        r["power_sensor"] = None
        check("jtop power reading ('Power TOT')", False, "jtop not installed (pip install jetson-stats)", True)
    except Exception as e:  # jtop service not running, no permission, ...
        r["power_sensor"] = None
        check("jtop power reading ('Power TOT')", False, f"jtop failed: {e}", True)

    # --- memory / cache / disk
    mem = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        k, _, v = line.partition(":")
        if k in ("MemTotal", "MemAvailable", "Cached", "SwapTotal", "SwapFree"):
            mem[k] = v.strip()
    r["memory"] = mem
    r["page_cache_drop_writable"] = os.access("/proc/sys/vm/drop_caches", os.W_OK)
    check("can drop page cache (needed for cold switching runs)", r["page_cache_drop_writable"],
          "run as root or: sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'")
    total, used, free = shutil.disk_usage(os.getcwd())
    r["disk_free_gb"] = round(free / 1e9, 1)
    check("free disk >= 3 GB (delta file + per-level models)", free > 3e9, f"{free / 1e9:.1f} GB free in {os.getcwd()}")
    r["paper_protocol"] = PAPER_PROTOCOL
    return r


def print_report(r):
    print("=" * 70)
    print("SHARD Jetson harness preflight")
    print("=" * 70)
    p = r["platform"]
    print(f"machine           : {p['machine']}  ({p['system']})")
    print(f"python            : {p['python']}")
    print(f"device-tree model : {p['device_tree_model']}")
    print(f"L4T release       : {p['nv_tegra_release']}")
    print(f"nvpmodel          : {(r['nvpmodel'] or 'n/a').replace(chr(10), ' | ')}")
    if r["cpu_clocks"]:
        c0 = next(iter(r["cpu_clocks"].values()))
        print(f"cpu0 clocks (kHz) : cur={c0['cur_khz']} min={c0['min_khz']} max={c0['max_khz']} governor={c0['governor']}")
    for k, v in r["software"].items():
        print(f"{k:18s}: {v}")
    print("-" * 70)
    for c in r["checks"]:
        tag = "PASS" if c["ok"] else ("FAIL" if c["mandatory"] else "warn")
        print(f"[{tag}] {c['name']}: {c['detail']}")
    print("-" * 70)
    print("Paper protocol for comparison:", json.dumps(PAPER_PROTOCOL))
    print("For comparable numbers also: pin the power mode (sudo nvpmodel -m <id>), optionally sudo jetson_clocks, let the board")
    print("cool down, and stop other GPU processes. This script does not change any setting.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None)
    ap.add_argument("--require-jetson", action="store_true")
    args = ap.parse_args(argv)
    r = collect()
    print_report(r)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(r, f, indent=2)
        print(f"wrote {args.json}")
    if args.require_jetson and not r["is_jetson"]:
        return 2
    return 1 if any(c["mandatory"] and not c["ok"] for c in r["checks"]) else 0


if __name__ == "__main__":
    sys.exit(main())
