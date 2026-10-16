"""
Per-level latency / power / RAM benchmark for the Jetson experiments (paper Sec. 4.4, 4.5, Table 3, Fig. 4, Fig. 5).

This is the consolidated, documented form of `profile_model_energy()` (PyTorch eager) and `profile_model_onnx()`
(ONNX Runtime). The measurement procedure is:

  per level (1 level = 1 model):
      [model preparation: move_to(level) / build model / ONNX export + session creation]   not timed
      warm-up: `--warmup` inferences (default 20), not recorded
      then `--runs` times (default 100):
          cuda synchronize; t0; inference; cuda synchronize; latency = perf_counter() - t0
          read jtop snapshot: GPU %, mean CPU %, RAM fraction, "Power TOT" (mW)      <- right AFTER the inference
          sleep(`--interval`)  (default 0.1 s)
      cleanup + sleep(`--cooldown`) (default 2 s)

Behaviour to be aware of:
  * Power is a jtop snapshot taken after the timed inference, not an integral over the inference.
    Energy per inference (Table 3) is computed afterwards as  power_mW/1000 * latency_s  (see energy_stats.py).
  * With the ONNX Runtime backend the timed region includes `x.cpu().numpy()` (device->host copy of the input)
    and ORT's own host->device copy (no IO binding).  `--input-copy outside` hoists the copy out of the timed region;
    it is NOT what the published numbers used.
  * ONNX export must run with autograd enabled: under `torch.no_grad()` eval-mode `nn.MultiheadAttention` takes the
    fused fast path (`aten::_native_multi_head_attention`) which torch 2.1 cannot export.
  * "ram" is the fraction RAM-in-use reported by jtop for the whole system, not process memory.

Methods (--method):
  shard        one SHARD model; level i is produced with the Delta Runtime Manager (FileDeltaManager.move_to(i))
  independent  separate regular model per level (randomly initialised, same architecture as the SHARD level)
  original     DeiT-B/16 (timm `deit_base_patch16_224.fb_in1k`, pretrained=False), repeated `--repeats` times
  scala        Scala baseline (external repo BeSpontaneous/Scala-pytorch must be importable as `scala`); eager only

Weights do not influence latency/power, so by default no checkpoint is needed; pass --checkpoint to benchmark real weights.

Without a Jetson (no jtop) the benchmark still runs and records latency, but power/energy are stored as null and
a warning is printed - values are never invented.

Example (on the board, after utils/jetson_harness/scripts/preflight.sh):
    python -m utils.jetson_harness.benchmark_levels --method shard --backend onnxruntime --out jetson_runs/shard_onnx.json
"""
from __future__ import annotations

import argparse
import gc
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import torch

from utils.switching_harness import common

NOT_AVAILABLE = None


# --------------------------------------------------------------------------------------------- sampling
class JtopSampler:
    """Context manager around `jtop` that returns the fields used in the result files."""

    def __init__(self):
        from jtop import jtop  # noqa: F401  (ImportError => caller falls back to NullSampler)
        self._jtop = jtop
        self._ctx = None
        self.jetson = None

    def __enter__(self):
        self._ctx = self._jtop()
        self.jetson = self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)

    def sample(self) -> dict:
        stats = self.jetson.stats
        cpus = [v for k, v in stats.items() if re.fullmatch(r"CPU\d+", k)]
        return {
            "cpu_util": sum(cpus) / len(cpus) if cpus else NOT_AVAILABLE,
            "gpu_util": stats.get("GPU", NOT_AVAILABLE),
            "ram": stats.get("RAM", NOT_AVAILABLE),
            "power_mW": stats.get("Power TOT", NOT_AVAILABLE),
        }


class NullSampler:
    """Used when jtop is unavailable: nothing is measured, nothing is invented."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def sample(self) -> dict:
        return {"cpu_util": NOT_AVAILABLE, "gpu_util": NOT_AVAILABLE, "ram": NOT_AVAILABLE, "power_mW": NOT_AVAILABLE}


def make_sampler(allow_missing: bool):
    try:
        return JtopSampler(), True
    except ImportError:
        if not allow_missing:
            sys.exit("jtop is not installed (pip install jetson-stats, needs sudo/root). "
                     "Use --allow-missing-power to benchmark latency only.")
        print("[warn] jtop not available: power / RAM / utilisation will be recorded as null", file=sys.stderr)
        return NullSampler(), False


# ------------------------------------------------------------------------------------------ model setup
def export_onnx_bytes(model, x, opset: int = 17) -> bytes:
    """In-memory ONNX export.  Do NOT wrap in torch.no_grad() (see module docstring)."""
    buf = io.BytesIO()
    torch.onnx.export(model, x, buf, input_names=["input"], output_names=["output"], opset_version=opset)
    return buf.getvalue()


def ort_session(onnx_bytes: bytes, device: str, providers=None):
    import onnxruntime as ort
    if providers is None:
        providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    missing = [p for p in providers if p not in available]
    if missing:
        sys.exit(f"ONNX Runtime providers {missing} are not available (have: {available}).")
    return ort.InferenceSession(onnx_bytes, providers=providers)


def cleanup(device: str, cooldown: float):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    time.sleep(cooldown)


# ---------------------------------------------------------------------------------------------- profiling
def profile(model, x, args, sampler) -> dict:
    """Profile one model; returns the original per-model metric dict."""
    device = args.device
    sync = common.make_sync(device)
    model = model.eval().to(device)
    x = x.to(device)

    if args.backend == "onnxruntime":
        onnx_bytes = export_onnx_bytes(model, x)
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        session = ort_session(onnx_bytes, device, args.ort_providers)
        input_name = session.get_inputs()[0].name
        x_np = x.cpu().numpy()

        def infer():
            arr = x.cpu().numpy() if args.input_copy == "inside" else x_np
            return session.run(None, {input_name: arr})
    else:
        def infer():
            if args.scala_ratio is not None:
                return model(x, ratio=args.scala_ratio)
            return model(x)

    with torch.no_grad():
        for _ in range(args.warmup):
            infer()
            sync()

    metrics = {"latency_ms": [], "cpu_util": [], "gpu_util": [], "ram": [], "power_mW": []}
    with torch.no_grad(), sampler:
        for _ in range(args.runs):
            sync()
            t0 = time.perf_counter()
            infer()
            sync()
            metrics["latency_ms"].append((time.perf_counter() - t0) * 1000.0)
            for k, v in sampler.sample().items():
                metrics[k].append(v)
            time.sleep(args.interval)

    # field kept for compatibility with the result-file schema: sum(power_mW * interval) / 1000.
    # NOT the per-inference energy of Table 3 - use energy_stats.py for that.
    powers = [p for p in metrics["power_mW"] if p is not None]
    metrics["energy_J"] = sum(p * args.interval for p in powers) / 1000.0 if powers else NOT_AVAILABLE
    return metrics


def summarize_line(label, m) -> str:
    lat = m["latency_ms"]
    mean = sum(lat) / len(lat)
    s = f"{label}: latency {mean:.2f} ms (max {max(lat):.2f})"
    p = [v for v in m["power_mW"] if v is not None]
    if p:
        s += f", power {sum(p) / len(p):.0f} mW"
    return s


# --------------------------------------------------------------------------------------------------- methods
def run_shard(cfg, args, sampler):
    from networks import level_delta_utils as delta
    shard = common.build_shard_model(cfg, args.checkpoint)
    size = cfg.structure.image_size
    x = torch.randn(1, 3, size, size)
    results = {}
    with tempfile.TemporaryDirectory(dir=args.workdir) as tmp:
        path = os.path.join(tmp, "vit.delta")
        with open(path, "wb") as f:
            delta.FileDeltaManager.make_delta_file(f, shard, starting_level=0)
        del shard
        reg_config = cfg.create_base_config(0).no_prebuilt()
        with delta.file_delta_manager(path, reg_config) as manager:
            manager.set_managed_model(manager.managed_model().to(args.device))
            for level in range(manager.max_level() + 1):
                print(f"\n=== SHARD level {level} ===", flush=True)
                model = manager.move_to(level)
                results[f"Level {level}"] = m = profile(model, x, args, sampler)
                print(summarize_line(f"Level {level}", m))
                cleanup(args.device, args.cooldown)
    return results


def run_independent(cfg, args, sampler):
    size = cfg.structure.image_size
    x = torch.randn(1, 3, size, size)
    results = {}
    for level in range(cfg.max_level() + 1):
        print(f"\n=== Independent model, level {level} ===", flush=True)
        torch.manual_seed(level)
        model = cfg.create_base_config(level).no_prebuilt().make_model()
        results[f"Level {level}"] = m = profile(model, x, args, sampler)
        print(summarize_line(f"Level {level}", m))
        del model
        cleanup(args.device, args.cooldown)
    return results


def run_original(args, sampler):
    import timm
    x = torch.randn(1, 3, 224, 224)
    results = {}
    model = timm.create_model("deit_base_patch16_224.fb_in1k", pretrained=False)
    for i in range(args.repeats):
        print(f"\n=== Original DeiT-B, repeat {i} ===", flush=True)
        results[f"Level {i}"] = m = profile(model, x, args, sampler)   # labels follow the result-file schema
        print(summarize_line(f"repeat {i}", m))
        cleanup(args.device, args.cooldown)
    return results


def run_scala(args, sampler):
    if args.backend != "pytorch":
        sys.exit("scala is only supported with --backend pytorch (its ratio argument cannot be exported)")
    try:
        import scala.models_scala  # noqa: F401  registers deit_base_patch16_224_scala with timm
    except ImportError:
        sys.exit("Scala baseline needs https://github.com/BeSpontaneous/Scala-pytorch on PYTHONPATH (module `scala`).")
    from timm.models import create_model
    x = torch.randn(1, 3, 224, 224)
    results = {}
    for ratio in (0.5, 0.625, 0.75, 0.875, 1.0):
        print(f"\n=== Scala ratio {ratio} ===", flush=True)
        model = create_model("deit_base_patch16_224_scala", pretrained=False, num_classes=1000)
        args.scala_ratio = ratio
        results[f"Ratio {ratio}"] = m = profile(model, x, args, sampler)
        print(summarize_line(f"ratio {ratio}", m))
        del model
        cleanup(args.device, args.cooldown)
    return results


# ------------------------------------------------------------------------------------------------------ main
def load_defaults(path):
    if not path:
        return {}
    with open(path) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None, help="JSON with defaults, e.g. utils/jetson_harness/configs/paper_15W_onnx_fp32.json")
    known, _ = pre.parse_known_args(argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[pre])
    ap.add_argument("--method", choices=["shard", "independent", "original", "scala"], default="shard")
    ap.add_argument("--backend", choices=["onnxruntime", "pytorch"], default="onnxruntime")
    ap.add_argument("--arch", choices=common.ARCHS, default="shard_vit",
                    help="level architecture. Default `shard_vit` = the model class used for the reported Jetson measurements; "
                         "`shard_deit_v3` = the released checkpoint's architecture (adds LayerScale)")
    ap.add_argument("--checkpoint", default=None, help="optional state dict (shard method); not needed for latency/power")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--interval", type=float, default=0.1, help="sleep after each timed run / jtop sampling period (s)")
    ap.add_argument("--cooldown", type=float, default=2.0, help="sleep between models (s)")
    ap.add_argument("--repeats", type=int, default=5, help="repetitions for --method original")
    ap.add_argument("--input-copy", choices=["inside", "outside"], default="inside",
                    help="ONNX Runtime only: include x.cpu().numpy() in the timed region (default behaviour)")
    ap.add_argument("--ort-providers", nargs="+", default=None, help="default: CUDAExecutionProvider (cuda) / CPUExecutionProvider")
    ap.add_argument("--workdir", default=None, help="directory for the temporary delta file")
    ap.add_argument("--tiny", action="store_true", help="tiny model, for smoke tests only")
    ap.add_argument("--allow-missing-power", action="store_true", help="run without jtop (latency only; power recorded as null)")
    ap.add_argument("--out", required=True, help="JSON output (same schema as the reported result files files)")
    ap.set_defaults(**load_defaults(known.config))
    args = ap.parse_args(argv)
    args.scala_ratio = None
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda requested but CUDA is not available")
    sampler, power_ok = make_sampler(args.allow_missing_power)

    cfg = common.tiny_config(args.arch) if args.tiny else common.paper_config(args.arch)
    t_start = time.time()
    if args.method == "shard":
        results = run_shard(cfg, args, sampler)
    elif args.method == "independent":
        results = run_independent(cfg, args, sampler)
    elif args.method == "original":
        results = run_original(args, sampler)
    else:
        results = run_scala(args, sampler)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=4)
    meta = {
        "schema": "utils/jetson_harness/benchmark_levels/v1",
        "args": {k: v for k, v in vars(args).items()},
        "power_available": power_ok,
        "wall_time_s": time.time() - t_start,
        "env": common.environment_info(),
        "nvpmodel": _run(["nvpmodel", "-q"]),
        "note": "latency_ms/power_mW lists use the result-file schema; see utils/jetson_harness/README.md for definitions.",
    }
    try:
        import onnxruntime as ort
        meta["onnxruntime"] = ort.__version__
    except ImportError:
        pass
    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {args.out} and {meta_path}")


def _run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return NOT_AVAILABLE


if __name__ == "__main__":
    main()
