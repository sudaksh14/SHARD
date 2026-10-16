"""
Baseline: switching by loading an independent, complete model for the target level from storage.

Protocol:
  * one full (regular, plain) model per level is stored as a pickled `torch.save(model)` file;
  * for every ordered pair (from, to):  synchronize(); t0; `torch.load(path_to, map_location=device)` x `--trials`;
    synchronize(); t1;   reported time = (t1 - t0) / trials.
  The loaded model is only released after the pair has been timed; unloading the previous model is not timed.

The per-level files are written from the same SHARD model (`make_base_copy()` at each level), so they are
functionally identical to what the delta path reconstructs.

Per-level files are saved compactly (every tensor cloned to its own tight storage, 88 / 137 / 196 / 266 / 346 MB for the
5-level ViT-B). `make_base_copy()` itself returns views into the SHARD model's full tensors, which `torch.save` would
expand to the whole storage; `--no-compact` reproduces that older, larger file layout.

    python -m utils.switching_harness full-model --arch shard_deit_v3 --device cuda --trials 10 \
        --model-dir /path/with/space --out switch_full_model.json
"""
from __future__ import annotations

import argparse
import gc
import os
import time

import torch

from utils.switching_harness import common


def compact_(module):
    """Give every parameter/buffer its own, tightly sized storage (clone of the view)."""
    for p in module.parameters():
        p.data = p.data.clone()
    for b in module.buffers():
        b.data = b.data.clone()
    return module


def export_level_models(model, model_dir: str, compact: bool = True):
    os.makedirs(model_dir, exist_ok=True)
    paths = []
    for level in range(model.max_level() + 1):
        model.set_level_use(level)
        path = os.path.join(model_dir, f"level_{level}.pt")
        reg = model.make_base_copy().cpu().eval()
        torch.save(compact_(reg) if compact else reg, path)
        paths.append(path)
    return paths


def _load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False).eval()
    except TypeError:  # torch < 1.13 has no weights_only argument
        return torch.load(path, map_location=device).eval()


def measure(paths, device, trials: int):
    sync = common.make_sync(device)
    n = len(paths)
    total_ns = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            sync()
            start = time.monotonic_ns()
            for _ in range(trials):
                m = _load(paths[j], device)
            sync()
            total_ns[i][j] = time.monotonic_ns() - start
            print(f"  {i} -> {j}: {total_ns[i][j] / 1e6 / trials:9.3f} ms", flush=True)
            del m
            gc.collect()
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
    return total_ns


def main(argv=None):
    from utils.switching_harness.delta_switching import summarize

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=common.ARCHS, default="shard_deit_v3")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--trials", type=int, default=10, help="loads per pair (paper script: 100)")
    ap.add_argument("--model-dir", required=True,
                    help="directory for the per-level model files (~1 GB in total for the 5-level ViT-B)")
    ap.add_argument("--no-compact", dest="compact", action="store_false",
                    help="save per-level files as views of the full tensors (larger files; legacy layout)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = common.tiny_config(args.arch) if args.tiny else common.paper_config(args.arch)
    model = common.build_shard_model(cfg, args.checkpoint)
    paths = export_level_models(model, args.model_dir, args.compact)
    total_ns = measure(paths, args.device, args.trials)
    ms = [[v / 1e6 / args.trials for v in row] for row in total_ns]
    result = {
        "schema": "utils/switching_harness/full_model/v1", "arch": args.arch, "tiny": args.tiny,
        "levels": len(paths), "device": args.device, "trials": args.trials, "compact": args.compact,
        "file_bytes": [os.path.getsize(p) for p in paths],
        "matrix_ms": ms, "matrix_total_ns": total_ns, "summary_ms": summarize(ms),
        "env": common.environment_info(),
    }
    print("summary (ms):", result["summary_ms"])
    if args.out:
        common.dump_json(result, args.out)
    return result


if __name__ == "__main__":
    main()
