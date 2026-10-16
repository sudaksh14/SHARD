"""
Level-to-level switching latency with the file-based Delta Runtime Manager.

Protocol:
  1. Build the SHARD model (random init or --checkpoint) and write a delta file with
     `FileDeltaManager.make_delta_file(file, model, starting_level=0)`.  (Offline step, not timed.)
  2. For every ordered pair (from, to): open a FRESH `FileDeltaManager` on that file, move the managed
     base model to the target device, then repeat `--iters` times:
         manager.move_to(from)                    # untimed, resets the state
         synchronize(); t0
         manager.move_to(to)                      # timed: read chunk(s) + deserialize + apply deltas
         synchronize(); t1
  3. Report the mean time per switch.  Multi-level jumps apply the intermediate deltas one after another.

Times are reported in milliseconds computed from the raw nanosecond totals (`matrix_total_ns` is stored as well).

    python -m utils.switching_harness delta --arch shard_deit_v3 --device cuda --iters 10 --out switch_delta.json
    python -m utils.switching_harness delta --no-cache ...      # evict chunks from the page cache after each read (Linux)
"""
from __future__ import annotations

import argparse
import gc
import os
import tempfile
import time

import torch

from networks import level_delta_utils as delta
from utils.switching_harness import common


def time_manager(manager, level_from: int, level_to: int, iters: int, sync) -> int:
    """Total nanoseconds spent in `iters` timed switches level_from -> level_to."""
    total_ns = 0
    for _ in range(iters):
        manager.move_to(level_from)
        sync()
        start = time.monotonic_ns()
        manager.move_to(level_to)
        sync()
        total_ns += time.monotonic_ns() - start
    return total_ns


def write_delta_file(model, path: str, starting_level: int = 0) -> int:
    with open(path, "wb") as f:
        delta.FileDeltaManager.make_delta_file(f, model, starting_level=starting_level)
    return os.path.getsize(path)


def measure(cfg, model, device, iters: int, no_cache: bool, shared_manager: bool, workdir: str):
    sync = common.make_sync(device)
    n = model.max_level() + 1
    delta_path = os.path.join(workdir, "vit.delta")
    size = write_delta_file(model, delta_path, starting_level=0)
    reg_config = cfg.create_base_config(0).no_prebuilt()
    if no_cache:
        from utils.switching_harness.nocache import file_delta_manager_nocache as opener
    else:
        opener = delta.file_delta_manager

    total_ns = [[0] * n for _ in range(n)]
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

    def run_pair(manager, i, j):
        total_ns[i][j] = time_manager(manager, i, j, iters, sync)
        print(f"  {i} -> {j}: {total_ns[i][j] / 1e6 / iters:8.3f} ms", flush=True)

    if shared_manager:
        with opener(delta_path, reg_config) as manager:
            manager.set_managed_model(manager.managed_model().to(device))
            for i, j in pairs:
                run_pair(manager, i, j)
    else:
        for i, j in pairs:
            with opener(delta_path, reg_config) as manager:
                manager.set_managed_model(manager.managed_model().to(device))
                run_pair(manager, i, j)
            del manager
            gc.collect()
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
    return total_ns, size


def summarize(matrix):
    """Mean over up-switches (to > from), down-switches (to < from) and both; same matrix orientation as the paper."""
    n = len(matrix)
    up = [matrix[i][j] for i in range(n) for j in range(n) if j > i]
    down = [matrix[i][j] for i in range(n) for j in range(n) if j < i]

    def mean(v):
        return sum(v) / len(v)
    return {"mean_up_ms": mean(up), "mean_down_ms": mean(down), "mean_all_ms": mean(up + down)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=common.ARCHS, default="shard_deit_v3")
    ap.add_argument("--checkpoint", default=None,
                    help="optional state dict; switching latency does not depend on the weight values")
    ap.add_argument("--tiny", action="store_true", help="tiny model for smoke tests")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iters", type=int, default=10, help="timed switches per (from,to) pair (paper script: 100)")
    ap.add_argument("--no-cache", action="store_true",
                    help="evict delta chunks from the OS page cache after reading (Linux)")
    ap.add_argument("--shared-manager", action="store_true",
                    help="reuse one manager for all pairs (SHARD repo variant) instead of a fresh one per pair (Jetson variant)")
    ap.add_argument("--workdir", default=None, help="where to write the temporary delta file (default: temp dir)")
    ap.add_argument("--out", default=None, help="write JSON results here")
    args = ap.parse_args(argv)
    if args.no_cache and not hasattr(os, "posix_fadvise"):
        ap.error("--no-cache needs os.posix_fadvise (Linux only)")

    cfg = common.tiny_config(args.arch) if args.tiny else common.paper_config(args.arch)
    model = common.build_shard_model(cfg, args.checkpoint, device="cpu")
    size = cfg.structure.image_size
    with torch.no_grad():  # warm-up forward
        model.to(args.device)(torch.randn(1, 3, size, size, device=args.device))
    model.cpu()

    with tempfile.TemporaryDirectory(dir=args.workdir) as tmp:
        total_ns, file_bytes = measure(cfg, model, args.device, args.iters, args.no_cache, args.shared_manager, tmp)

    ms = [[v / 1e6 / args.iters for v in row] for row in total_ns]
    result = {
        "schema": "utils/switching_harness/delta/v1",
        "arch": args.arch, "tiny": args.tiny, "checkpoint": args.checkpoint,
        "levels": len(total_ns), "device": args.device, "iters": args.iters,
        "no_cache": args.no_cache, "shared_manager": args.shared_manager,
        "delta_file_bytes": file_bytes,
        "matrix_ms": ms, "matrix_total_ns": total_ns,
        "summary_ms": summarize(ms),
        "env": common.environment_info(),
    }
    print("summary (ms):", result["summary_ms"])
    if args.out:
        common.dump_json(result, args.out)
    return result


if __name__ == "__main__":
    main()
