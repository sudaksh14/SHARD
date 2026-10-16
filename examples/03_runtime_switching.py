#!/usr/bin/env python3
"""
Example C - runtime switching with the Delta Runtime Manager (real implementation: networks/level_delta_utils.py).

A regular (plain) model sits at one level; `move_to(level)` loads/applies the parameter deltas to move it to
another level, in place. The output is checked against the SHARD model at that level.

    python examples/03_runtime_switching.py --tiny
    python examples/03_runtime_switching.py --checkpoint checkpoints/SHARD_deit_v3_5level.pt --start 0 --targets 4 2 0 3

Two managers are shown:
  InMemoryDeltaManager  all deltas held in RAM
  FileDeltaManager      deltas read from a delta file on disk (what is used on the Jetson)
"""
import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from networks import level_delta_utils as delta
from utils.switching_harness import common


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--targets", type=int, nargs="+", default=[2, 0, 1])
    args = ap.parse_args()

    cfg = common.tiny_config("shard_deit_v3") if args.tiny else common.paper_config("shard_deit_v3")
    shard = common.build_shard_model(cfg, args.checkpoint)
    size = cfg.structure.image_size
    x = torch.randn(1, 3, size, size)

    def check(manager, name):
        for level in args.targets:
            t0 = time.perf_counter()
            reg = manager.move_to(level)
            dt = (time.perf_counter() - t0) * 1e3
            shard.set_level_use(level)
            with torch.no_grad():
                ok = torch.allclose(reg.eval()(x), shard(x), rtol=1e-3, atol=1e-5)
            print(f"[{name}] -> level {level}: {dt:8.1f} ms, matches SHARD model: {ok}")

    # 1) all deltas in memory; the managed model starts at --start
    shard.set_level_use(args.start)
    check(delta.InMemoryDeltaManager(shard, args.start), "in-memory")

    # 2) delta file on disk (written once, offline) + file-backed manager
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.delta")
        with open(path, "wb") as f:
            delta.FileDeltaManager.make_delta_file(f, shard, starting_level=args.start)
        print(f"delta file: {os.path.getsize(path) / 1e6:.1f} MB")
        base_cfg = cfg.create_base_config(args.start).no_prebuilt()   # config of the regular model at the start level
        with delta.file_delta_manager(path, base_cfg) as manager:
            check(manager, "file")


if __name__ == "__main__":
    main()
