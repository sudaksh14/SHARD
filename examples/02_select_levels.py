#!/usr/bin/env python3
"""
Example B - compare the SHARD levels: parameters, GFLOPs and (optionally) CPU latency per level.

    python examples/02_select_levels.py --checkpoint checkpoints/SHARD_deit_v3_5level.pt
    python examples/02_select_levels.py --latency          # also time each level on this machine (batch 1)

Each level is turned into a regular model with `make_base_copy()`; FLOPs are counted with torch-pruning
(multiply-accumulates, the convention of the paper's GFLOPs column: 4.62 ... 17.59). Higher level = wider embedding
(D = 384 ... 768) = more capacity, more compute, higher accuracy (paper Table 2: 80.39 % ... 83.03 % top-1 on ImageNet-1k;
measuring accuracy needs ImageNet, see scripts/evaluate_imagenet_levels.py).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from utils.switching_harness import common


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--latency", action="store_true")
    args = ap.parse_args()

    cfg = common.paper_config("shard_deit_v3")
    model = common.build_shard_model(cfg, args.checkpoint)
    x = torch.randn(1, 3, 224, 224)
    try:
        import torch_pruning as tp
    except ImportError:
        tp = None
        print("(torch-pruning not installed: GFLOPs skipped)")
    print(f"{'level':>5} {'D':>5} {'params(M)':>10} {'GFLOPs':>8}" + ("  latency(ms)" if args.latency else ""))
    for level in range(model.max_level() + 1):
        model.set_level_use(level)
        reg = model.make_base_copy().eval()
        params = sum(p.numel() for p in reg.parameters()) / 1e6
        gflops = tp.utils.count_ops_and_params(reg, x)[0] / 1e9 if tp else float("nan")
        line = f"{level:>5} {cfg.hidden_dims[level]:>5} {params:>10.2f} {gflops:>8.2f}"
        if args.latency:
            with torch.no_grad():
                for _ in range(3):
                    reg(x)
                t0 = time.perf_counter()
                for _ in range(10):
                    reg(x)
            line += f"  {(time.perf_counter() - t0) * 100:9.1f}"
        print(line)


if __name__ == "__main__":
    main()
