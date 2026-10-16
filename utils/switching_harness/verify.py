"""
Correctness check for runtime switching (CPU by default, random weights).

For every supported switching sequence the delta-managed *regular* model must produce the same output as the
SHARD model run at the same level:
  * in-memory deltas   (`InMemoryDeltaManager`)
  * file-based deltas  (`FileDeltaManager`, delta file written to an in-memory buffer)
Sequences: top -> bottom one step at a time, bottom -> top one step at a time, jumps (0 -> max, max -> 0, ...)
and a fixed pseudo-random walk, so multi-level transitions are exercised too.

    python -m utils.switching_harness verify                 # tiny config, a few seconds on CPU
    python -m utils.switching_harness verify --paper-size    # full 5-level ViT-B (slow on CPU, needs a few GB RAM)

The exit status is 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import io
import random
import sys

import torch

from networks import level_delta_utils as levels
from utils.switching_harness import common


def _close(a, b):
    return torch.isclose(a, b, rtol=1e-3, atol=1e-5).all().item()


def _randomize(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        p.data = torch.rand(p.shape, generator=g).to(p.data) / 100
    return model


def sequences(max_level):
    down = list(range(max_level - 1, -1, -1))
    up = list(range(1, max_level + 1))
    jumps = [0, max_level, 0, max_level // 2, max_level, max_level // 2]
    rng = random.Random(0)
    walk = [rng.randint(0, max_level) for _ in range(12)]
    return {"down": down, "up": up, "jumps": jumps, "random_walk": walk}


def check(arch: str, paper_size: bool = False, device="cpu"):
    """Returns a list of failures (empty when everything matches)."""
    cfg = common.paper_config(arch) if paper_size else common.tiny_config(arch)
    model = _randomize(common.build_shard_model(cfg, device=device)).eval()
    size = cfg.structure.image_size
    x = torch.rand(1, 3, size, size, device=device)
    max_level = model.max_level()
    failures = []

    def run(name, manager):
        manager.set_managed_model(manager.managed_model().to(device).eval())
        model.set_level_use(manager.current_level())
        if not _close(model(x), manager.managed_model()(x)):
            failures.append((arch, name, "start", manager.current_level()))
        for seq_name, seq in sequences(max_level).items():
            for lvl in seq:
                model.set_level_use(lvl)
                manager.move_to(lvl)
                if not _close(model(x), manager.managed_model()(x)):
                    failures.append((arch, name, seq_name, lvl))
        status = "FAIL" if any(f[1] == name for f in failures) else "PASS"
        print(f"[{arch}] {name}: {status}")

    with torch.no_grad():
        model.set_level_use(max_level)
        run("in-memory deltas", levels.InMemoryDeltaManager(model, max_level))

        buf = io.BytesIO()
        levels.FileDeltaManager.make_delta_file(buf, model, starting_level=0)
        buf.seek(0)
        run("file deltas", levels.FileDeltaManager(buf, cfg.create_base_config(0).no_prebuilt()))
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=common.ARCHS + ("both",), default="both")
    ap.add_argument("--paper-size", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    archs = common.ARCHS if args.arch == "both" else (args.arch,)
    failures = []
    for a in archs:
        failures += check(a, args.paper_size, args.device)
    if failures:
        print("FAILED:", failures)
        return 1
    print("all switching checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
