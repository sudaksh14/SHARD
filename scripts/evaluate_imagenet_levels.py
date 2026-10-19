#!/usr/bin/env python3
"""
Evaluate every SHARD level of a trained checkpoint on ImageNet-1k (val split) and report
top-1 accuracy, GFLOPs and parameter count per level.

The model is built from a
`ViTConfig_v3`, the checkpoint is loaded with `strict=True`, and each level is turned into a
regular (plain) model with `make_base_copy()` before it is evaluated.

    python scripts/evaluate_imagenet_levels.py --checkpoint checkpoints/SHARD_deit_v3_5level.pt

    # no ImageNet? count FLOPs/params only (still needs the checkpoint to be loadable):
    python scripts/evaluate_imagenet_levels.py --checkpoint checkpoints/SHARD_deit_v3_5level.pt --no-accuracy

Presets (--preset) are the level configurations used in the paper and its ablations:
    shard5      5 levels, D = 384/480/576/672/768, 12 heads each        (main model, SHARD_deit_v3_5level.pt)
    lowflops7   7 levels, D = 192/288/384/480/576/672/768               (SHARD_deit_v3_7level.pt)
    lowflops5   5 levels, D = 192/384/576/672/768                       (SHARD_deit_v3_lowFLOPS_5level.pt)
    heads5      5 levels, 4/6/8/10/12 heads of size 64 (head removal)   (models of the shard_deit_v3_ablation_* experiments)

Requires ImageNet-1k at $IMAGENET_PATH (train/ and val/), `torch-pruning` for FLOPs counting,
and scikit-learn for the accuracy computation.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import utils
from networks.shard_deit_v3 import ViTConfig_v3

PRESETS = {
    "shard5": dict(
        num_heads=(12,) * 5,
        hidden_dims=tuple(k * 12 for k in (32, 40, 48, 56, 64)),
        mlp_dims=tuple(k * 48 for k in (32, 40, 48, 56, 64))),
    "lowflops7": dict(
        num_heads=(12,) * 7,
        hidden_dims=tuple(k * 12 for k in (16, 24, 32, 40, 48, 56, 64)),
        mlp_dims=tuple(k * 48 for k in (16, 24, 32, 40, 48, 56, 64))),
    "lowflops5": dict(
        num_heads=(12,) * 5,
        hidden_dims=tuple(k * 12 for k in (16, 32, 48, 56, 64)),
        mlp_dims=tuple(k * 48 for k in (16, 32, 48, 56, 64))),
    "heads5": dict(
        num_heads=(4, 6, 8, 10, 12),
        hidden_dims=tuple(64 * k for k in (4, 6, 8, 10, 12)),
        mlp_dims=tuple(64 * k for k in (16, 24, 32, 40, 48))),
}


def load_shard_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sdict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(sdict, strict=True)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="state dict written by utils.save_model / released .pt")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="shard5")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    ap.add_argument("--no-accuracy", action="store_true", help="skip ImageNet evaluation (FLOPs/params only)")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else utils.get_device()
    cfg = ViTConfig_v3(num_classes=1000, **PRESETS[args.preset]).no_prebuilt()
    model = load_shard_checkpoint(cfg.make_model(), args.checkpoint, device).eval().to(device)
    print(f"Loaded {args.checkpoint} into preset '{args.preset}' ({model.max_level() + 1} levels)")

    test_loader = None
    if not args.no_accuracy:
        from utils.dataset import load_imagenet
        _, _, test_loader = load_imagenet(batch_size=args.batch_size, debug=False)

    import torch_pruning as tp
    example = torch.randn(1, 3, 224, 224).to(device)
    for level in range(model.max_level() + 1):
        model.set_level_use(level)
        reg_model = model.make_base_copy()
        flops, params = tp.utils.count_ops_and_params(reg_model, example)
        line = f"Level {level}: GFLOPs {flops / 1e9:.2f}, Params (M) {params / 1e6:.2f}"
        if test_loader is not None:
            acc = utils.evaluate_model(reg_model, test_loader, device)
            line = f"Level {level}: Accuracy {acc * 100:.2f}%, " + line.split(": ", 1)[1]
        print(line, flush=True)


if __name__ == "__main__":
    main()
