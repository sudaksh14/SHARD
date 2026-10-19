#!/usr/bin/env python3
"""
Example A - load a SHARD model, pick a level, run inference.

    python examples/01_load_and_run.py --checkpoint checkpoints/SHARD_deit_v3_5level.pt --level 4 [--image cat.jpg]
    python examples/01_load_and_run.py --tiny        # no checkpoint / no download: random 3-level toy model

Without --image a random tensor is used (control-flow check only). With an image, the standard ImageNet
preprocessing of the evaluation code is applied (resize 256 -> center crop 224, DeiT mean/std) and the top-5
ImageNet-1k class *indices* are printed (class names are not bundled).
Output: logits of shape (1, num_classes) for the selected level.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from utils.switching_harness import common


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="SHARD_deit_v3_5level.pt (state dict of the 5-level model)")
    ap.add_argument("--level", type=int, default=None, help="0 = smallest ... max_level = full width (default: largest)")
    ap.add_argument("--image", default=None)
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()

    cfg = common.tiny_config("shard_deit_v3") if args.tiny else common.paper_config("shard_deit_v3")
    model = common.build_shard_model(cfg, args.checkpoint)        # strict load; random init if no checkpoint
    level = model.max_level() if args.level is None else args.level
    model.set_level_use(level)                                   # selects the nested sub-transformer
    print(f"levels available: 0..{model.max_level()}, using level {level}")

    size = cfg.structure.image_size
    if args.image:
        from PIL import Image
        from torchvision import transforms
        from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        tf = transforms.Compose([transforms.Resize(int(size / 0.875), interpolation=3), transforms.CenterCrop(size),
                                 transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        x = tf(Image.open(args.image).convert("RGB")).unsqueeze(0)
    else:
        x = torch.randn(1, 3, size, size)

    with torch.no_grad():
        logits = model(x)
    print("logits shape:", tuple(logits.shape))
    print("top-5 class indices:", logits.softmax(-1).topk(min(5, logits.shape[-1])).indices[0].tolist())


if __name__ == "__main__":
    main()
