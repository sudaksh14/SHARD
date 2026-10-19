"""Shared helpers: model configs (paper / tiny test sizes), model construction, environment fingerprint."""
from __future__ import annotations

import json
import os
import platform
import sys
import time

import torch

ARCHS = ("shard_deit_v3", "shard_vit")

# Embedding width per level of the paper's 5-level model: D = 384, 480, 576, 672, 768 with 12 heads.
PAPER_WIDTH_UNITS = (32, 40, 48, 56, 64)


def paper_config(arch: str = "shard_deit_v3"):
    """5-level ViT-B/16 SHARD configuration used throughout the paper (no pretrained weights)."""
    kw = dict(
        num_classes=1000,
        num_heads=(12,) * 5,
        hidden_dims=tuple(u * 12 for u in PAPER_WIDTH_UNITS),
        mlp_dims=tuple(u * 48 for u in PAPER_WIDTH_UNITS),
    )
    return _make(arch, **kw)


def tiny_config(arch: str = "shard_deit_v3", levels: int = 3):
    """A very small multi-level config (2 layers, 32x32 input) for CPU smoke tests."""
    heads = tuple(range(2, 2 + levels))            # 2, 3, 4, ...
    kw = dict(
        num_classes=10,
        num_heads=heads,
        hidden_dims=tuple(16 * h for h in heads),  # head size stays 16
        mlp_dims=tuple(64 * h for h in heads),
    )
    structure_args = (32, 16, 2, heads[-1], 16 * heads[-1], 64 * heads[-1])
    return _make(arch, structure=structure_args, **kw)


def _make(arch: str, structure=None, **kw):
    if arch == "shard_deit_v3":
        from networks import shard_deit_v3 as mod
        from networks.vit_v3 import ViTStructureConfig
        cfg_cls = mod.ViTConfig_v3
    elif arch == "shard_vit":
        from networks import shard_vit as mod
        from networks.vit import ViTStructureConfig
        cfg_cls = mod.ViTConfig
    else:
        raise ValueError(f"unknown arch {arch!r}; choose from {ARCHS}")
    if structure is not None:
        kw["structure"] = ViTStructureConfig(*structure)
    return cfg_cls(**kw).no_prebuilt()


def build_shard_model(cfg, checkpoint: str | None = None, seed: int = 0, device="cpu"):
    """Instantiate the SHARD model; random init (seeded) unless a state dict is given."""
    torch.manual_seed(seed)
    model = cfg.make_model()
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt, strict=True)
    return model.to(device).eval()


def make_sync(device):
    """cuda synchronisation function for timing; a no-op on CPU."""
    if torch.device(device).type == "cuda":
        return torch.cuda.synchronize
    return lambda: None


def environment_info() -> dict:
    info = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    for path in ("/proc/device-tree/model", "/etc/nv_tegra_release"):
        try:
            with open(path, "rb") as f:
                info[os.path.basename(path)] = f.read().decode(errors="replace").strip("\x00\n ")
        except OSError:
            pass
    return info


def dump_json(obj, path: str | None):
    text = json.dumps(obj, indent=2)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
        print(f"wrote {path}")
    return text
