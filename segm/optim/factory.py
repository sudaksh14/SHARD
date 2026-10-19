from timm import scheduler
from timm import optim
import math
import torch
from torch.optim import AdamW

from segm.optim.scheduler import PolynomialLR


def create_scheduler(opt_args, optimizer):
    if opt_args.sched == "polynomial":
        # lr_scheduler = PolynomialLR(
        #     optimizer,
        #     opt_args.poly_step_size,
        #     opt_args.iter_warmup,
        #     opt_args.iter_max,
        #     opt_args.poly_power,
        #     opt_args.min_lr,
        # )
        lr_scheduler = PolynomialLR(
            optimizer,
            opt_args.iter_warmup,
            opt_args.iter_max,
            opt_args.poly_power,
            opt_args.min_lr,
        )
    else:
        lr_scheduler, _ = scheduler.create_scheduler(opt_args, optimizer)
    return lr_scheduler


def create_optimizer(opt_args, model):
    return optim.create_optimizer(opt_args, model)


def get_num_layer_for_vit(name, num_layers):
    if name in ("cls_token", "pos_embed"):
        return 0
    elif name.startswith("patch_embed"):
        return 0
    elif name.startswith("blocks"):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers - 1

def get_layer_scale(layer_id, num_layers, layer_decay):
    return layer_decay ** (num_layers - layer_id - 1)

def build_optimizer(model, base_lr=6e-5, decoder_lr=6e-4, weight_decay=0.05, layer_decay=0.75):
    param_groups = []
    num_layers = len(model.encoder.blocks) + 1  # ViT depth

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Skip weight decay for certain params
        if len(param.shape) == 1 or name.endswith(".bias") or "pos_embed" in name or "cls_token" in name:
            decay = 0.0
        else:
            decay = weight_decay

        # Encoder vs Decoder LR
        if name.startswith("encoder"):
            layer_id = get_num_layer_for_vit(name.replace("encoder.", ""), num_layers)
            scale = get_layer_scale(layer_id, num_layers, layer_decay)
            lr = base_lr * scale
        else:
            lr = decoder_lr

        param_groups.append({
            "params": [param],
            "lr": lr,
            "weight_decay": decay,
        })

    optimizer = AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    return optimizer


class WarmupCosineScheduler:
    def __init__(self, optimizer, total_iters, warmup_iters=1500, min_lr=1e-6):
        self.optimizer = optimizer
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.iter = 0

    def step(self):
        self.iter += 1

        if self.iter < self.warmup_iters:
            # Linear warmup
            lr_scale = self.iter / self.warmup_iters
        else:
            # Cosine decay
            progress = (self.iter - self.warmup_iters) / (self.total_iters - self.warmup_iters)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        for i, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            lr = self.min_lr + (base_lr - self.min_lr) * lr_scale
            param_group["lr"] = lr