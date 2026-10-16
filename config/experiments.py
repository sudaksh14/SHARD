"""
Experiment registry used by `main.py`  (`python main.py run <key>` / `python main.py list all`).

Keys
----
shard_deit_v3                      final model of the paper: 5 levels, D = 384/480/576/672/768, DeiT-III ViT-B/16
shard_deit_v3_7level               7 levels, D = 192/288/384/480/576/672/768   (granularity study, Fig. 3a)
shard_deit_v3_lowFLOPS_5level      5 levels, D = 192/384/576/672/768           (granularity study, Fig. 3a)
shard_deit_v3_ablation_random      head ablation: 4/6/8/10/12 heads, pretrained heads in random order
shard_deit_v3_ablation_importance  head ablation: 4/6/8/10/12 heads, heads ordered by importance
shard_deit_v3_ablation_naive       head ablation: 4/6/8/10/12 heads, pretrained head order unchanged (plain head removal)
shard_deit                         earlier torchvision-style ViT-B/16 (networks/shard_vit.py), SHARD trainer
shard_resnet                       CNN compatibility example: ResNet-20, 3 levels, CIFAR-10
shard_vgg                          CNN compatibility example: VGG-11, 3 levels, CIFAR-10

All `shard_deit_v3*` entries share one training recipe (`_deit_v3_context`): LAMB, lr 5e-4, weight decay 0.05, cosine
schedule with 5 warm-up epochs, 100 epochs, batch size 256, mixup/cutmix. The recipe of the released five-level model is
documented in docs/reproduction.md; for the 7-level, lowFLOPS and head-ablation entries the model configurations were
recovered from the released checkpoints / paper, while their training recipe is assumed to be this same one.
"""
from functools import partial

import torch.optim as optim
from timm.optim import Lamb
from timm.scheduler import CosineLRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import CIFAR10

import training.shard_trainer as shard_trainer
import utils.dataset
import utils
from networks import shard_deit_v3, shard_resnet, shard_vgg, shard_vit
from training.simple_trainer import ShardModelTrainer, ShardTrainingContext, TrainerBuilder


# ----------------------------------------------------------------------------- CNN compatibility examples
class ModelTraining(ShardTrainingContext):
    def __init__(self, *args, **kwargs):
        super().__init__(partial(utils.load_data, CIFAR10),
                         patience=50, epochs=-1, *args, **kwargs)

    def make_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-5)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=300)


# ----------------------------------------------------------------------------------- DeiT-III recipe
def _deit_v3_context():
    return shard_trainer.ShardDistillContext(
        loader_function=partial(utils.dataset.load_imagenet, batch_size=256),
        make_optimizer=lambda m: Lamb(m.parameters(), lr=5e-4, weight_decay=0.05, betas=(0.9, 0.999)),
        make_scheduler=lambda opt: CosineLRScheduler(
            optimizer=opt, t_initial=100, lr_min=1e-6, warmup_lr_init=1e-6,
            warmup_t=5, cycle_limit=1, t_in_epochs=True),
        mixup_fn=utils.mixup_fn,
        patience=20, epochs=100,
        label_smoothing=0.11, gradient_clip_val=1.0)


def _deit_v3_builder(width_units, head_permutation="none", num_heads=None, head_size=None):
    """Width-unit (x12 for the embedding, x48 for the MLP) description used by the 12-head models."""
    if num_heads is None:
        cfg = shard_deit_v3.ViTConfig_v3(
            num_classes=1000,
            num_heads=(12,) * len(width_units),
            hidden_dims=tuple(u * 12 for u in width_units),
            mlp_dims=tuple(u * 48 for u in width_units),
            head_permutation=head_permutation)
    else:  # head-removal models: fixed head size, fewer heads in the lower levels
        cfg = shard_deit_v3.ViTConfig_v3(
            num_classes=1000,
            num_heads=tuple(num_heads),
            hidden_dims=tuple(head_size * h for h in num_heads),
            mlp_dims=tuple(head_size * 4 * h for h in num_heads),
            head_permutation=head_permutation)
    return TrainerBuilder(shard_trainer.ShardDistillTrainer, cfg, _deit_v3_context())


MAIN_WIDTHS = (32, 40, 48, 56, 64)
ABLATION_HEADS = (4, 6, 8, 10, 12)

CONFIGS = {
    # ---- final model and granularity studies
    "shard_deit_v3": _deit_v3_builder(MAIN_WIDTHS),
    "shard_deit_v3_7level": _deit_v3_builder((16, 24, 32, 40, 48, 56, 64)),
    "shard_deit_v3_lowFLOPS_5level": _deit_v3_builder((16, 32, 48, 56, 64)),

    # ---- head-aware sharding ablations (all keep 64-dim heads, heads are removed instead of shrunk)
    "shard_deit_v3_ablation_random": _deit_v3_builder(
        None, head_permutation="random", num_heads=ABLATION_HEADS, head_size=64),
    "shard_deit_v3_ablation_importance": _deit_v3_builder(
        None, head_permutation="importance", num_heads=ABLATION_HEADS, head_size=64),
    "shard_deit_v3_ablation_naive": _deit_v3_builder(
        None, head_permutation="none", num_heads=ABLATION_HEADS, head_size=64),

    # ---- earlier ViT-B/16 with the same SHARD trainer
    "shard_deit": TrainerBuilder(
        shard_trainer.ShardDistillTrainer,
        shard_vit.ViTConfig(
            num_classes=1000,
            num_heads=(12, 12, 12, 12, 12),
            hidden_dims=tuple(u * 12 for u in MAIN_WIDTHS),
            mlp_dims=tuple(u * 48 for u in MAIN_WIDTHS)),
        shard_trainer.ShardDistillContext(
            loader_function=partial(utils.dataset.load_imagenet, batch_size=512),
            make_optimizer=lambda m: optim.AdamW(m.parameters(), lr=5e-4, weight_decay=0.05),
            make_scheduler=lambda opt: CosineAnnealingLR(optimizer=opt, T_max=150, eta_min=1e-5),
            mixup_fn=utils.mixup_fn,
            patience=20, epochs=150,
            label_smoothing=0.11, gradient_clip_val=1.0)),

    # ---- CNN compatibility (the level/delta machinery is not ViT specific)
    "shard_resnet": TrainerBuilder(
        ShardModelTrainer, shard_resnet.ResnetConfig(), ModelTraining()),
    "shard_vgg": TrainerBuilder(
        ShardModelTrainer, shard_vgg.VGGConfig(), ModelTraining()),
}
