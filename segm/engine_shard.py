import torch
import math
import sys
from segm.utils.logger import MetricLogger
from segm.metrics import gather_data, compute_metrics
from segm.model import utils
from segm.data.utils import IGNORE_LABEL
import segm.utils.torch as ptu

import torch.nn.functional as F
import torch.nn as nn


def train_one_epoch(
    model,
    data_loader,
    optimizer,
    lr_scheduler,
    epoch,
    amp_autocast,
    loss_scaler,
    ce_coefficient=1.0,
    distill_type="soft",   # "soft" or "hard"
    transfer_type="progressive",  # "progressive" or "US"
):
    """
    One epoch of SHARD training for segmentation (same scheme as training/shard_trainer.py, paper Algorithm 1):
    every level, from the widest to level 0, is run on the mini-batch and back-propagated; the accumulated
    gradients give one optimiser step. The widest level uses the segmentation labels; each narrower level is
    distilled (KL or hard labels) from the next wider one ("progressive") or from the widest ("US"), plus
    `ce_coefficient` x cross-entropy.
    """
    ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"
    print_freq = 100

    model.train()
    data_loader.set_epoch(epoch)
    num_updates = epoch * len(data_loader)

    max_level = getattr(model, "module", model).encoder.max_level()

    def backward(loss):
        if loss_scaler is not None:
            loss_scaler.scale(loss).backward()
        else:
            loss.backward()

    for batch in logger.log_every(data_loader, print_freq, header):
        im = batch["im"].to(ptu.device)
        seg_gt = batch["segmentation"].long().to(ptu.device)

        optimizer.zero_grad()

        level_losses = {}
        with amp_autocast():
            out_full = None
            teacher = None
            for level in range(max_level, -1, -1):
                out = model(im, level=level)
                if level == max_level:
                    loss_level = ce_loss_fn(out, seg_gt)
                    out_full = out
                else:
                    if distill_type == "hard":
                        kd = F.cross_entropy(out, teacher.detach().argmax(dim=1))
                    else:
                        kd = kl_loss(F.log_softmax(out, dim=1), F.softmax(teacher.detach(), dim=1))
                    loss_level = kd + ce_coefficient * ce_loss_fn(out, seg_gt)
                backward(loss_level)
                level_losses[level] = loss_level.detach()
                teacher = out if transfer_type != "US" else out_full

            loss = sum(level_losses.values())

        loss_val = loss.item()
        if not math.isfinite(loss_val):
            print(f"Loss is {loss_val}, stopping training")
            sys.exit(1)

        # =========================
        # BACKWARD
        # =========================
        if loss_scaler is not None:
            loss_scaler.step(optimizer)
            loss_scaler.update()
        else:
            optimizer.step()
            

        # =========================
        # LR STEP
        # =========================
        # num_updates += 1
        # lr_scheduler.step_update(num_updates=num_updates)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # =========================
        # LOGGING
        # =========================
        logger.update(
            loss=loss.item(),
            **{f"loss_level{k}": v.item() for k, v in level_losses.items()},
            lr_backbone=float(optimizer.param_groups[0]["lr"]),
            lr_decoder=float(optimizer.param_groups[1]["lr"]),
        )

    return logger


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    val_seg_gt,
    window_size,
    window_stride,
    amp_autocast,
):
    model_without_ddp = model
    if hasattr(model, "module"):
        model_without_ddp = model.module

    logger = MetricLogger(delimiter="  ")
    header = "Eval:"
    print_freq = 50

    model.eval()

    max_level = 4

    # Store predictions per level
    val_seg_pred_levels = {
        level: {} for level in range(max_level + 1)
    }

    for batch in logger.log_every(data_loader, print_freq, header):
        ims = [im.to(ptu.device) for im in batch["im"]]
        ims_metas = batch["im_metas"]

        ori_shape = ims_metas[0]["ori_shape"]
        ori_shape = (ori_shape[0].item(), ori_shape[1].item())
        filename = batch["im_metas"][0]["ori_filename"][0]

        for level in range(max_level + 1):

            with amp_autocast():
                seg_pred = utils.inference(
                    model_without_ddp,
                    level,
                    ims,
                    ims_metas,
                    ori_shape,
                    window_size,
                    window_stride,
                    batch_size=1,
                )
                seg_pred = seg_pred.argmax(0)

            seg_pred = seg_pred.cpu().numpy()
            val_seg_pred_levels[level][filename] = seg_pred

    # =========================
    # Compute metrics per level
    # =========================
    results = {}

    for level in range(max_level + 1):
        preds = gather_data(val_seg_pred_levels[level])

        scores = compute_metrics(
            preds,
            val_seg_gt,
            data_loader.unwrapped.n_cls,
            ignore_index=IGNORE_LABEL,
            distributed=ptu.distributed,
        )

        # Log per-level metrics
        for k, v in scores.items():
            logger.update(**{f"{k}_level{level}": v, "n": 1})

        results[level] = scores

    return logger, results