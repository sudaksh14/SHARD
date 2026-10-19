"""
Image-retrieval experiment (paper Sec. 4.7, Fig. 6b): 1-NN accuracy / mAP@10 on the ImageNet-1k
validation split (self-matches excluded). Database features come from a full DeiT-B model; queries come from
DeiT-S / DeiT-B baselines and from every SHARD level. The features are the L2-normalised model outputs (classifier logits).

    python scripts/evaluate_retrieval.py --checkpoint checkpoints/SHARD_deit_v3_5level.pt

Needs: ImageNet-1k val (IMAGENET_PATH), a CUDA GPU, internet access for the timm baselines
(downloaded by timm on first use), and torch-pruning for FLOPs counting.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
import torch_pruning as tp
from utils.dataset import *
from networks.shard_deit_v3 import ViTConfig_v3

# -----------------------
# CONFIG
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128
num_workers = 8
levels = [0, 1, 2, 3, 4]

# proxy compute cost (relative)
level_compute = {0: 0.2, 1: 0.4, 2: 0.6, 3: 0.8, 4: 1.0}
shard_cfg = ViTConfig_v3(
                num_classes=1000,
                num_heads=(12, 12, 12, 12, 12),
                hidden_dims=(32 * 12, 40 * 12, 48 * 12, 56 * 12, 64 * 12),
                mlp_dims=(32 * 48, 40 * 48, 48 * 48, 56 * 48, 64 * 48)).no_prebuilt()  # weights come from --checkpoint
# -----------------------
# SHARD
# -----------------------
def load_shard(model_cfg, checkpoint_path=None):
    model = model_cfg.make_model()
    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device)
        sdict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(sdict, strict=True)
        print(f"Loaded pretrained model from {checkpoint_path}")
    return model

# -----------------------
# FEATURE EXTRACTION
# -----------------------
@torch.no_grad()
def extract_features(model, loader, level=None):
    model.eval()
    feats, labels = [], []

    for images, target in tqdm(loader):
        images = images.to(device)

        if level is None:
            out = model(images)
        else:
            model.set_level_use(level)
            out = model(images)

        # CLS token
        if out.dim() == 3:
            out = out[:, 0]

        out = F.normalize(out, dim=1)

        feats.append(out.cpu())
        labels.append(target)

    return torch.cat(feats), torch.cat(labels)

# -----------------------
# METRICS
# -----------------------
def compute_metrics(q_feat, q_lbl, db_feat, db_lbl, k=10):

    sim = q_feat @ db_feat.T

    # ❗ remove self-match
    sim.fill_diagonal_(-1e9)

    # -------- 1-NN --------
    nn_idx = sim.argmax(dim=1)
    nn_labels = db_lbl[nn_idx]
    acc = (nn_labels == q_lbl).float().mean().item()

    # -------- mAP@k --------
    topk = sim.topk(k=k, dim=1).indices  # (N, k)
    topk_labels = db_lbl[topk]           # (N, k)

    correct = (topk_labels == q_lbl.unsqueeze(1)).float()

    # precision@i
    precision = correct.cumsum(dim=1) / torch.arange(1, k+1)

    # AP per query
    ap = (precision * correct).sum(dim=1) / correct.sum(dim=1).clamp(min=1)

    map_k = ap.mean().item()

    return acc, map_k

# -----------------------
# MAIN BENCHMARK
# -----------------------
def main(checkpoint_path, batch_size_=256):

    _, val_loader,_ = load_imagenet(batch_size=batch_size_, debug=False)
    
    full_model = timm.create_model('deit3_base_patch16_224.fb_in22k_ft_in1k', pretrained=True, num_classes=1000)
    full_model.eval().to(device)
    
    print("🔵 Building DB features (DeiT-B III pretrained )...")
    db_feat, db_lbl = extract_features(full_model, val_loader)
    
    acc, map10 = compute_metrics(db_feat, db_lbl, db_feat, db_lbl, k=10)
    flops, param = tp.utils.count_ops_and_params(full_model, torch.randn(1,3,224,224).to(device))
    print(f"DeiT-B III pretrained → 1-NN Acc: {acc*100:.2f}%, GFLOPs: {flops / 1e9:.2f}, Params (M): {param / 1e6:.2f}, mAP@10: {map10:.4f}")
    
    vit_s = timm.create_model('deit3_small_patch16_224.fb_in22k_ft_in1k', pretrained=True).to(device).eval()
    q_feat, q_lbl = extract_features(vit_s, val_loader)
    acc, map10 = compute_metrics(q_feat, q_lbl, db_feat, db_lbl, k=10)
    flops, param = tp.utils.count_ops_and_params(vit_s, torch.randn(1,3,224,224).to(device))
    print(f"DeiT-S III pretrained → 1-NN Acc: {acc*100:.2f}%, GFLOPs: {flops / 1e9:.2f}, Params (M): {param / 1e6:.2f}, mAP@10: {map10:.4f}")

    full_model = timm.create_model('deit_base_patch16_224.fb_in1k', pretrained=True, num_classes=1000)
    full_model.eval().to(device)
    
    print("🔵 Building DB features (DeiT-B pretrained )...")
    db_feat, db_lbl = extract_features(full_model, val_loader)
    
    acc, map10 = compute_metrics(db_feat, db_lbl, db_feat, db_lbl, k=10)
    flops, param = tp.utils.count_ops_and_params(full_model, torch.randn(1,3,224,224).to(device))
    print(f"DeiT-B pretrained → 1-NN Acc: {acc*100:.2f}%, GFLOPs: {flops / 1e9:.2f}, Params (M): {param / 1e6:.2f}, mAP@10: {map10:.4f}")
    
    vit_s = timm.create_model('deit3_small_patch16_224.fb_in22k_ft_in1k', pretrained=True).to(device).eval()
    q_feat, q_lbl = extract_features(vit_s, val_loader)
    acc, map10 = compute_metrics(q_feat, q_lbl, db_feat, db_lbl, k=10)
    flops, param = tp.utils.count_ops_and_params(vit_s, torch.randn(1,3,224,224).to(device))
    print(f"DeiT-S pretrained → 1-NN Acc: {acc*100:.2f}%, GFLOPs: {flops / 1e9:.2f}, Params (M): {param / 1e6:.2f}, mAP@10: {map10:.4f}")
    
    shard_model = load_shard(shard_cfg, checkpoint_path)
    shard_model.eval().to(device)
    
    for lvl in levels:
        print(f"\n🟢 Query Level {lvl}")

        q_feat, q_lbl = extract_features(shard_model, val_loader, level=lvl)

        acc, map10 = compute_metrics(q_feat, q_lbl, db_feat, db_lbl, k=10)

        print(f"Level {lvl} → 1-NN Acc: {acc*100:.2f}, mAP@10: {map10:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="SHARD 5-level state dict (SHARD_deit_v3_5level.pt)")
    ap.add_argument("--batch-size", type=int, default=256)
    a = ap.parse_args()
    main(a.checkpoint, a.batch_size)   