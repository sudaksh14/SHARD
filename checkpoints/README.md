# Checkpoints

Model weights are **not** stored in git. Download them from the project's SURFdrive share and place the files of its `SHARD/` folder directly into this directory and keep the `segm/` sub-folder:

    https://surfdrive.surf.nl/s/RpMqw3sSd2p5SBg      (folder "SHARD checkpoints")

| File | What it is | Config / preset |
|---|---|---|
| `SHARD_deit_v3_5level.pt` | state dict of the final 5-level SHARD ViT-B/16 (D = 384/480/576/672/768, 12 heads) - used for everything in the paper | `shard_deit_v3`, preset `shard5` |
| `SHARD_deit_v3_7level.pt` | 7 levels, D = 192 ... 768 (granularity study) | `shard_deit_v3_7level`, preset `lowflops7` |
| `SHARD_deit_v3_lowFLOPS_5level.pt` | 5 levels, D = 192/384/576/672/768 | `shard_deit_v3_lowFLOPS_5level`, preset `lowflops5` |
| `segm/checkpoint.pth`, `segm/log.txt` | ADE20K segmentation snapshot (model + optimiser state) and per-epoch metrics | `segm/` |

Load a model with

```python
from utils.switching_harness import common
model = common.build_shard_model(common.paper_config("shard_deit_v3"), "checkpoints/SHARD_deit_v3_5level.pt")   # strict load
```

`SHA256SUMS` lists the checksums (`sha256sum -c SHA256SUMS` from this directory).

The weights are fine-tuned from the DeiT-III ImageNet-21k-pretrained ViT-B/16 (`timm`: `deit3_base_patch16_224.fb_in22k_ft_in1k`) and trained on ImageNet-1k;
please check the terms of those assets before redistributing the weights.
