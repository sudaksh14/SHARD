# Reproducing the paper's results

Install the dependencies with `pip install -r requirements.txt` and run all commands from the repository root.

Checkpoints are described in [`../checkpoints/README.md`](../checkpoints/README.md):
`SHARD_deit_v3_5level.pt` (main model), `SHARD_deit_v3_7level.pt`, `SHARD_deit_v3_lowFLOPS_5level.pt`, and the
segmentation snapshot `segm/checkpoint.pth`.

## Experiments

| Paper result | Code | Config / key | Checkpoint | Command |
|---|---|---|---|---|
| Table 2: parameters and GFLOPs per level | `networks/shard_deit_v3.py` | `utils.switching_harness.common.paper_config("shard_deit_v3")` | `SHARD_deit_v3_5level.pt` | `python examples/02_select_levels.py --checkpoint checkpoints/SHARD_deit_v3_5level.pt` |
| Table 2: top-1 accuracy per level | `scripts/evaluate_imagenet_levels.py` | preset `shard5` | `SHARD_deit_v3_5level.pt` | `IMAGENET_PATH=/data/imagenet python scripts/evaluate_imagenet_levels.py --checkpoint checkpoints/SHARD_deit_v3_5level.pt` |
| Sec. 4.2, Table 1: SHARD training | `main.py`, `training/shard_trainer.py` | `shard_deit_v3` | writes `shard_deit_v3.pt` to `$SHARD_SAVED_MODELS` | `IMAGENET_PATH=/data/imagenet python main.py run shard_deit_v3` |
| Fig. 3a: granularity study | same | `shard_deit_v3_7level`, `shard_deit_v3_lowFLOPS_5level` | `SHARD_deit_v3_7level.pt`, `SHARD_deit_v3_lowFLOPS_5level.pt` | `python main.py run shard_deit_v3_7level`; evaluate with `--preset lowflops7` / `lowflops5` |
| Head-aware sharding ablation | `networks/head_ablations.py` | `shard_deit_v3_ablation_random`, `_importance`, `_naive` | trained with `main.py` | `python main.py run shard_deit_v3_ablation_importance`; evaluate with `--preset heads5` |
| Table 3: energy per inference (Jetson Orin) | `utils/jetson_harness/` | `utils/jetson_harness/configs/paper_15W_onnx_fp32.json` | not needed | measure on a Jetson: `utils/jetson_harness/scripts/run_energy_benchmark.sh OUT`; energy table from the output: `python -m utils.jetson_harness.energy_stats OUT/*.json` |
| Fig. 4, Fig. 5: latency, power, memory per level | same | same | not needed | same runs; the plots use the `latency_ms`, `power_mW` and `ram` fields of the output JSON |
| Sec. 4.6: switching latency (independent models vs. delta manager) | `utils/switching_harness/` | - | optional | `python -m utils.switching_harness delta --iters 100 ...`, `python -m utils.switching_harness full-model --trials 100 ...`, or `ITERS=100 utils/jetson_harness/scripts/run_switching_benchmark.sh OUT` (paper results: 100 timed switches/loads per pair; the CLI default is 10) |
| Fig. 6a: segmentation (ADE20K, linear decoder) | `segm/` | `segm/config.yml` (`shard`) | backbone initialisation `SHARD_deit_v3_5level.pt` | see "Segmentation" below |
| Fig. 6b: image retrieval | `scripts/evaluate_retrieval.py` | - | `SHARD_deit_v3_5level.pt` | `python scripts/evaluate_retrieval.py --checkpoint checkpoints/SHARD_deit_v3_5level.pt` |

Training needs a CUDA GPU and the ImageNet-1k folder at `$IMAGENET_PATH` (`train/` and `val/`). Output locations are set with
environment variables defined in `config/paths.py` (`SHARD_WORKDIR`, `SHARD_SAVED_MODELS`, `SHARD_CHECKPOINT_DIR`, `SHARD_DATA_DIR`).

## Training modes

`ShardDistillContext` (in `training/shard_trainer.py`) selects which levels are trained in each step:

* `training_mode='sampled'` with `num_sampled=N` (default: `N = 4`): the widest level, the narrowest level (0) and `N - 2` levels drawn at random from the levels in between.
* `training_mode='simple'`: every level, from the widest to level 0 (paper Algorithm 1).
  `N = 2` trains only the two extreme levels; `N` at least the number of levels trains all levels. The draw is repeated in every step.

In both modes every trained level is distilled from the next wider trained level (`transfer_type='progressive'`, the default), or from the widest level
with `transfer_type='US'`.

## Energy and latency measurement (Jetson)

`utils/jetson_harness/benchmark_levels.py` runs 20 warm-up inferences, then 100 timed inferences per level. After each timed inference the jtop
`Power TOT` value is read. Energy per inference is power x latency of the same run (`utils/jetson_harness/energy_stats.py`). The SHARD row of Table 3
is the ONNX Runtime fp32 run; the PyTorch-eager runs use the same procedure.

Software stack of the reported measurements: Jetson Orin 8 GB, power mode 15 W, L4T R35 rev. 5.0, PyTorch 2.1.0a0 (NVIDIA build), CUDA 11.4,
Python 3.8.10, ONNX Runtime 1.15.1 (CUDA execution provider, fp32, batch size 1).

## Segmentation (Fig. 6a)

The segmentation loop (`segm/engine_shard.py`) uses the same scheme as the ImageNet trainer. The reported run used 100 epochs on 2 GPUs with batch size 8,
AdamW (lr 1e-4), a polynomial schedule (power 0.9), weight decay 0, AMP, 512x512 crops and a linear decoder. Command:

```bash
pip install mmcv==1.3.8 mmsegmentation==0.14.1   # old pins, use a separate environment (see segm/requirements.txt)
python segm/scripts/prepare_ade20k.py /data      # downloads ADE20K into /data/ade20k
export DATASET=/data
python -m segm.train --log-dir runs/seg_shard_poly --dataset ade20k --im-size 512 --crop-size 512 \
    --backbone shard --decoder linear --optimizer adamw --scheduler polynomial \
    --weight-decay 0.0 --batch-size 8 --epochs 100 -lr 0.0001 --eval-freq 2 --amp
```

`segm/config.yml` (`model.shard.pretrained_model`) points to `checkpoints/SHARD_deit_v3_5level.pt` by default.
