"""
Filesystem locations used by training, evaluation and dataset loading.

Every location can be overridden with an environment variable, so nothing here
is tied to a particular machine.  Importing this module does NOT create any
directory; call `ensure_dir` on a path right before writing to it.

    SHARD_WORKDIR         scratch/working directory        (default: ./.shard)
    SHARD_TMPDIR          staging copy of the dataset dir  (default: $SHARD_WORKDIR/temp)
    SHARD_DATA_DIR        CIFAR & other small datasets     (default: $SHARD_WORKDIR/data)
    SHARD_SAVED_MODELS    trained state dicts (utils.save_model)
                                                           (default: $SHARD_WORKDIR/saved_models)
    SHARD_CHECKPOINT_DIR  Lightning training checkpoints   (default: $SHARD_WORKDIR/checkpoints)
    IMAGENET_PATH         unpacked ImageNet-1k folder with train/ and val/
                                                           (default: ./data/imagenet)
"""
from __future__ import annotations

from pathlib import Path
import os


def ensure_dir(path) -> Path:
    """Create `path` (and parents) if needed and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_path(path) -> Path:
    """Backwards-compatible alias that (unlike the original) is only called on demand."""
    return ensure_dir(path)


def make_file(path) -> Path:
    ensure_dir(Path(path).parent)
    return Path(path)


def _env_path(name: str, default) -> Path:
    return Path(os.environ.get(name, default))


# SLURM_JOBID is only used to keep per-job log directories apart (0 when not set).
SLURM_JOBID = int(os.environ.get('SLURM_JOBID', default='0'))

PROJECT_DIR = _env_path('SHARD_WORKDIR', './.shard')
TMPDIR = _env_path('SHARD_TMPDIR', PROJECT_DIR / 'temp')

JOBDIR = PROJECT_DIR / 'wandb' / f'job_{SLURM_JOBID}'
DATA_PATH = _env_path('SHARD_DATA_DIR', PROJECT_DIR / 'data')
LOG_PATH = JOBDIR / 'logs'

TRAINED_MODELS = _env_path('SHARD_SAVED_MODELS', PROJECT_DIR / 'saved_models')
CHECKPOINT_PATH = _env_path('SHARD_CHECKPOINT_DIR', PROJECT_DIR / 'checkpoints')

FIGURES = PROJECT_DIR / 'figures'
IMAGENET_PATH = _env_path('IMAGENET_PATH', './data/imagenet')
