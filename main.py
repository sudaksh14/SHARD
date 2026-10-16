#!/usr/bin/env python3
"""
Experiment launcher.

    python main.py list all         # list every experiment key 
    python main.py run  KEY         # run one experiment, e.g. `shard_deit_v3`

KEY is a key of `config.experiments.CONFIGS` (`list all` prints them with a leading `all,`, which is optional):
shard_deit_v3 (the paper's final model), shard_deit_v3_7level, shard_deit_v3_lowFLOPS_5level,
shard_deit_v3_ablation_{random,importance,naive}, shard_deit, shard_resnet, shard_vgg.

Training is launched directly (single node, all visible GPUs); wrap this script
in the job system of your choice if you need one.  Weights & Biases is used in
offline mode unless WANDB_MODE is set in the environment.
"""
from typing import Callable, Generator
import os
import sys

os.environ.setdefault("WANDB_MODE", "offline")

from config.experiments import CONFIGS  # noqa: E402
from training.simple_trainer import BaseTrainer  # noqa: E402


def resolve_from_str(config, start=CONFIGS, return_on_index_error=False) -> Callable[[], BaseTrainer]:
    config = config.split(',')
    subpart = start
    for i in config:
        if i == 'all':
            continue
        try:
            i = int(i)
        except ValueError:
            pass
        try:
            subpart = subpart[i]
        except (KeyError, TypeError) as e:
            if not return_on_index_error:
                raise e
            return subpart
    return subpart


def iter_over_conf(conf, basestr) -> Generator[str, None, None]:
    if isinstance(conf, dict):
        for key, val in conf.items():
            for s in iter_over_conf(val, basestr + f",{key}"):
                yield s
    elif isinstance(conf, list):
        for idx, val in enumerate(conf):
            for s in iter_over_conf(val, basestr + f",{idx}"):
                yield s
    else:
        yield basestr


def print_all_conf_paths(conf, basestr, file=sys.stdout) -> None:
    for s in iter_over_conf(conf, basestr):
        print(s, file=file)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    command, conf = sys.argv[1:]
    res = resolve_from_str(conf)
    if command == "list":
        print_all_conf_paths(res, conf)
    elif command == "run":
        res(conf)
    else:
        sys.exit(f"unknown command {command!r}\n\n{__doc__}")
