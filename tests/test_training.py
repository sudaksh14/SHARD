from config.experiments import CONFIGS
import shutil
import dataclasses

from torchvision.transforms import (
    Compose, RandomHorizontalFlip, RandomRotation,
    ColorJitter, ToTensor, Normalize, Resize, CenterCrop, ConvertImageDtype
)
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch

import config.paths as paths
import networks.shard_model
import shard_modules as fm
import networks.config
import unittest
import training.simple_trainer as training
import networks
import utils


torch.manual_seed(seed=0)


def load_dummy_data(data_dir=paths.DATA_PATH, tmp_dir=paths.TMPDIR, batch_size=8):
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if tmp_dir:
        utils.try_make_dir(data_dir)
        utils.try_make_dir(tmp_dir)
        shutil.copytree(data_dir, tmp_dir, dirs_exist_ok=True)
    dataset = CIFAR10(
        root=data_dir if tmp_dir is None else tmp_dir,
        train=True, download=True, transform=transform)

    if tmp_dir is not None:
        shutil.copytree(tmp_dir, data_dir, dirs_exist_ok=True)

    dataset, _ = random_split(
        dataset, [64 * 8, len(dataset) - 64 * 8])

    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=8)
    test_dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=8)

    return train_dataloader, val_dataloader, test_dataloader


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 60, kernel_size=3, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace=True),
            nn.Conv2d(60, 60, kernel_size=3, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace=True),
            nn.Conv2d(60, 60, kernel_size=3, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(inplace=True),
        )

        self.head = fm.LinearHead(60, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class DummyShardModel(networks.shard_model.ShardModel):
    def __init__(self):
        super().__init__(DummyShardModelConfig())
        self.features = nn.Sequential(
            fm.Conv2d([3, 3, 3], [30, 50, 60], kernel_size=3, padding=1),
            fm.BatchNorm2d([30, 50, 60]),
            nn.ReLU(inplace=True),
            fm.Conv2d([30, 50, 60], [30, 50, 60], kernel_size=3, padding=1),
            fm.BatchNorm2d([30, 50, 60]),
            nn.ReLU(inplace=True),
            fm.Conv2d([30, 50, 60], [30, 50, 60], kernel_size=3, padding=1),
            fm.BatchNorm2d([30, 50, 60]),
            nn.ReLU(inplace=True)
        )

        self.head = fm.LinearSelect([30, 50, 60], [10, 10, 10])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

    def current_level(self):
        return self.head.current_level()

    def max_level(self):
        return self.head.max_level()

    @staticmethod
    def base_type() -> type[nn.Module]:
        return DummyModel


@dataclasses.dataclass
class DummyModelConfig(networks.config.ShardModelConfig):
    def make_model(self) -> networks.shard_model.ShardModel:
        return DummyModel()

    def no_prebuilt(self) -> networks.config.ShardModelConfig:
        return self


@dataclasses.dataclass
class DummyShardModelConfig(networks.config.ShardModelConfig):
    def make_model(self) -> networks.shard_model.ShardModel:
        return DummyShardModel()

    def no_prebuilt(self) -> networks.config.ShardModelConfig:
        return self

    def create_base_config(self, level) -> networks.config.ModelConfig:
        return DummyModelConfig()

    def max_level(self) -> int:
        return 2


@dataclasses.dataclass
class DummyShardTrainingContext(training.ShardTrainingContext):
    def make_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-5)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=self.epochs)


@dataclasses.dataclass
class DummyTrainingContext(training.TrainingContext):
    def make_optimizer(self, model):
        return optim.Adam(model.parameters(), lr=1e-5)

    def make_scheduler(self, optimizer):
        return CosineAnnealingLR(optimizer, T_max=self.epochs)


CONFIGS['dummy'] = training.TrainerBuilder(
    training.ShardModelTrainer,
    DummyShardModelConfig(),
    DummyShardTrainingContext(
        load_dummy_data, patience=1, epochs=1, wandb_project_name=None, unittest_mode=True)
)


@unittest.skipUnless(torch.cuda.is_available(),
                     "integration test: needs a CUDA GPU (trainers hard-code accelerator='gpu') and downloads CIFAR-10")
class TestTrainer(unittest.TestCase):
    def test_shard(self):
        training.TrainerBuilder(
            training.ShardModelTrainer,
            DummyShardModelConfig(),
            DummyShardTrainingContext(
                load_dummy_data, patience=1, epochs=1, wandb_project_name=None, unittest_mode=True)
        ).run_training('dummy')

    def test_shard_load_from(self):
        training.TrainerBuilder(
            training.ShardModelTrainer,
            DummyShardModelConfig(),
            DummyShardTrainingContext(
                load_dummy_data,
                patience=1,
                epochs=1,
                wandb_project_name=None,
                unittest_mode=True,
                load_from='dummy')
        ).run_training('loadfromdummy')

    def test_shard_load_from(self):
        training.TrainerBuilder(
            training.ShardModelTrainer,
            DummyShardModelConfig(),
            DummyShardTrainingContext(
                load_dummy_data,
                patience=1,
                epochs=1,
                wandb_project_name=None,
                unittest_mode=True,
                load_from='dummy',
                load_shard_from=(0, 1, 2),
                load_shard_to=(0, 1, 2))
        ).run_training('loadfromdummy')

    def test_simple(self):
        training.TrainerBuilder(
            training.SimpleTrainer,
            DummyModelConfig(),
            DummyTrainingContext(
                load_dummy_data, patience=1, epochs=1, wandb_project_name=None, unittest_mode=True)
        ).run_training('dummysimple')
