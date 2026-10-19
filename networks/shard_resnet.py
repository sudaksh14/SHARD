from __future__ import annotations

from typing import Iterable
import dataclasses
import utils

from torch import nn
import torch.nn.functional as F
import torch

from networks.resnet import KNOWN_MODEL_PRETRAINED
from networks.shard_model import ShardModel
from networks.config import ShardModelConfig, ModelConfig
import shard_modules as fm
import networks.resnet


@dataclasses.dataclass
class ResnetConfig(ShardModelConfig):
    num_blocks: Iterable[int] = (3, 3, 3)
    num_classes: int = 10
    small_channels: Iterable[int] = (8, 12, 16)
    mid_channels: Iterable[int] = (16, 24, 32)
    large_channels: Iterable[int] = (32, 48, 64)
    prebuilt: bool = True
    prebuilt_level: int = -1

    def make_model(self) -> 'Resnet':
        return Resnet(self)

    def no_prebuilt(self):
        self.prebuilt = False
        return self

    def create_base_config(self, level) -> ModelConfig:
        return networks.resnet.ResnetConfig(
            self.num_blocks,
            self.num_classes,
            self.small_channels[level],
            self.mid_channels[level],
            self.large_channels[level],
            self.prebuilt)

    def max_level(self) -> int:
        return len(self.small_channels) - 1


class BasicBlock(nn.Module):
    def __init__(self, in_channels: Iterable[int], mid_channels: Iterable[int], out_channels: Iterable[int], stride=1) -> None:
        super().__init__()
        self.conv1 = fm.Conv2d(
            in_channels, mid_channels,
            kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = fm.BatchNorm2d(mid_channels)

        self.conv2 = fm.Conv2d(mid_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = fm.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                fm.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                fm.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        return F.relu(out)


class Resnet(ShardModel):
    def __init__(self, config: ResnetConfig) -> None:
        super().__init__(config)
        self.build_net(config)

    def build_net(self, config: ResnetConfig) -> None:
        self.levels = len(config.small_channels)

        self.conv1 = fm.Conv2d(
            [3] * self.levels,
            config.small_channels,
            kernel_size=3, padding=1, bias=False)
        self.bn1 = fm.BatchNorm2d(config.small_channels)

        self.layer1 = self._make_base_layer(
            config.small_channels, config.small_channels, config.num_blocks[0])
        self.layer2 = self._make_base_layer(
            config.small_channels, config.mid_channels, config.num_blocks[1], stride=2)
        self.layer3 = self._make_base_layer(
            config.mid_channels, config.large_channels, config.num_blocks[2], stride=2)

        self.fc = fm.LinearSelect(
            config.large_channels,
            [config.num_classes] * self.levels)

        self.set_level_use(self.levels - 1)
        self.level = self.levels - 1

        if config.prebuilt:
            if config.prebuilt_level < 0:
                self.set_level_use(self.max_level() + 1 +
                                   config.prebuilt_level)
            else:
                self.set_level_use(config.prebuilt_level)
            prebuild_config = (config.num_classes, config.num_blocks, (
                config.small_channels[self.current_level()], config.mid_channels[self.current_level()], config.large_channels[self.current_level()]))
            if prebuild_config not in KNOWN_MODEL_PRETRAINED:
                raise RuntimeError("prebuilt model not found")
            prebuilt = KNOWN_MODEL_PRETRAINED[prebuild_config]()
            utils.shard_model_copy(prebuilt, self)

    def _make_base_layer(self, in_channels, out_channels, blocks, stride=1) -> nn.Sequential:
        layers = []
        layers.append(BasicBlock(
            in_channels, out_channels, out_channels, stride))
        in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(in_channels, out_channels, out_channels))
        return nn.Sequential(*layers)

    @staticmethod
    def base_type() -> type[nn.Module]:
        return networks.resnet.Resnet

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return self.levels - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
