from typing import Iterable, Dict, Callable, Hashable
import dataclasses

from torch import nn
import torch.nn.functional as F
import torch

import utils
from networks.config import ModelConfig

# basic implementation from https://github.com/chenyaofo/pytorch-cifar-models


@dataclasses.dataclass
class ResnetConfig(ModelConfig):
    num_blocks: Iterable[int] = (3, 3, 3)
    num_classes: int = 10
    small_channels: Iterable[int] = 16
    mid_channels: Iterable[int] = 32
    large_channels: Iterable[int] = 64
    prebuilt: bool = True

    def make_model(self) -> 'Resnet':
        return Resnet(self)

    def no_prebuilt(self):
        self.prebuilt = False
        return self


KNOWN_MODEL_PRETRAINED: Dict[Hashable, Callable[[], nn.Module]] = {
    (10, (3, 3, 3), (16, 32, 64)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True, verbose=False),
    (10, (5, 5, 5), (16, 32, 64)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True, verbose=False),
    (10, (7, 7, 7), (16, 32, 64)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet44", pretrained=True, verbose=False),
    (10, (9, 9, 9), (16, 32, 64)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True, verbose=False),
    (100, (3, 3, 3), (16, 32, 64)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True, verbose=False),
    (100, (5, 5, 5), (16, 32, 64)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet32", pretrained=True, verbose=False),
    (100, (7, 7, 7), (16, 32, 64)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet44", pretrained=True, verbose=False),
    (100, (9, 9, 9), (16, 32, 64)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True, verbose=False),
}


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, stride=1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels,
            kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
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


class Resnet(nn.Module):
    def __init__(self, config: ResnetConfig) -> None:
        super().__init__()
        self.build_net(config)

    def build_net(self, config: ResnetConfig) -> None:
        self.conv1 = nn.Conv2d(
            3, config.small_channels,
            kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(config.small_channels)

        self.layer1 = self._make_base_layer(
            config.small_channels, config.small_channels, config.num_blocks[0])
        self.layer2 = self._make_base_layer(
            config.small_channels, config.mid_channels, config.num_blocks[1], stride=2)
        self.layer3 = self._make_base_layer(
            config.mid_channels, config.large_channels, config.num_blocks[2], stride=2)

        self.fc = utils.LinearHead(
            config.large_channels, config.num_classes)

        if config.prebuilt:
            prebuild_config = (
                config.num_classes, config.num_blocks,
                (config.small_channels, config.mid_channels, config.large_channels))
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
