from __future__ import annotations

from typing import Union, List, Dict, cast, Callable, Hashable
import dataclasses

import torch.nn as nn
import torch

from networks.config import ModelConfig
import utils

# basic implementation from https://github.com/chenyaofo/pytorch-cifar-models


@dataclasses.dataclass
class VGGConfig(ModelConfig):
    version: int = 11
    small_channels: int = 64
    mid_channels: int = 128
    large_channels: int = 256
    max_channels: int = 512
    num_classes: int = 10
    prebuilt: bool = True

    def make_model(self) -> 'VGG':
        return VGG(self)

    def no_prebuilt(self):
        self.prebuilt = False
        return self


LAYER_CONFIGS: Dict[int, Callable[[int, int, int, int], list[Union[int, str]]]] = {
    11: lambda a, b, c, d: [a, 'M', b, 'M', c, c, 'M', d, d, 'M', d, d, 'M'],
    13: lambda a, b, c, d: [a, a, 'M', b, b, 'M', c, c, 'M', d, d, 'M', d, d, 'M'],
    16: lambda a, b, c, d: [a, a, 'M', b, b, 'M', c, c, c, 'M', d, d, d, 'M', d, d, d, 'M'],
    19: lambda a, b, c, d: [a, a, 'M', b, b, 'M', c, c, c, c, 'M', d, d, d, d, 'M', d, d, d, d, 'M'],
}

KNOWN_MODEL_PRETRAINED: Dict[Hashable, Callable[[], nn.Module]] = {
    (10, 11, (64, 128, 256, 512)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg11_bn", pretrained=True, verbose=False),
    (10, 13, (64, 128, 256, 512)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg13_bn", pretrained=True, verbose=False),
    (10, 16, (64, 128, 256, 512)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True, verbose=False),
    (10, 19, (64, 128, 256, 512)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg19_bn", pretrained=True, verbose=False),
    (100, 11, (64, 128, 256, 512)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg11_bn", pretrained=True, verbose=False),
    (100, 13, (64, 128, 256, 512)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg13_bn", pretrained=True, verbose=False),
    (100, 16, (64, 128, 256, 512)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg16_bn", pretrained=True, verbose=False),
    (100, 19, (64, 128, 256, 512)): lambda: torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_vgg19_bn", pretrained=True, verbose=False),
}


class VGG(nn.Module):
    def __init__(self, config: VGGConfig) -> None:
        super().__init__()
        self.features = self.make_layers(LAYER_CONFIGS[config.version](
            config.small_channels, config.mid_channels, config.large_channels, config.max_channels))

        self.classifier = nn.Sequential(
            utils.LinearHead(config.max_channels, config.max_channels),
            nn.ReLU(True),
            nn.Dropout(),
            utils.LinearHead(config.max_channels, config.max_channels),
            nn.ReLU(True),
            nn.Dropout(),
            utils.LinearHead(config.max_channels, config.num_classes),
        )

        if config.prebuilt:
            prebuild_config = (
                config.num_classes, config.version,
                (config.small_channels, config.mid_channels, config.large_channels, config.max_channels))
            if prebuild_config not in KNOWN_MODEL_PRETRAINED:
                raise RuntimeError("prebuilt model not found")
            prebuilt = KNOWN_MODEL_PRETRAINED[prebuild_config]()
            utils.shard_model_copy(prebuilt, self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def make_layers(self, cfg) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
