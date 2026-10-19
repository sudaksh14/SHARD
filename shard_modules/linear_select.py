from __future__ import annotations

from torch import nn
import torch

from shard_modules.shard_select import AdaptSelect
from networks.modules import LinearHead


class LinearSelect(AdaptSelect):
    def __init__(self, in_sizes, out_sizes, *args, **kwargs) -> None:
        self._args = args
        self._kwargs = kwargs
        self._in_sizes = in_sizes
        self._out_sizes = out_sizes
        layers = [
            nn.Linear(ins, outs, *args, **kwargs) for ins, outs in zip(in_sizes, out_sizes)
        ]
        super().__init__(layers)

    @staticmethod
    def base_type() -> type[nn.Linear]:
        return LinearHead

    def _make_reg_layer(self):
        return nn.Linear(
            self._in_sizes[self.current_level()], self._out_sizes[self.current_level()], *self._args, **self._kwargs)
