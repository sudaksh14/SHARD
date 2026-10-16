from __future__ import annotations

from torch import nn
import torch

from shard_modules.shard_select import AdaptSelect
from shard_modules.module import DownDelta, UpDelta


class LayerNorm(AdaptSelect):
    def __init__(self, channels, *args, **kwargs) -> None:
        self._args = args
        self._kwargs = kwargs
        self._channels = channels
        layers = [
            nn.LayerNorm(c, *args, **kwargs) for c in channels
        ]
        super().__init__(layers)

    @staticmethod
    def base_type() -> type[nn.LayerNorm]:
        return nn.LayerNorm

    def _make_reg_layer(self) -> nn.Module:
        return nn.LayerNorm(
            self._channels[self.current_level()], *self._args, **self._kwargs)

    @staticmethod
    @torch.no_grad()
    def apply_level_delta_down(model: nn.LayerNorm, level_delta: DownDelta[tuple[torch.Tensor, torch.Tensor]]):
        AdaptSelect.apply_level_delta_down(model, level_delta)
        model.normalized_shape = level_delta.delta[0][0].shape[0],

    @staticmethod
    @torch.no_grad()
    def apply_level_delta_up(model: nn.LayerNorm, level_delta: UpDelta[tuple[torch.Tensor, torch.Tensor]]):
        AdaptSelect.apply_level_delta_up(model, level_delta)
        model.normalized_shape = level_delta.delta[0][0].shape[0],
