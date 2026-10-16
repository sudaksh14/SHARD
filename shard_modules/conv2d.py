from __future__ import annotations

from typing import Iterable

from torch import nn
import torch

from shard_modules.module import Module, UpDelta, DownDelta, LevelDelta
import torch.nn.functional as F


class Conv2d(Module):
    def __init__(self, in_sizes: Iterable[int], out_sizes: Iterable[int], *args, **kwargs) -> None:
        super().__init__()

        def is_positive(x): return x > 0
        assert (len(in_sizes) > 0)
        assert (len(in_sizes) == len(out_sizes))
        assert (all(map(is_positive, in_sizes)))
        assert (all(map(is_positive, out_sizes)))
        assert (max(in_sizes) == in_sizes[-1])
        assert (max(out_sizes) == out_sizes[-1])

        self.in_sizes = in_sizes
        self.out_sizes = out_sizes

        self._args = args
        self._kwargs = kwargs

        self.max_in_size = self.in_sizes[-1]
        self.max_out_size = self.out_sizes[-1]

        self.set_level_use(self.max_level())
        self.conv = nn.Conv2d(
            self.max_in_size, self.max_out_size, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = F.pad(x, (0, 0, 0, 0, 0,
        #               self.max_in_size - self.in_sizes[self.level]))
        # x = self.conv(x)
        # x = x[:, :self.out_sizes[self.level]]
        # return x

        weight_part = self.conv.weight[
            :self.out_sizes[self.level],
            :self.in_sizes[self.level]]
        bias_part = None
        if self.conv.bias is not None:
            bias_part = self.conv.bias[:self.out_sizes[self.level]]

        # Copied straight from the torch Conv2d forward method
        if self.conv.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    x, self.conv._reversed_padding_repeated_twice, mode=self.conv.padding_mode
                ),
                weight_part,
                bias_part,
                self.conv.stride,
                (0, 0),
                self.conv.dilation,
                self.conv.groups,
            )
        return F.conv2d(
            x, weight_part, bias_part, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups
        )

    def set_level_use(self, level: int) -> None:
        assert (level >= 0)
        assert (level <= self.max_level())
        self.level = level

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return len(self.in_sizes) - 1

    @staticmethod
    def base_type() -> type[nn.Conv2d]:
        return nn.Conv2d

    @torch.no_grad()
    def copy_to_base(self, dest: nn.Conv2d) -> None:
        dest.weight.data = self.conv.weight.data[:self.out_sizes[self.level],
                                                 :self.in_sizes[self.level]]
        if self.conv.bias is not None:
            dest.bias.data = self.conv.bias.data[:self.out_sizes[self.level]]

    @torch.no_grad()
    def load_from_base(self, src: nn.Conv2d) -> None:
        self.conv.weight.data[:self.out_sizes[self.level],
                              :self.in_sizes[self.level]] = src.weight.data
        if src.bias is not None:
            self.conv.bias.data[:self.out_sizes[self.level]] = src.bias.data

    @torch.no_grad()
    def make_base_copy(self) -> nn.Conv2d:
        conv = nn.Conv2d(
            self.in_sizes[self.level], self.out_sizes[self.level], *self._args, **self._kwargs)
        self.copy_to_base(conv)
        conv.train(self.training)
        return conv

    def _make_reg_layer(self):
        return nn.Conv2d(
            self.in_sizes[self.level], self.out_sizes[self.level], *self._args, **self._kwargs)

    @torch.no_grad()
    def export_level_delta(self) -> tuple[DownDelta[tuple[int, int]], UpDelta[tuple[torch.Tensor, torch.Tensor]]]:
        weights = self.conv.weight.data
        lower_part = weights[:self.out_sizes[self.level],
                             self.in_sizes[self.level-1]:self.in_sizes[self.level], ]
        right_part = weights[
            self.out_sizes[self.level-1]:self.out_sizes[self.level], :self.in_sizes[self.level-1]]
        bias_part = None
        if self.conv.bias is not None:
            bias_part = self.conv.bias.data[
                self.out_sizes[self.level-1]:self.out_sizes[self.level]]
        prune_up = (lower_part, right_part, bias_part)
        prune_down = (self.in_sizes[self.level],
                      self.out_sizes[self.level])
        return DownDelta(prune_down), UpDelta(prune_up)

    @staticmethod
    @torch.no_grad()
    def apply_level_delta_down(model: nn.Conv2d, level_delta: DownDelta[tuple[int, int]]) -> None:
        in_size, out_size = level_delta.delta
        model.weight.data = model.weight.data[:out_size, :in_size].to(
            model.weight.data)
        if model.bias is not None:
            model.bias.data = model.bias.data[:out_size].to(model.bias.data)

    @staticmethod
    @torch.no_grad()
    def apply_level_delta_up(model: nn.Conv2d, level_delta: UpDelta[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> None:
        weights = model.weight.data
        lower_part, right_part, bias_part = level_delta.delta
        out_size, in_size, *_ = weights.size()
        weights = F.pad(
            weights, (0, 0, 0, 0, 0, lower_part.size(1), 0, right_part.size(0)))
        weights[:, in_size:] = lower_part.to(weights)
        weights[out_size:, :in_size] = right_part.to(weights)
        if model.bias is not None:
            model.bias.data = torch.cat(
                [model.bias.data, bias_part.to(model.bias.data)])
        model.weight.data = weights
        model.zero_grad()
