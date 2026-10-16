from __future__ import annotations

from typing import Any, Iterable

from torch import nn
import torch

from shard_modules.module import Module, DownDelta, UpDelta
import networks.modules as vmod


class ClassTokenLayer(Module):
    def __init__(self, hidden_dim: Iterable[int]):
        super().__init__()

        def is_positive(x): return x > 0
        assert (all(map(is_positive, hidden_dim)))
        assert (max(hidden_dim) == hidden_dim[-1])

        self.hidden_dims = hidden_dim
        self.token = nn.Parameter(torch.zeros(1, 1, hidden_dim[-1]))
        self.level = self.max_level()

    def forward(self, x, n):
        batch_class_token = self.token[
            :, :, :self.hidden_dims[self.level]].expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        return x

    def set_level_use(self, level: int) -> None:
        self.level = level

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return len(self.hidden_dims) - 1

    @staticmethod
    def base_type() -> type[nn.Module]:
        return vmod.ClassTokenLayer

    @torch.no_grad()
    def copy_to_base(self, dest: vmod.ClassTokenLayer) -> None:
        dest.token.data = self.token.data[:, :, :self.hidden_dims[self.level]]

    @torch.no_grad()
    def load_from_base(self, src: vmod.ClassTokenLayer) -> None:
        self.token.data[:, :, :self.hidden_dims[self.level]] = src.token.data

    def _make_reg_layer(self) -> nn.Module:
        return vmod.ClassTokenLayer(self.hidden_dims[self.level])

    @torch.no_grad()
    def export_level_delta(self) -> tuple[DownDelta[int], UpDelta[torch.Tensor]]:
        return (
            DownDelta(self.hidden_dims[self.level]),
            UpDelta(self.token.data[
                :, :,
                self.hidden_dims[self.level - 1]:self.hidden_dims[self.level]
            ])
        )

    @staticmethod
    @torch.no_grad()
    def apply_level_delta_down(model: vmod.ClassTokenLayer, level_delta: DownDelta[int]) -> None:
        model.token.data = model.token.data[:, :, :level_delta.delta].to(
            model.token.data)

    @staticmethod
    @torch.no_grad()
    def apply_level_delta_up(model: vmod.ClassTokenLayer, level_delta: UpDelta[torch.Tensor]) -> None:
        model.token.data = torch.cat(
            [model.token.data, level_delta.delta.to(model.token.data)], dim=2)
