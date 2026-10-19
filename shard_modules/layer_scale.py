from __future__ import annotations

from typing import Iterable
from torch import nn
import torch
from shard_modules.module import Module, DownDelta, UpDelta
import networks.modules as vmod


class LayerScale(Module):
    def __init__(self, hidden_dim: Iterable[int], init_value: float = 1e-6):
        super().__init__()

        def is_positive(x): return x > 0
        assert all(map(is_positive, hidden_dim))
        assert max(hidden_dim) == hidden_dim[-1]

        self.hidden_dims = hidden_dim
        self.level = self.max_level()
        self.init_value = init_value

        # Learnable scaling parameters
        self.gamma = nn.Parameter(torch.ones(hidden_dim[-1]) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply per-channel scaling
        gamma = self.gamma[:self.hidden_dims[self.level]]
        return x * gamma

    def set_level_use(self, level: int) -> None:
        self.level = level

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return len(self.hidden_dims) - 1

    @staticmethod
    def base_type() -> type[nn.Module]:
        return vmod.LayerScale

    @torch.no_grad()
    def copy_to_base(self, dest: vmod.LayerScale) -> None:
        dest.gamma.data = self.gamma.data[:self.hidden_dims[self.level]]

    @torch.no_grad()
    def load_from_base(self, src: vmod.LayerScale) -> None:
        self.gamma.data[:self.hidden_dims[self.level]] = src.gamma.data

    def _make_reg_layer(self) -> nn.Module:
        return vmod.LayerScale(self.hidden_dims[self.level], self.init_value)

    @torch.no_grad()
    def export_level_delta(self) -> tuple[DownDelta[int], UpDelta[torch.Tensor]]:
        return (
            DownDelta(self.hidden_dims[self.level]),
            UpDelta(
                self.gamma.data[self.hidden_dims[self.level - 1]:self.hidden_dims[self.level]]
            ),
        )

    @staticmethod
    @torch.no_grad()
    def apply_level_delta_down(model: vmod.LayerScale, level_delta: DownDelta[int]) -> None:
        model.gamma.data = model.gamma.data[:level_delta.delta].to(model.gamma.data)

    @staticmethod
    @torch.no_grad()
    def apply_level_delta_up(model: vmod.LayerScale, level_delta: UpDelta[torch.Tensor]) -> None:
        model.gamma.data = torch.cat(
            [model.gamma.data, level_delta.delta.to(model.gamma.data)])
