from __future__ import annotations

from typing import Iterable

from torch import nn
import torch

from shard_modules.module import Module, LevelDelta, UpDelta, DownDelta


class AdaptSelect(Module):
    def __init__(self, layers: Iterable[nn.Module]) -> None:
        super().__init__()
        self.layers = layers
        for level, l in enumerate(self.layers):
            self.add_module(f"level{level}", l)
        self.set_level_use(self.max_level())

    def current_layer(self) -> nn.Module:
        return self.layers[self.level]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.current_layer()(x)

    def set_level_use(self, level: int) -> None:
        self.level = level

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return len(self.layers) - 1

    @torch.no_grad()
    def copy_to_base(self, dest: nn.Module) -> None:
        dest.load_state_dict(self.current_layer().state_dict())

    @torch.no_grad()
    def load_from_base(self, src: nn.Module) -> None:
        self.current_layer().load_state_dict(src.state_dict())

    @torch.no_grad()
    def export_level_delta(self) -> tuple[DownDelta[tuple[torch.Tensor, ...]], UpDelta[tuple[torch.Tensor, ...]]]:
        params = self.current_layer().parameters()
        buffers = self.current_layer().buffers()
        deltas = map(lambda t: t.data, params)
        deltas = tuple(deltas), tuple(buffers)
        return DownDelta(deltas), UpDelta(deltas)

    @staticmethod
    @torch.no_grad()
    def _apply_level_delta(model: nn.Module, level_delta: LevelDelta[tuple[torch.Tensor, ...]]):
        params, buffers = level_delta.delta
        for p, src_p in zip(model.parameters(), params):
            p.data = src_p.to(p.data)
        for (name, b), src_b in zip(model.named_buffers(), buffers):
            if b is not None:
                src_b = src_b.to(b)
            setattr(model, name, src_b)

    @staticmethod
    @torch.no_grad()
    def apply_level_delta_down(model: nn.Module, level_delta: DownDelta[tuple[torch.Tensor, ...]]):
        return __class__._apply_level_delta(model, level_delta)

    @staticmethod
    @torch.no_grad()
    def apply_level_delta_up(model: nn.Module, level_delta: UpDelta[tuple[torch.Tensor, ...]]):
        return __class__._apply_level_delta(model, level_delta)
