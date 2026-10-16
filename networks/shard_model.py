from __future__ import annotations

from typing import Any

from torch import nn
import torch

import shard_modules as fm
from networks import config


class ShardModel(fm.Module):
    """
    Defines an interface followed by all SHARD models.
    """

    def __init__(self, config: config.ShardModelConfig):
        super().__init__()
        self.config = config

    def set_level_use(self, level) -> None:
        self.level = level
        last_set = None
        for _, module in self.named_modules():
            if module is self:
                continue
            if self._is_module_in(module, last_set):
                continue
            if isinstance(module, fm.Module):
                module.set_level_use(level)
                last_set = module

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return self.config.max_level()

    @staticmethod
    def _is_module_in(child: nn.Module, parent: nn.Module):
        if parent is None:
            return False
        return child in parent.modules()

    @staticmethod
    def shard_copy(src, dest, verbose=0):
        # print(type(src), type(dest))
        # print("src length:", len(list(src.named_modules())))
        # print("dest length:", len(list(dest.named_modules())))
              
        # for src_name, src_module in src.named_modules():
        #     print(src_name)
        # for dest_name, dest_module in dest.named_modules():
        #     print(dest_name)
        # exit()

        dest_iter = iter(dest.named_modules())

        last_copied_from = None
        last_copied_to: nn.Module = None

        for src_name, src_module in src.named_modules():
            if src_module is src:
                continue

            src_is_shard = isinstance(src_module, fm.Module)
            if not src_is_shard and not fm.LevelDeltas.is_registered(type(src_module)):
                continue

            if ShardModel._is_module_in(src_module, last_copied_from):
                if verbose >= 2:
                    print(f"Skip copying layer {src_name}")
                continue

            while True:
                try:
                    dest_name, dest_module = next(dest_iter)
                except StopIteration as e:
                    print(type(e))
                    print(src_module, src_name)
                    raise

                if dest_module is dest:
                    continue

                dest_is_shard = isinstance(dest_module, fm.Module)

                if ShardModel._is_module_in(dest_module, last_copied_to):
                    continue

                if verbose >= 1:
                    print(f"copy from {src_name} to {dest_name}")
                try:
                    if src_is_shard:
                        # print("here")
                        if dest_is_shard:
                            dest_module.load_from_base(
                                src_module.make_base_copy())
                        else:
                            src_module.copy_to_base(dest_module)
                    else:
                        if dest_is_shard:
                            # print("there")
                            dest_module.load_from_base(src_module)
                        else:
                            dest_module.load_state_dict(
                                src_module.state_dict())
                    last_copied_to = dest_module
                    break
                except Exception as e:
                    if verbose >= 2:
                        print(e)

            last_copied_from = src_module

    def copy_to_base(self, dest: nn.Module) -> None:
        self.shard_copy(self, dest)

    def load_from_base(self, src: nn.Module) -> None:
        self.shard_copy(src, self)

    def _make_reg_layer(self):
        return self.config.create_base_config(
            self.current_level()).no_prebuilt().make_model()

    @torch.no_grad()
    def export_level_delta(self) -> tuple[
            fm.DownDelta[tuple[fm.DownDelta, ...]],
            fm.UpDelta[tuple[fm.UpDelta, ...]]]:
        """
        Level deltas on the model level are used for some
        specific parts not covered by the deltas of the modules.
        """
        delta_downs = []
        delta_ups = []
        last_extracted = None
        for module in self.modules():
            if module is self:
                continue
            if self._is_module_in(module, last_extracted):
                continue
            if not isinstance(module, fm.Module):
                continue
            delta_down, delta_up = module.export_level_delta()
            delta_downs.append(delta_down)
            delta_ups.append(delta_up)
            last_extracted = module
        delta_downs = tuple(delta_downs)
        delta_ups = tuple(delta_ups)
        return fm.DownDelta(delta_downs), fm.UpDelta(delta_ups)

    @staticmethod
    def apply_level_delta_down(
            model: nn.Module,
            level_delta: fm.DownDelta[tuple[fm.DownDelta, ...]]) -> None:
        """
        Takes regular layer and applies a delta down to it.
        """
        dest_it = iter(model.modules())
        last_applied = None
        for delta in level_delta.delta:
            while True:
                module = next(dest_it)
                if module is model:
                    continue
                if ShardModel._is_module_in(module, last_applied):
                    continue

                try:
                    delta.apply(module)
                    last_applied = module
                    break
                except KeyError:
                    pass

    @staticmethod
    def apply_level_delta_up(
            model: nn.Module,
            level_delta: fm.UpDelta[tuple[fm.UpDelta, ...]]) -> None:
        """
        Takes regular layer and applies a delta up to it.
        """
        dest_it = iter(model.modules())
        last_applied = None

        for delta in level_delta.delta:
            while True:
                module = next(dest_it)
                if module is model:
                    continue
                if ShardModel._is_module_in(module, last_applied):
                    continue

                try:
                    delta.apply(module)
                    last_applied = module
                    break
                except KeyError:
                    pass
