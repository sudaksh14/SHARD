from __future__ import annotations

from typing import Any, TypeVar, Generic, Callable, overload, Union
from functools import partial
import copy

from torch import nn

import torch

T = TypeVar('T')


class LevelDelta(Generic[T]):
    def __init__(self, delta):
        self.delta: T = delta

    def verify_format(self) -> bool:
        return _verify_level_delta(self)

    def map_tensors(self, func: Callable[[torch.Tensor], torch.Tensor]) -> 'LevelDelta':
        return _map_level_delta_tensors(func, self)

    def str_structure(self) -> str:
        return _str_structure_level_delta(self)

    @overload
    def to(self, dtype: torch.dtype, *args, **kwargs) -> 'LevelDelta': ...

    @overload
    def to(self, device: Union[str, torch.device, int], *args, **kwargs) -> 'LevelDelta':
        ...

    @overload
    def to(self, other: torch.Tensor, *args, **kwargs) -> 'LevelDelta': ...

    def to(self, *args, **kwargs) -> 'LevelDelta':
        return self.map_tensors(lambda t: t.to(*args, **kwargs))

    def clone(self) -> 'LevelDelta':
        return self.map_tensors(lambda t: t.clone())

    def detach(self) -> 'LevelDelta':
        return self.map_tensors(lambda t: t.detach())

    def cpu(self):
        return self.map_tensors(lambda t: t.cpu())

    def apply(self, module: nn.Module):
        raise NotImplemented()


class UpDelta(LevelDelta[T]):
    def __init__(self, delta):
        super().__init__(delta)

    def apply(self, module):
        return LevelDeltas.apply_level_delta_up(module, self)


class DownDelta(LevelDelta[T]):
    def __init__(self, delta):
        super().__init__(delta)

    def apply(self, module):
        return LevelDeltas.apply_level_delta_down(module, self)


# torch < 2.4 (e.g. JetPack 5 builds) has no safe-globals registry
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([UpDelta, DownDelta, LevelDelta])


class Module(nn.Module):
    def set_level_use(self, level: int) -> None:
        """
        Sets the level the shardbible module will use.
        """
        raise NotImplementedError()

    def __init_subclass__(cls):
        super().__init_subclass__()
        LevelDeltas.register(cls)

    def current_level(self) -> int:
        """
        Queries the level currently used by the Module.
        """
        raise NotImplementedError()

    def max_level(self) -> int:
        """
        Queries the highest level this module can be set to.
        """
        raise NotImplementedError()

    def copy_to_base(self, dest: nn.Module) -> None:
        """
        Copies the current level to a regular module.

        The type of regular module can be queried using Module.base_type()

        Note that the regular modules dimensions have to be compatible with
        the dimensions of the current level.
        """
        raise NotImplementedError()

    def load_from_base(self, src: nn.Module) -> None:
        """
        Copies a regular module to the current level.

        The type of regular module can be queried using Module.base_type()

        Note that the regular modules dimensions have to be compatible with
        the dimensions of the current level.

        Note that due to the nature of shared weights this action may also affect other levels.
        """
        raise NotImplementedError()

    @torch.no_grad()
    def _make_reg_layer(self) -> nn.Module:
        raise NotImplementedError()

    def make_base_copy(self) -> nn.Module:
        """
        Makes a new regular layer with the current layer copied into it.
        """
        reg = self._make_reg_layer()
        self.copy_to_base(reg)
        reg.train(self.training)
        param = next(self.parameters(), None)
        if param is not None:
            reg.to(param)
        return reg

    @staticmethod
    def base_type() -> type[nn.Module]:
        """
        Queries the type of regular module this SHARD module is based on.
        """
        raise NotImplementedError()

    def export_level_delta(self) -> tuple[DownDelta, UpDelta]:
        """
        This function extracts part of the SHARD layer into a down delta and an up delta. If
        you have a regular layer that is equivalent to one below the current
        level of the SHARD layer. You can apply the up delta to it to make it equivalent
        to the current level. Similarly if you have a regular layer equivalent to one above the
        current level of the SHARD layer, you can apply a delta down to make it equivalent
        to the current level.

        The functions to apply these deltas are apply_level_delta_down, and apply_level_delta_up
        """
        raise NotImplementedError()

    @staticmethod
    def apply_level_delta_down(module: nn.Module, level_delta: DownDelta) -> None:
        """
        Takes regular layer and applies a delta down to it.
        """
        raise NotImplementedError()

    @staticmethod
    def apply_level_delta_up(module: nn.Module, level_delta: UpDelta) -> None:
        """
        Takes regular layer and applies a delta up to it.
        """
        raise NotImplementedError()

    @staticmethod
    def register_self(cls: type['Module']) -> type['Module']:
        return LevelDeltas.register(cls)


class LevelDeltas:
    registered: dict[type[nn.Module], Module] = dict()

    @staticmethod
    def register(cls: type[Module]) -> type[Module]:
        try:
            base_type = cls.base_type()
        except NotImplementedError:
            return
        # assert (not __class__.is_registered(base_type))
        assert (not __class__.is_registered(base_type)), (
            f"Duplicate registration for base_type: {base_type.__name__}, "
            f"attempted by: {cls.__name__}"
        )
        __class__.registered[base_type] = cls

    @staticmethod
    def is_registered(module_type: type[nn.Module]):
        return module_type in __class__.registered

    @staticmethod
    def apply_level_delta_down(module: nn.Module, level_delta: DownDelta) -> None:
        """
        Takes regular layer and applies a delta down to it.
        """
        if not __class__.is_registered(type(module)):
            raise KeyError()
        return __class__.registered[type(module)].apply_level_delta_down(module, level_delta)

    @staticmethod
    def apply_level_delta_up(module: nn.Module, level_delta: UpDelta) -> None:
        """
        Takes regular layer and applies a delta up to it.
        """
        if not __class__.is_registered(type(module)):
            raise KeyError()
        return __class__.registered[type(module)].apply_level_delta_up(module, level_delta)


def _verify_level_delta(delta: LevelDelta) -> bool:
    if isinstance(delta, LevelDelta):
        return _verify_level_delta(delta.delta)
    elif isinstance(delta, tuple):
        return all(map(_verify_level_delta, delta))
    elif isinstance(delta, int):
        return True
    elif isinstance(delta, torch.Tensor):
        return True
    elif delta is None:
        return True
    return False


def _map_level_delta_tensors(func: Callable[[torch.Tensor], torch.Tensor], delta: LevelDelta) -> LevelDelta:
    if isinstance(delta, LevelDelta):
        cpy = copy.copy(delta)
        cpy.delta = _map_level_delta_tensors(func, delta.delta)
        return cpy
    elif isinstance(delta, tuple):
        return tuple(map(partial(_map_level_delta_tensors, func), delta))
    elif isinstance(delta, int):
        return delta
    elif isinstance(delta, torch.Tensor):
        return func(delta)
    elif delta is None:
        return None
    raise RuntimeError("Level delta is not of valid format")


def _str_structure_level_delta(delta):
    if isinstance(delta, LevelDelta):
        return f"Delta({_str_structure_level_delta(delta.delta)})"
    elif isinstance(delta, tuple):
        lst = ", ".join(map(_str_structure_level_delta, delta))
        return f"({lst})"
    elif isinstance(delta, int):
        return str(int)
    elif isinstance(delta, torch.Tensor):
        return str(delta.size())
    elif delta is None:
        return "None"
    else:
        return "ERROR"
