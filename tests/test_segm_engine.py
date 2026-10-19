"""CPU test of the segmentation training loop (segm/engine_shard.py) with a stub model: all levels, widest first."""
import contextlib
import importlib.util
import unittest

import torch
from torch import nn

import segm.utils.torch as ptu

import sys
from unittest import mock

for _pkg in ("mmseg", "mmcv", "cityscapesscripts"):   # not used by the training loop; stub for import only
    if importlib.util.find_spec(_pkg) is None:
        sys.modules[_pkg] = mock.MagicMock(__path__=[])
        for _sub in ("core", "datasets", "datasets.pipelines", "utils", "runner", "parallel"):
            sys.modules[f"{_pkg}.{_sub}"] = mock.MagicMock()

try:
    from segm.engine_shard import train_one_epoch
except Exception as e:  # segm extras (mmcv / mmseg) not installed
    train_one_epoch = None
    print("segm import failed:", repr(e))
    _err = e


class _Encoder:
    def max_level(self):
        return 3


class _Stub(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()
        self.heads = nn.ModuleList([nn.Conv2d(3, 5, 1) for _ in range(4)])
        self.levels = []

    def forward(self, im, level):
        self.levels.append(level)
        return self.heads[level](im)


class _Loader:
    def __init__(self):
        self.data = [{"im": torch.randn(2, 3, 8, 8), "segmentation": torch.randint(0, 5, (2, 8, 8))}]

    def set_epoch(self, e):
        pass

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)


@unittest.skipIf(train_one_epoch is None, "segm dependencies (mmcv/mmseg) not installed")
class TestSegmEngine(unittest.TestCase):
    def test_all_levels_each_step_and_params_update(self):
        ptu.device = torch.device("cpu")
        model = _Stub()
        before = [p.detach().clone() for p in model.parameters()]
        opt = torch.optim.SGD([{"params": model.heads[:2].parameters()}, {"params": model.heads[2:].parameters()}], lr=0.1)
        for transfer in ("progressive", "US"):
            model.levels.clear()
            train_one_epoch(model, _Loader(), opt, None, 0, contextlib.nullcontext, None, transfer_type=transfer)
            self.assertEqual(model.levels, [3, 2, 1, 0])
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))


if __name__ == "__main__":
    unittest.main()
