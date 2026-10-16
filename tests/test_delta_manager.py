import unittest
import sys
import os

import torch
from torch import nn

import io

import networks.level_delta_utils as levels
import networks.shard_resnet as shard_resnet
import networks.shard_vgg as shard_vgg
import networks.shard_vit as shard_vit
import networks.config
import utils


torch.manual_seed(seed=0)


def randomize_params(net):
    for p in net.parameters():
        p.data = torch.rand(*p.shape).to(p.data) / 100
    return net


def make_randomized(func):
    def f(*args, **kwargs):
        net = func(*args, **kwargs)
        return randomize_params(net)
    return f


class DeltaManagerTestMixin:
    """Shared checks; concrete TestCase classes below supply the model config and input.
    (Not named Test* so pytest does not try to run the abstract version.)"""

    @staticmethod
    def check_equiv(a, b):
        if torch.isclose(a, b, rtol=1e-3, atol=1e-5).all():
            return True
        print(torch.isclose(a, b), file=sys.stderr)
        print(torch.abs(a - b))
        return False

    def get_shard_config(self) -> networks.config.ShardModelConfig:
        raise NotImplementedError()

    def make_input(self) -> torch.Tensor:
        raise NotImplementedError()

    def test_deltas(self):
        aconfig = self.get_shard_config()
        model = aconfig.make_model().to(utils.get_device())
        randomize_params(model)

        model.set_level_use(model.max_level())
        delta_manager = levels.InMemoryDeltaManager(model, model.max_level())
        delta_manager.set_managed_model(
            delta_manager.managed_model().to(utils.get_device()))
        reg_model = delta_manager.managed_model()

        model.eval()
        reg_model.eval()

        x = self.make_input().to(utils.get_device())
        self.assertTrue(self.check_equiv(model(x), reg_model(x)))

        for i in range(model.max_level() - 1, -1, -1):
            model.set_level_use(i)
            delta_manager.move_to(i)
            self.assertTrue(self.check_equiv(model(x), reg_model(x)))

        for i in range(1, model.max_level() + 1):
            model.set_level_use(i)
            delta_manager.move_to(i)
            self.assertTrue(self.check_equiv(model(x), reg_model(x)))

    def test_deltas_file_deltas(self):
        config = self.get_shard_config()
        model = config.make_model().to(utils.get_device())
        model.eval()

        f = io.BytesIO()
        levels.FileDeltaManager.make_delta_file(f, model, starting_level=0)

        x = self.make_input().to(utils.get_device())
        reg_config = config.create_base_config(0).no_prebuilt()

        f.seek(0)
        manager = levels.FileDeltaManager(f, reg_config)
        manager.set_managed_model(
            manager.managed_model().to(utils.get_device()))
        reg_model = manager.managed_model()
        reg_model.eval()
        manager.move_to(model.current_level())

        self.assertTrue(self.check_equiv(model(x), reg_model(x)))

        for i in range(model.max_level() - 1, -1, -1):
            model.set_level_use(i)
            manager.move_to(i)
            self.assertTrue(self.check_equiv(model(x), reg_model(x)))

        for i in range(1, model.max_level() + 1):
            model.set_level_use(i)
            manager.move_to(i)
            self.assertTrue(self.check_equiv(model(x), reg_model(x)))


class TestDeltaResnet(DeltaManagerTestMixin, unittest.TestCase):
    def get_shard_config(self) -> networks.config.ShardModelConfig:
        return shard_resnet.ResnetConfig()

    def make_input(self) -> torch.Tensor:
        return torch.rand(1, 3, 32, 32)


class TestDeltaViT(DeltaManagerTestMixin, unittest.TestCase):
    def get_shard_config(self) -> networks.config.ShardModelConfig:
        return shard_vit.ViTConfig()

    def make_input(self) -> torch.Tensor:
        return torch.rand(1, 3, 224, 224)


class TestDeltaVGG(DeltaManagerTestMixin, unittest.TestCase):
    def get_shard_config(self) -> networks.config.ShardModelConfig:
        return shard_vgg.VGGConfig().no_prebuilt()

    def make_input(self) -> torch.Tensor:
        return torch.rand(1, 3, 32, 32)
