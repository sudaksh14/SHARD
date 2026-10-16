import unittest
import sys

from torch import nn
import torch

from shard_modules.module import Module
import shard_modules as fm
import utils

from networks import shard_vgg, shard_vit, shard_resnet, vgg, vit, resnet


torch.manual_seed(seed=0)


class LayerTester:
    def make_shard_module(self) -> Module:
        raise NotImplementedError()

    def make_reg_module(self, level) -> nn.Module:
        raise NotImplementedError()

    def make_input(self, level) -> torch.Tensor:
        raise NotImplementedError()

    def augment_for_reg(self, x, level) -> torch.Tensor:
        return x

    def augment_reg_output(self, y, level) -> torch.Tensor:
        return y

    @staticmethod
    def check_equiv(a, b):
        if torch.isclose(a, b).all():
            return True
        print(torch.isclose(a, b), file=sys.stderr)
        print(torch.abs(a - b))
        return False

    def test_basic_forward(self):
        layer = self.make_shard_module()
        for i in range(layer.max_level() + 1):
            x = self.make_input(i)
            layer.set_level_use(i)
            layer(*x)

    def test_copy_to_base(self):
        layer = self.make_shard_module()
        for i in range(layer.max_level() + 1):
            layer.set_level_use(i)
            reg_layer = self.make_reg_module(i)
            layer.copy_to_base(reg_layer)
            x = self.make_input(i)
            self.assertTrue(self.check_equiv(layer(*x), reg_layer(*x)))

    def test_load_from_base(self):
        layer = self.make_shard_module()
        for i in range(layer.max_level() + 1):
            layer.set_level_use(i)
            reg_layer = self.make_reg_module(i)
            layer.load_from_base(reg_layer)
            x = self.make_input(i)
            self.assertTrue(self.check_equiv(layer(*x), reg_layer(*x)))

    def test_make_base_copy(self):
        layer = self.make_shard_module()
        for i in range(layer.max_level() + 1):
            layer.set_level_use(i)
            cpy = layer.make_base_copy()

            reg_layer = self.make_reg_module(i)
            reg_layer.load_state_dict(cpy.state_dict())

            x = self.make_input(i)
            self.assertTrue(self.check_equiv(layer(*x), reg_layer(*x)))

    def test_level_deltas(self):
        layer = self.make_shard_module()
        layer.set_level_use(layer.max_level())
        reg_layer = self.make_reg_module(layer.max_level())
        layer.copy_to_base(reg_layer)

        x = self.make_input(layer.max_level())
        self.assertTrue(self.check_equiv(layer(*x), reg_layer(*x)))

        for i in range(layer.max_level() - 1, -1, -1):
            layer.set_level_use(i)
            delta_down, delta_up = layer.export_level_delta()
            self.assertTrue(delta_down.verify_format(),
                            delta_down.str_structure())
            self.assertTrue(delta_up.verify_format(), delta_up.str_structure())
            layer.apply_level_delta_down(reg_layer, delta_down)
            x = self.make_input(i)
            self.assertTrue(self.check_equiv(layer(*x), reg_layer(*x)))

        for i in range(1, layer.max_level() + 1):
            layer.set_level_use(i)
            delta_down, delta_up = layer.export_level_delta()
            self.assertTrue(delta_down.verify_format(),
                            delta_down.str_structure())
            self.assertTrue(delta_up.verify_format(), delta_up.str_structure())
            layer.apply_level_delta_up(reg_layer, delta_up)
            x = self.make_input(i)
            self.assertTrue(self.check_equiv(layer(*x), reg_layer(*x)))

    def test_level_deltas_LevelDeltas(self):
        layer = self.make_shard_module()
        layer.set_level_use(layer.max_level())
        reg_layer = self.make_reg_module(layer.max_level())
        layer.copy_to_base(reg_layer)

        x = self.make_input(layer.max_level())
        self.assertTrue(self.check_equiv(layer(*x), reg_layer(*x)))

        for i in range(layer.max_level() - 1, -1, -1):
            layer.set_level_use(i)
            delta_down, delta_up = layer.export_level_delta()
            assert (delta_down.verify_format())
            assert (delta_up.verify_format())
            delta_down.apply(reg_layer)
            x = self.make_input(i)
            self.assertTrue(self.check_equiv(layer(*x), reg_layer(*x)))

        delta_down, delta_up = layer.export_level_delta()

        for i in range(1, layer.max_level() + 1):
            layer.set_level_use(i)
            delta_down, delta_up = layer.export_level_delta()
            assert (delta_down.verify_format())
            assert (delta_up.verify_format())
            delta_up.apply(reg_layer)
            x = self.make_input(i)
            self.assertTrue(self.check_equiv(layer(*x), reg_layer(*x)))

        delta_down, delta_up = layer.export_level_delta()


def randomize_params(net, factor=1.0):
    for p in net.parameters():
        p.data = torch.rand(*p.shape).to(p.data) * factor
    net.eval()
    return net


def make_randomized(func):
    def f(*args, **kwargs):
        net = func(*args, **kwargs)
        return randomize_params(net)
    return f


def make_randomized_f(factor):
    def make_randomized(func):
        def f(*args, **kwargs):
            net = func(*args, **kwargs)
            return randomize_params(net, factor)
        return f
    return make_randomized


class TestConv2d(LayerTester, unittest.TestCase):
    IN_CHANNELS = [5, 9, 12]
    OUT_CHANNELS = [5, 9, 12]

    @make_randomized
    def make_shard_module(self) -> Module:
        return fm.conv2d.Conv2d(
            self.IN_CHANNELS, self.OUT_CHANNELS, kernel_size=3, bias=True)

    @make_randomized
    def make_reg_module(self, level) -> nn.Module:
        return nn.Conv2d(
            self.IN_CHANNELS[level], self.OUT_CHANNELS[level], kernel_size=3, bias=True)

    def make_input(self, level) -> torch.Tensor:
        return torch.rand(10, self.IN_CHANNELS[level], 100, 100),


class SelfAttention(nn.MultiheadAttention):
    def forward(self, x):
        return super().forward(x, x, x)[0]


fm.LevelDeltas.registered[SelfAttention] = fm.SelfAttention


class TestAttention(LayerTester, unittest.TestCase):
    TOKEN_SIZE = [25, 30, 56, 100]
    HEADS = [5, 6, 8, 10]

    @make_randomized
    def make_shard_module(self) -> Module:
        return fm.SelfAttention(self.TOKEN_SIZE, self.HEADS)

    @make_randomized
    def make_reg_module(self, level) -> nn.Module:
        return SelfAttention(
            self.TOKEN_SIZE[level],
            self.HEADS[level],
            batch_first=True)

    def make_input(self, level) -> torch.Tensor:
        return torch.rand((10, 20, self.TOKEN_SIZE[level])),


class TestLinear(LayerTester, unittest.TestCase):
    IN_SIZE = [10, 20, 25, 40]
    OUT_SIZE = [5, 15, 30, 50]

    @make_randomized
    def make_shard_module(self) -> Module:
        return fm.Linear(self.IN_SIZE, self.OUT_SIZE)

    @make_randomized
    def make_reg_module(self, level) -> nn.Module:
        return nn.Linear(self.IN_SIZE[level], self.OUT_SIZE[level])

    def make_input(self, level) -> torch.Tensor:
        return torch.rand((10, 100, self.IN_SIZE[level])),


class TestLayerNorm(LayerTester, unittest.TestCase):
    TOKEN_SIZE = [25, 30, 56, 100]

    @make_randomized
    def make_shard_module(self):
        return fm.LayerNorm(self.TOKEN_SIZE, eps=1e-6)

    @make_randomized
    def make_reg_module(self, level):
        return nn.LayerNorm(self.TOKEN_SIZE[level], eps=1e-6)

    def make_input(self, level):
        return torch.rand(10, 100, self.TOKEN_SIZE[level]),


class TestPosEmbedding(LayerTester, unittest.TestCase):
    TOKEN_SIZE = [25, 30, 56, 100]

    @make_randomized
    def make_shard_module(self):
        return fm.PosEmbeddingLayer(100, self.TOKEN_SIZE)

    @make_randomized
    def make_reg_module(self, level):
        return utils.PosEmbeddingLayer(100, self.TOKEN_SIZE[level])

    def make_input(self, level):
        return torch.rand(10, 100, self.TOKEN_SIZE[level]),


class TestLinearSelect(LayerTester, unittest.TestCase):
    IN_SIZE = [10, 20, 25, 40]
    OUT_SIZE = [5, 15, 30, 50]

    @make_randomized
    def make_shard_module(self) -> Module:
        return fm.LinearSelect(self.IN_SIZE, self.OUT_SIZE)

    @make_randomized
    def make_reg_module(self, level) -> nn.Module:
        return utils.LinearHead(self.IN_SIZE[level], self.OUT_SIZE[level])

    def make_input(self, level) -> torch.Tensor:
        return torch.rand((10, 100, self.IN_SIZE[level])),


class TestClassToken(LayerTester, unittest.TestCase):
    TOKEN_SIZE = [25, 30, 56, 100]

    @make_randomized
    def make_shard_module(self):
        return fm.ClassTokenLayer(self.TOKEN_SIZE)

    @make_randomized
    def make_reg_module(self, level):
        return utils.ClassTokenLayer(self.TOKEN_SIZE[level])

    def make_input(self, level):
        return torch.rand(10, 100, self.TOKEN_SIZE[level]), 10


class TestBatchnorm2d(LayerTester, unittest.TestCase):
    CHANNELS = [5, 9, 12]

    @make_randomized
    def make_shard_module(self) -> Module:
        return fm.BatchNorm2d(self.CHANNELS)

    @make_randomized
    def make_reg_module(self, level) -> nn.Module:
        return nn.BatchNorm2d(self.CHANNELS[level])

    def make_input(self, level) -> torch.Tensor:
        return torch.rand(1, self.CHANNELS[level], 5, 5),


class TestResnet(LayerTester, unittest.TestCase):
    @staticmethod
    def check_equiv(a, b):
        if torch.isclose(a, b, rtol=1e-4).all():
            return True
        print(torch.isclose(a, b), file=sys.stderr)
        print(torch.abs(a - b))
        return False

    @make_randomized
    def make_shard_module(self) -> Module:
        return shard_resnet.ResnetConfig().make_model().to(utils.get_device())

    @make_randomized
    def make_reg_module(self, level) -> nn.Module:
        return shard_resnet.ResnetConfig().create_base_config(level).no_prebuilt().make_model().to(utils.get_device())

    def make_input(self, level) -> torch.Tensor:
        return torch.rand(1, 3, 32, 32).to(utils.get_device()),


class TestVGG(LayerTester, unittest.TestCase):
    @staticmethod
    def check_equiv(a, b):
        if torch.isclose(a, b, rtol=1e-4).all():
            return True
        print(torch.isclose(a, b), file=sys.stderr)
        print(torch.abs(a - b))

    @make_randomized_f(0.01)
    def make_shard_module(self) -> Module:
        return shard_vgg.VGGConfig().make_model().to(utils.get_device())

    @make_randomized_f(0.01)
    def make_reg_module(self, level) -> nn.Module:
        return shard_vgg.VGGConfig().create_base_config(level).no_prebuilt().make_model().to(utils.get_device())

    def make_input(self, level) -> torch.Tensor:
        return torch.rand(1, 3, 32, 32).to(utils.get_device()),


class TestViT(LayerTester, unittest.TestCase):
    @staticmethod
    def check_equiv(a, b):
        if torch.isclose(a, b, rtol=1e-4).all():
            return True
        print(torch.isclose(a, b), file=sys.stderr)
        print(torch.abs(a - b))

    @make_randomized_f(0.01)
    def make_shard_module(self) -> Module:
        return shard_vit.ViTConfig().make_model().to(utils.get_device())

    @make_randomized_f(0.01)
    def make_reg_module(self, level) -> nn.Module:
        return shard_vit.ViTConfig().create_base_config(level).no_prebuilt().make_model().to(utils.get_device())

    def make_input(self, level) -> torch.Tensor:
        return torch.rand(1, 3, 224, 224).to(utils.get_device()),
