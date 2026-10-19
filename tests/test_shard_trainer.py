"""
CPU tests for the SHARD training step (training/shard_trainer.py) and the experiment registry.

The step must visit every level from the widest to level 0 (paper Algorithm 1) and, with transfer_type='progressive',
distil each level from the next wider one.
"""
import unittest

import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from training.shard_trainer import ShardDistillContext, ShardDistillTrainer
from utils.switching_harness import common


def make_trainer(transfer_type="progressive", training_mode="simple", num_sampled=None):
    cfg = common.tiny_config("shard_deit_v3", levels=4)
    ctx = ShardDistillContext(
        loader_function=None, patience=1, epochs=1, wandb_project_name=None, unittest_mode=True,
        teacher_loader=lambda: nn.Identity(),
        make_optimizer=lambda m: torch.optim.SGD(m.parameters(), lr=1e-3),
        make_scheduler=lambda o: CosineAnnealingLR(o, T_max=1),
        transfer_type=transfer_type, training_mode=training_mode, num_sampled=num_sampled)
    return ShardDistillTrainer(cfg, ctx)


def loader():
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    return DataLoader(TensorDataset(x, y), batch_size=4)


class TestSampleLevels(unittest.TestCase):
    def levels(self, mode, num_sampled, max_level=6):
        return make_trainer(training_mode=mode, num_sampled=num_sampled).sample_levels(max_level)

    def test_simple_is_all_levels(self):
        self.assertEqual(self.levels("simple", None), [6, 5, 4, 3, 2, 1, 0])

    def test_sampled_two_is_only_max_and_min(self):
        self.assertEqual(self.levels("sampled", 2), [6, 0])

    def test_sampled_three_adds_one_middle_level(self):
        for _ in range(20):
            levels = self.levels("sampled", 3)
            self.assertEqual(len(levels), 3)
            self.assertEqual(levels[0], 6)
            self.assertEqual(levels[-1], 0)
            self.assertTrue(0 < levels[1] < 6)
            self.assertEqual(levels, sorted(levels, reverse=True))

    def test_sampled_draws_different_middle_levels(self):
        drawn = {tuple(self.levels("sampled", 3)) for _ in range(200)}
        self.assertGreater(len(drawn), 1)

    def test_sampled_all_levels_when_num_sampled_is_large(self):
        self.assertEqual(self.levels("sampled", 7), [6, 5, 4, 3, 2, 1, 0])
        self.assertEqual(self.levels("sampled", 99), [6, 5, 4, 3, 2, 1, 0])

    def test_default_is_sampled_with_four_levels(self):
        ctx = ShardDistillContext(loader_function=None, teacher_loader=lambda: nn.Identity())
        self.assertEqual(ctx.training_mode, "sampled")
        self.assertEqual(ctx.num_sampled, 4)
        levels = ShardDistillTrainer(common.tiny_config("shard_deit_v3", levels=5), ctx).sample_levels(4)
        self.assertEqual(len(levels), 4)
        self.assertEqual(levels[0], 4)
        self.assertEqual(levels[-1], 0)

    def test_invalid_settings_rejected(self):
        with self.assertRaises(ValueError):
            make_trainer(training_mode="sampled", num_sampled=1)
        with self.assertRaises(ValueError):
            make_trainer(training_mode="sampled", num_sampled=None)
        with self.assertRaises(ValueError):
            make_trainer(training_mode="random")


class TestTrainingStep(unittest.TestCase):
    def run_fit(self, trainer_module):
        visited = []

        def hook(module, _):
            visited.append(module.current_level())
        trainer_module.submodel.register_forward_pre_hook(hook)
        pl.Trainer(max_epochs=1, limit_train_batches=2, limit_val_batches=0, accelerator="cpu", devices=1,
                   logger=False, enable_checkpointing=False, enable_progress_bar=False,
                   enable_model_summary=False).fit(trainer_module, loader())
        return visited

    def test_every_level_is_trained_each_step_widest_first(self):
        m = make_trainer()
        before = [p.detach().clone() for p in m.submodel.parameters()]
        visited = self.run_fit(m)
        self.assertEqual(visited[:4], [3, 2, 1, 0])          # step 1: widest -> level 0
        self.assertEqual(visited[4:8], [3, 2, 1, 0])         # step 2
        after = list(m.submodel.parameters())
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, after)))   # parameters were updated

    def test_sampled_step_visits_only_sampled_levels(self):
        visited = self.run_fit(make_trainer(training_mode="sampled", num_sampled=2))
        self.assertEqual(visited[:2], [3, 0])
        self.assertEqual(visited[2:4], [3, 0])
        visited = self.run_fit(make_trainer(training_mode="sampled", num_sampled=3))
        self.assertEqual(visited[0], 3)
        self.assertEqual(visited[-1], 0)
        self.assertEqual(len(visited) % 3, 0)

    def test_us_transfer_also_visits_all_levels(self):
        self.assertEqual(self.run_fit(make_trainer("US"))[:4], [3, 2, 1, 0])

    def test_level_loss_uses_given_teacher(self):
        m = make_trainer()
        out = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        same = m._level_loss(out, out.clone(), targets)       # teacher == student: KL term is 0, only CE remains
        ce = nn.CrossEntropyLoss()(out, targets)
        torch.testing.assert_close(same, ce)
        other = m._level_loss(out, torch.randn(4, 10), targets)
        self.assertGreater(other.item(), ce.item())


class TestExperimentRegistry(unittest.TestCase):
    def test_keys_and_configs(self):
        from config.experiments import CONFIGS
        expected = {"shard_deit_v3", "shard_deit_v3_7level", "shard_deit_v3_lowFLOPS_5level",
                    "shard_deit_v3_ablation_random", "shard_deit_v3_ablation_importance",
                    "shard_deit_v3_ablation_naive", "shard_deit", "shard_resnet", "shard_vgg"}
        self.assertEqual(set(CONFIGS) - {"dummy"}, expected)   # tests/test_training.py registers "dummy"
        main = CONFIGS["shard_deit_v3"].model_config
        self.assertEqual(tuple(main.hidden_dims), (384, 480, 576, 672, 768))
        self.assertEqual(CONFIGS["shard_deit_v3_7level"].model_config.max_level(), 6)
        for name, perm in (("random", "random"), ("importance", "importance"), ("naive", "none")):
            cfg = CONFIGS[f"shard_deit_v3_ablation_{name}"].model_config
            self.assertEqual(cfg.head_permutation, perm)
            self.assertEqual(tuple(cfg.num_heads), (4, 6, 8, 10, 12))
            self.assertEqual(tuple(cfg.hidden_dims), (256, 384, 512, 640, 768))
            self.assertEqual(tuple(cfg.mlp_dims), (1024, 1536, 2048, 2560, 3072))


if __name__ == "__main__":
    unittest.main()
