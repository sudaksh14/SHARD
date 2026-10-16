from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Optional
import dataclasses
import copy

from torch import nn
import torch

from networks.vit import ViTStructureConfig, ViTStructure, ViTPrebuilt, KNOWN_MODEL_PRETRAINED, DEFAULT_NUM_CLASSES
from networks.shard_model import ShardModel
from networks.config import ShardModelConfig, ModelConfig
import shard_modules as fm
import networks.vit
import utils


# This model is mostly an adapted version from torchvision.models.vision_transformer

def scale_with_heads_list(heads, max_hidden_dims):
    max_num_heads = heads[-1]
    if max_hidden_dims % max_num_heads:
        raise RuntimeError()
    return [max_hidden_dims // max_num_heads * i for i in heads]


@dataclasses.dataclass
class ViTConfig(ShardModelConfig):
    structure: ViTStructureConfig = ViTStructure.b16
    prebuilt: ViTPrebuilt = ViTPrebuilt.default
    num_classes: int = DEFAULT_NUM_CLASSES
    dropout: float = 0.0
    attention_dropout: float = 0.0

    hidden_dims: Iterable[int] = (768 // 2, (768 // 3) * 2, 768)
    num_heads: Iterable[int] = (6, 8, 12)
    mlp_dims: Iterable[int] = (3072 // 2, (3072 // 3) * 2, 3072)

    def make_model(self):
        return VisionTransformer(self)

    def no_prebuilt(self):
        self.prebuilt = ViTPrebuilt.noprebuild
        return self

    def create_base_config(self, level) -> ModelConfig:
        return networks.vit.ViTConfig(
            networks.vit.ViTStructureConfig(
                self.structure.image_size,
                self.structure.patch_size,
                self.structure.num_layers,
                self.num_heads[level],
                self.hidden_dims[level],
                self.mlp_dims[level]),
            self.prebuilt,
            self.num_classes,
            self.dropout,
            self.attention_dropout
        )

    def max_level(self) -> int:
        return len(self.hidden_dims) - 1


class MLPBlock(nn.Sequential):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: Iterable[int], mlp_dim: Iterable[int], dropout: float):
        super().__init__(
            fm.Linear(in_dim, mlp_dim, bias=True),
            nn.GELU(),
            torch.nn.Dropout(dropout),
            fm.Linear(mlp_dim, in_dim, bias=True),
            torch.nn.Dropout(dropout),
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: Iterable[int],
        mlp_dim: Iterable[int],
        dropout: float,
        attention_dropout: float
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = fm.LayerNorm(hidden_dim, eps=1e-6)
        self.self_attention = fm.SelfAttention(
            hidden_dim, num_heads, dropout=attention_dropout)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = fm.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim(
        ) == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x = self.self_attention(x)

        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)

        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: Iterable[int],
        mlp_dim: Iterable[int],
        dropout: float,
        attention_dropout: float
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = fm.PosEmbeddingLayer(seq_length, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout
            )
        self.layers = nn.Sequential(layers)
        self.ln = fm.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim(
        ) == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = self.pos_embedding(input)
        return self.ln(self.layers(self.dropout(input)))
    
class VisionTransformer(ShardModel):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        config: ViTConfig,
    ):
        image_size = config.structure.image_size
        patch_size = config.structure.patch_size
        num_layers = config.structure.num_layers
        num_heads = config.num_heads
        hidden_dim = config.hidden_dims
        mlp_dim = config.mlp_dims

        num_classes = config.num_classes
        dropout = config.dropout
        attention_dropout = config.attention_dropout

        super().__init__(config)
        torch._assert(image_size % patch_size == 0,
                      "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes

        self.class_token = fm.ClassTokenLayer(hidden_dim)

        self.conv_proj = fm.Conv2d(
            [3] * len(hidden_dim), hidden_dim, kernel_size=patch_size, stride=patch_size)

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout
        )

        self.seq_length = seq_length

        heads = OrderedDict()
        heads['head'] = fm.LinearSelect(
            hidden_dim, [DEFAULT_NUM_CLASSES] * len(hidden_dim))
        self.heads = nn.Sequential(heads)

        self.set_level_use(self.max_level())
        self.level = self.max_level()
        if config.prebuilt != ViTPrebuilt.noprebuild:
            prebuild_config = (config.structure, config.prebuilt)
            if prebuild_config not in KNOWN_MODEL_PRETRAINED:
                raise RuntimeError("prebuilt model not found")
            prebuilt = KNOWN_MODEL_PRETRAINED[prebuild_config]()
            utils.shard_model_copy(prebuilt, self)
            self.class_token.token = copy.deepcopy(prebuilt.class_token)
            self.encoder.pos_embedding.embedding = copy.deepcopy(
                prebuilt.encoder.pos_embedding)

        if config.num_classes != DEFAULT_NUM_CLASSES:
            heads = OrderedDict()
            heads['head'] = fm.LinearSelect(
                hidden_dim, [num_classes] * len(hidden_dim))
            self.heads = nn.Sequential(heads)

    @staticmethod
    def base_type() -> type[nn.Module]:
        return networks.vit.VisionTransformer

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return len(self.hidden_dim) - 1

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size,
                      f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size,
                      f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim[self.current_level()], n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        x = self.class_token(x, n)
        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x

    @torch.no_grad()
    def export_level_delta(self) -> tuple[
            fm.DownDelta[tuple[fm.DownDelta, ...]],
            fm.UpDelta[tuple[fm.UpDelta, ...]]]:
        delta_down, delta_up = super().export_level_delta()
        return fm.DownDelta(
            (self.hidden_dim[self.level], delta_down)
        ), fm.UpDelta(
            (self.hidden_dim[self.level], delta_up))

    @staticmethod
    def apply_level_delta_down(
            model: nn.Module,
            level_delta: fm.DownDelta[tuple[fm.DownDelta, ...]]) -> None:
        hidden_dim, module_deltas = level_delta.delta
        ShardModel.apply_level_delta_down(model, module_deltas)
        model.hidden_dim = hidden_dim

    @staticmethod
    def apply_level_delta_up(
            model: nn.Module,
            level_delta: fm.UpDelta[tuple[fm.UpDelta, ...]]) -> None:
        hidden_dim, module_deltas = level_delta.delta
        ShardModel.apply_level_delta_up(model, module_deltas)
        model.hidden_dim = hidden_dim

