from __future__ import annotations

from collections import OrderedDict
from typing import Iterable, Optional
import dataclasses
import copy

from torch import nn
import torch
from timm.models.layers import DropPath

from networks.vit_v3 import ViTStructureConfig, ViTStructure, ViTPrebuilt, KNOWN_MODEL_PRETRAINED, DEFAULT_NUM_CLASSES, remap_deitv3_to_shard_vit
from networks.shard_model import ShardModel
from networks.config import ShardModelConfig, ModelConfig
import shard_modules as fm
import networks.vit_v3
import utils
import networks.head_ablations as ha

# This model is mostly an adapted version from DeiT v3: Revenge of the ViT

def scale_with_heads_list(heads, max_hidden_dims):
    max_num_heads = heads[-1]
    if max_hidden_dims % max_num_heads:
        raise RuntimeError()
    return [max_hidden_dims // max_num_heads * i for i in heads]


@dataclasses.dataclass
class ViTConfig_v3(ShardModelConfig):
    structure: ViTStructureConfig = ViTStructure.b16
    prebuilt: ViTPrebuilt = ViTPrebuilt.Deit_v3_pretrain_21k
    num_classes: int = DEFAULT_NUM_CLASSES
    dropout: float = 0.0
    attention_dropout: float = 0.0
    drop_path_rate: float = 0.1
    init_value: float = 1e-4
    use_distillation: bool = False
    head_permutation : str = "none"  # options: {"none", "random", "importance"}

    hidden_dims: Iterable[int] = (768 // 2, (768 // 3) * 2, 768)
    num_heads: Iterable[int] = (6, 8, 12)
    mlp_dims: Iterable[int] = (3072 // 2, (3072 // 3) * 2, 3072)

    def make_model(self):
        return VisionTransformer_v3(self)

    def no_prebuilt(self):
        self.prebuilt = ViTPrebuilt.noprebuild
        return self

    def create_base_config(self, level) -> ModelConfig:
        return networks.vit_v3.ViTConfig_v3(
            networks.vit_v3.ViTStructureConfig(
                self.structure.image_size,
                self.structure.patch_size,
                self.structure.num_layers,
                self.num_heads[level],
                self.hidden_dims[level],
                self.mlp_dims[level]),
            self.prebuilt,
            self.num_classes,
            self.dropout,
            self.attention_dropout, 
            self.drop_path_rate,
            self.init_value
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
    """Transformer encoder block adapted for DeiT-3 (DropPath + optional LayerScale)."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        init_values: float = 1e-4  # LayerScale init
    ):
        super().__init__()
        self.num_heads = num_heads

        # === Attention block ===
        self.ln_1 = fm.LayerNorm(hidden_dim, eps=1e-6)
        self.self_attention = fm.SelfAttention(
            hidden_dim, num_heads, dropout=attention_dropout
        )

        # === MLP block ===
        self.ln_2 = fm.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

        # === LayerScale (optional, for DeiT-3) ===
        self.ls1 = fm.LayerScale(hidden_dim, init_value=init_values)
        self.ls2 = fm.LayerScale(hidden_dim, init_value=init_values)
        
        # === Stochastic depth ===
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()


    def forward(self, x: torch.Tensor):
        """
        x: (batch_size, seq_length, hidden_dim)
        """
        # === Attention branch ===
        residual = x
        x = self.ln_1(x)
        x = self.self_attention(x)
        x = self.ls1(x)
        x = residual + self.drop_path(x)

        # === MLP branch ===
        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = self.ls2(x)
        x = residual + self.drop_path(x)

        return x
    
    
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
        attention_dropout: float,
        max_drop_path: float,
        init_values: float
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        # self.pos_embedding = fm.PosEmbeddingLayer(seq_length, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                drop_path_rate=max_drop_path * i / (num_layers - 1),
                init_values=init_values
            )
        self.layers = nn.Sequential(layers)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim(
        ) == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        # input = self.pos_embedding(input)
        return self.layers(self.dropout(input))


class VisionTransformer_v3(ShardModel):
    """Vision Transformer with optional Distillation Token (DeiT-style)."""

    def __init__(self, config: ViTConfig_v3):
        image_size = config.structure.image_size
        patch_size = config.structure.patch_size
        num_layers = config.structure.num_layers
        num_heads = config.num_heads
        hidden_dim = config.hidden_dims
        mlp_dim = config.mlp_dims

        num_classes = config.num_classes
        dropout = config.dropout
        attention_dropout = config.attention_dropout
        max_drop_path = config.drop_path_rate
        init_value = config.init_value
        use_distillation = getattr(config, "use_distillation", False)

        super().__init__(config)
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.use_distillation = use_distillation

        # Patch embedding
        self.conv_proj = fm.Conv2d(
            [3] * len(hidden_dim),
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Sequence length (patches + class + optional dist token)
        seq_length = (image_size // patch_size) ** 2
        if self.use_distillation:
            seq_length += 1

        # Learnable class token and position embedding
        self.class_token = fm.ClassTokenLayer(hidden_dim)
        self.pos_embedding = fm.PosEmbeddingLayer(seq_length, hidden_dim)

        # Distillation token (for DeiT)
        if self.use_distillation:
            self.dist_token = fm.DistillTokenLayer(hidden_dim)
        else:
            self.dist_token = None

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            max_drop_path,
            init_value,
        )

        self.seq_length = seq_length
        self.norm = fm.LayerNorm(hidden_dim, eps=1e-6)

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
            # print(type(prebuilt))
            # print(prebuilt.state_dict().keys())
            reg_model = self.make_base_copy()
            reg_model.load_state_dict(remap_deitv3_to_shard_vit(prebuilt.state_dict()))
            
            # Head based ablations
            if config.head_permutation == "importance":
                ha.reorder_all_attention_heads(reg_model)
            elif config.head_permutation == "random":
                ha.reorder_all_attention_heads_random(reg_model, seed=42)
                
            self.load_from_base(reg_model)
            del reg_model
            # utils.shard_model_copy(prebuilt, self)
            # self.class_token.token = copy.deepcopy(prebuilt.class_token)
            # self.pos_embedding.embedding = copy.deepcopy(
            #     prebuilt.pos_embedding)

        if config.num_classes != DEFAULT_NUM_CLASSES:
            heads = OrderedDict()
            heads['head'] = fm.LinearSelect(
                hidden_dim, [num_classes] * len(hidden_dim))
            self.heads = nn.Sequential(heads)

    @staticmethod
    def base_type() -> type[nn.Module]:
        return networks.vit_v3.VisionTransformer_v3

    def current_level(self) -> int:
        return self.level

    def max_level(self) -> int:
        return len(self.hidden_dim) - 1

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim[self.current_level()], n_h * n_w)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x: torch.Tensor):
        # Patchify + embed
        x = self._process_input(x)
        n = x.shape[0]

        # Class token
        cls_tokens = self.class_token.token[:, :, :self.hidden_dim[self.current_level()]].expand(n, -1, -1)  # (n, 1, hidden_dim)
        
        if self.use_distillation:
            x = self.dist_token(x, n)

        # Add positional embedding
        x = x + self.pos_embedding.embedding[:, :, :self.hidden_dim[self.current_level()]]  # (n, num_patches, hidden_dim)

        # Concatenate CLS token
        x = torch.cat((cls_tokens, x), dim=1)  # (n, num_patches+1, hidden_dim)

        # Encode sequence
        x = self.encoder(x)
        x = self.norm(x)

        # Class token output
        cls_out = self.heads(x[:, 0])

        # Distillation head output (if enabled)
        if self.use_distillation:
            dist_out = self.heads(x[:, -1])
            # During training: return both heads
            if self.training:
                return cls_out, dist_out
            # During inference: average outputs
            return (cls_out + dist_out) / 2
        else:
            return cls_out

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