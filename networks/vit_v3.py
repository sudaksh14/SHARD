from collections import OrderedDict
from functools import partial
from typing import Callable
from enum import Enum
import dataclasses
import math
import copy
import timm
from timm.models.layers import DropPath
from torchvision.models import vision_transformer
from torch import nn
import torch

from networks.config import ModelConfig
import utils


def remap_deitv3_to_shard_vit(state_dict):
    remapped = {}
    for k, v in state_dict.items():
        new_k = k

        # --- Top-level tokens ---
        new_k = new_k.replace("cls_token", "class_token.token")
        new_k = new_k.replace("pos_embed", "pos_embedding.embedding")
        new_k = new_k.replace("patch_embed.proj.weight", "conv_proj.weight")
        new_k = new_k.replace("patch_embed.proj.bias", "conv_proj.bias")
        new_k = new_k.replace("head.weight", "heads.head.weight")
        new_k = new_k.replace("head.bias", "heads.head.bias")

        # --- Encoder blocks ---
        if new_k.startswith("blocks."):
            new_k = new_k.replace("blocks.", "encoder.layers.encoder_layer_")

            # Normalization
            new_k = new_k.replace(".norm1.", ".ln_1.")
            new_k = new_k.replace(".norm2.", ".ln_2.")

            # LayerScale
            new_k = new_k.replace(".ls1.", ".ls1.")
            new_k = new_k.replace(".ls2.", ".ls2.")

            # Attention
            new_k = new_k.replace(".attn.qkv.", ".self_attention.in_proj_")
            new_k = new_k.replace(".attn.proj.", ".self_attention.out_proj.")

            # MLP
            new_k = new_k.replace(".mlp.fc1.", ".mlp.0.")
            new_k = new_k.replace(".mlp.fc2.", ".mlp.3.")

        remapped[new_k] = v

    return remapped

# This model is mostly an adapted version from DeiT v3: Revenge of the ViT
@dataclasses.dataclass
class ViTStructureConfig(utils.SelfDescripting):
    image_size: int
    patch_size: int
    num_layers: int
    num_heads: int
    hidden_dim: int
    mlp_dim: int

    def __hash__(self):
        return hash(self.get_description())


class ViTStructure:
    b16 = ViTStructureConfig(224, 16, 12, 12, 768, 3072)
    b32 = ViTStructureConfig(224, 32, 12, 12, 768, 3072)
    l16 = ViTStructureConfig(224, 16, 24, 16, 1024, 4096)
    l32 = ViTStructureConfig(224, 32, 24, 16, 1024, 4096)
    h14 = ViTStructureConfig(224, 14, 32, 16, 1280, 5120)


class ViTPrebuilt(Enum):
    noprebuild = 0
    default = 1
    imagenet1k_v1 = 2
    imagenet1k_swag_e2e_v1 = 3
    imagenet1k_swag_linear_v1 = 4

    # DeiT v3 from https://arxiv.org/abs/2204.07118
    Deit_v1 = 5
    Deit_v3_pretrain_1k = 6
    Deit_v3_pretrain_21k = 7


DEFAULT_NUM_CLASSES = 1000


@dataclasses.dataclass
class ViTConfig_v3(ModelConfig):
    structure: ViTStructureConfig = ViTStructure.b16
    prebuilt: ViTPrebuilt = ViTPrebuilt.default
    num_classes: int = DEFAULT_NUM_CLASSES
    dropout: float = 0.0
    attention_dropout: float = 0.0
    drop_path_rate: float = 0.1
    init_value: float = 1e-4

    def make_model(self):
        return VisionTransformer_v3(self)

    def no_prebuilt(self):
        self.prebuilt = ViTPrebuilt.noprebuild
        return self


KNOWN_MODEL_PRETRAINED = {
    (ViTStructure.b16, ViTPrebuilt.Deit_v3_pretrain_1k):
        lambda: timm.create_model('deit3_base_patch16_224.fb_in1k', pretrained=True, num_classes=1000),

    (ViTStructure.b16, ViTPrebuilt.Deit_v3_pretrain_21k):
        lambda: timm.create_model('deit3_base_patch16_224.fb_in22k_ft_in1k', pretrained=True, num_classes=1000),

    (ViTStructure.b32, ViTPrebuilt.imagenet1k_v1):
    lambda: vision_transformer.vit_b_32(
        weights=vision_transformer.ViT_B_32_Weights.IMAGENET1K_V1),
    (ViTStructure.b32, ViTPrebuilt.default):
    lambda: vision_transformer.vit_b_32(
        weights=vision_transformer.ViT_B_32_Weights.DEFAULT),

    (ViTStructure.l16, ViTPrebuilt.imagenet1k_v1):
    lambda: vision_transformer.vit_l_16(
        weights=vision_transformer.ViT_L_16_Weights.IMAGENET1K_V1),
    (ViTStructure.l16, ViTPrebuilt.imagenet1k_swag_linear_v1):
    lambda: vision_transformer.vit_l_16(
        weights=vision_transformer.ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1),
    (ViTStructure.l16, ViTPrebuilt.imagenet1k_swag_e2e_v1):
    lambda: vision_transformer.vit_l_16(
        weights=vision_transformer.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1),
    (ViTStructure.l16, ViTPrebuilt.default):
    lambda: vision_transformer.vit_l_16(
        weights=vision_transformer.ViT_L_16_Weights.DEFAULT),

    (ViTStructure.l32, ViTPrebuilt.imagenet1k_v1):
    lambda: vision_transformer.vit_l_32(
        weights=vision_transformer.ViT_L_32_Weights.IMAGENET1K_V1),
    (ViTStructure.l32, ViTPrebuilt.default):
    lambda: vision_transformer.vit_l_32(
        weights=vision_transformer.ViT_L_32_Weights.DEFAULT),

    (ViTStructure.h14, ViTPrebuilt.imagenet1k_swag_e2e_v1):
    lambda: vision_transformer.vit_h_14(
        weights=vision_transformer.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1),
    (ViTStructure.h14, ViTPrebuilt.imagenet1k_swag_linear_v1):
    lambda: vision_transformer.vit_h_14(
        weights=vision_transformer.ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1),
    (ViTStructure.h14, ViTPrebuilt.default):
    lambda: vision_transformer.vit_h_14(
        weights=vision_transformer.ViT_H_14_Weights.DEFAULT),
}


class MLPBlock(nn.Sequential):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(
            torch.nn.Linear(in_dim, mlp_dim, bias=True),
            nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(mlp_dim, in_dim, bias=True),
            torch.nn.Dropout(dropout),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        init_value: float,
        drop_path_rate: float = 0.1,
        norm_layer: Callable[..., torch.nn.Module] = partial(
            nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)
        
        self.ls1 = utils.LayerScale(hidden_dim, init_value=init_value)
        self.ls2 = utils.LayerScale(hidden_dim, init_value=init_value)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim(
        ) == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)

        x = self.ls1(x)
        x = self.drop_path(x) + input

        y = self.ln_2(x)
        y = self.mlp(y)
        y = self.ls2(y)

        return x + self.drop_path(y)


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        max_drop_path: float,
        init_value: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(
            nn.LayerNorm, eps=1e-6),
    
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                init_value,
                drop_path_rate=max_drop_path * i / (num_layers - 1),
                norm_layer=norm_layer
            )
        self.layers = nn.Sequential(layers)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim(
        ) == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        return self.layers(input)


class VisionTransformer_v3(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        config: ViTConfig_v3,
    ):
        image_size = config.structure.image_size
        patch_size = config.structure.patch_size
        num_layers = config.structure.num_layers
        num_heads = config.structure.num_heads
        hidden_dim = config.structure.hidden_dim
        mlp_dim = config.structure.mlp_dim

        num_classes = config.num_classes
        dropout = config.dropout
        attention_dropout = config.attention_dropout
        max_drop_path = config.drop_path_rate
        init_values = config.init_value
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        super().__init__()
        torch._assert(image_size % patch_size == 0,
                      "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.norm_layer = norm_layer

        # Patch embedding
        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        # seq_length += 1

        # Learnable class token and position embedding
        self.class_token = utils.ClassTokenLayer(hidden_dim)
        self.pos_embedding = utils.PosEmbeddingLayer(seq_length, hidden_dim)

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            max_drop_path,
            init_values,
            norm_layer,
        )

        self.seq_length = seq_length
        self.norm = norm_layer(hidden_dim)

        heads = OrderedDict()
        heads['head'] = utils.LinearHead(hidden_dim, DEFAULT_NUM_CLASSES)
        self.heads = nn.Sequential(heads)

        # Weights initialisation
        fan_in = self.conv_proj.in_channels * \
            self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight,
                              std=math.sqrt(1 / fan_in))
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)
        nn.init.zeros_(self.heads.head.weight)
        nn.init.zeros_(self.heads.head.bias)

        if config.prebuilt != ViTPrebuilt.noprebuild:
            prebuild_config = (config.structure, config.prebuilt)
            if prebuild_config not in KNOWN_MODEL_PRETRAINED:
                raise RuntimeError("prebuilt model not found")
            prebuilt = KNOWN_MODEL_PRETRAINED[prebuild_config]()
            sdict = prebuilt.state_dict()
            sdict = remap_deitv3_to_shard_vit(sdict)

            utils.shard_model_copy(sdict, self)
            # self.class_token.token = copy.deepcopy(prebuilt.class_token)
            # self.encoder.pos_embedding.embedding = copy.deepcopy(
            #     prebuilt.encoder.pos_embedding)

        if config.num_classes != DEFAULT_NUM_CLASSES:
            heads = OrderedDict()
            heads['head'] = utils.LinearHead(hidden_dim, num_classes)
            self.heads = nn.Sequential(heads)

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
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

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

        # Class token
        cls_tokens = self.class_token.token.expand(n, -1, -1)  # (B, 1, hidden_dim)
        
        # Add positional embedding
        x = x + self.pos_embedding.embedding  # (B, num_patches, hidden_dim)
        
        # Concatenate CLS token
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, hidden_dim)
        
        x = self.encoder(x)
        x = self.norm(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x
