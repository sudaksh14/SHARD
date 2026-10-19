from torch import nn
import torch


class ClassTokenLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, x, n):
        batch_class_token = self.token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        return x


class PosEmbeddingLayer(nn.Module):
    def __init__(self, seq_length, hidden_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(
            1, seq_length, hidden_dim).normal_(std=0.02))

    def forward(self, x):
        return x + self.embedding


class LinearHead(nn.Linear):
    """
    This class only exists for compatibility with the level delta system, as we
    need a distinct type for a linear select and a shared weights linear layer.
    """

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)


class DistillTokenLayer(nn.Module):
    """
    Base distillation token layer (used by DeiT-style Vision Transformers).
    This is the non-shard version used for exporting/loading models.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """
        Append the distillation token to the sequence.

        Args:
            x (torch.Tensor): input sequence of shape (N, S, D)
            n (int): batch size

        Returns:
            torch.Tensor: sequence with distillation token appended (N, S+1, D)
        """
        dist_token = self.token.expand(n, -1, -1)
        x = torch.cat([x, dist_token], dim=1)
        return x

class LayerScale(nn.Module):
    """
    Base LayerScale module — used in DeiT-3 and CaiT.

    Applies a learnable, per-channel scaling to the output of
    a residual branch: x_out = x + gamma * F(x).

    Args:
        dim (int): hidden dimension
        init_value (float): initial gamma value, typically 1e-6
    """

    def __init__(self, dim: int, init_value: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale each hidden dimension independently
        return x * self.gamma