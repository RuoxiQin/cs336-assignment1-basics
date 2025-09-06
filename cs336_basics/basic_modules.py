import math
import torch
from torch import nn
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        # Store (out x in) so in_features will be stored consecutively in memory. During the matrix multiplication in_features dim
        # is scanned through and multiplied with input x vector.
        self.weight = nn.Parameter(torch.empty(out_features, in_features,
                                               dtype=dtype, device=device))
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std,
                              a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
