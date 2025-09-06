import math
import torch
from torch import nn
from einops import einsum
from jaxtyping import Float, Int


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

    def forward(self, x: Float[torch.Tensor, "... d_in"]) -> Float[torch.Tensor, "... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(
            num_embeddings, embedding_dim, dtype=dtype, device=device))
        nn.init.trunc_normal_(self.weight, 0, 1, -3, 3)

    def forward(self, token_ids: Int[torch.Tensor, "... seq_len"]) -> Float[torch.Tensor, "... seq_len d_model"]:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(
            d_model, dtype=dtype, device=device))

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms: Float[torch.Tensor, "... one"] = torch.rsqrt(torch.sum(x**2, -1, keepdim=True) /
                                                          self.d_model + self.eps)
        result = einsum(x, self.gain, rms,
                        "... d_model, d_model, ... one -> ... d_model")
        return result.to(in_dtype)
