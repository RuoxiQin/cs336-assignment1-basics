import math
import torch
from torch import nn, Tensor
from einops import einsum, rearrange
from jaxtyping import Float, Int


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        # Store (out x in) so in_features will be stored consecutively in memory. During the matrix multiplication in_features dim
        # is scanned through and multiplied with input x vector.
        self.weight = nn.Parameter(torch.empty(out_features, in_features,
                                               dtype=dtype))
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std,
                              a=-3.0 * std, b=3.0 * std)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(
            num_embeddings, embedding_dim, dtype=dtype))
        nn.init.trunc_normal_(self.weight, 0, 1, -3, 3)

    def forward(self, token_ids: Int[Tensor, "... seq_len"]) -> Float[Tensor, "... seq_len d_model"]:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(
            d_model, dtype=dtype))

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms: Float[Tensor, "... one"] = torch.rsqrt(torch.sum(x**2, -1, keepdim=True) /
                                                    self.d_model + self.eps)
        result = einsum(x, self.gain, rms,
                        "... d_model, d_model, ... one -> ... d_model")
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    """
    Postion-wise Feed-Forward Network with SwiGLU activation.
    """

    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        std = math.sqrt(2.0 / (d_model + d_ff))
        self.w1_weight = nn.Parameter(
            torch.empty(d_ff, d_model, dtype=dtype))
        nn.init.trunc_normal_(self.w1_weight, mean=0, std=std,
                              a=-3.0 * std, b=3.0 * std)
        self.w2_weight = nn.Parameter(
            torch.empty(d_model, d_ff, dtype=dtype))
        nn.init.trunc_normal_(self.w2_weight, mean=0, std=std,
                              a=-3.0 * std, b=3.0 * std)
        self.w3_weight = nn.Parameter(
            torch.empty(d_ff, d_model, dtype=dtype))
        nn.init.trunc_normal_(self.w3_weight, mean=0, std=std,
                              a=-3.0 * std, b=3.0 * std)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        gate_pre_activation = einsum(
            x, self.w1_weight, "... d_model, d_ff d_model -> ... d_ff")
        swish_linear_unit = gate_pre_activation * \
            torch.sigmoid(gate_pre_activation)
        linear_pre_activation = einsum(
            x, self.w3_weight, "... d_model, d_ff d_model -> ... d_ff")
        return einsum(swish_linear_unit * linear_pre_activation, self.w2_weight, "... d_ff, d_model d_ff -> ... d_model")


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        # d_k is the dimension of the key or query vectors
        assert d_k % 2 == 0
        self.d_k = d_k
        # A list of thetas [theta_0, theta_1, ..., theta_(d_k/2 - 1)].
        # Note that this is not the same as the textbook PDF. But turns out this is the one that unit test uses and it's the conanical one.
        thetas_by_k = theta ** (-1 * (2 * torch.arange(d_k//2)) / d_k)
        # A list of positions [0, 1, 2, ..., max_seq_len - 1].
        positions_by_i = torch.arange(max_seq_len)
        # thetas_by_i_k is a matrix of shape (max_seq_len, d_k/2). It's an index
        thetas_by_i_k = einsum(positions_by_i, thetas_by_k,
                               "seq_len, d_k_half -> seq_len d_k_half")
        cos_by_i_k = torch.cos(thetas_by_i_k)
        sin_by_i_k = torch.sin(thetas_by_i_k)
        # Register the pre-computed cos and sin values as buffers. They are not parameters because they are not learned.
        # They are not persistent because they don't need to be saved in the model state.
        # They can be re-computed from the model hyperparameters every time.
        self.register_buffer("cos_by_i_k", cos_by_i_k, persistent=False)
        self.register_buffer("sin_by_i_k", sin_by_i_k, persistent=False)
        self.cos_by_i_k: torch.Tensor
        self.sin_by_i_k: torch.Tensor

    def forward(self, x: Float[torch.Tensor, "... seq_len d_k"], token_positions: Int[torch.Tensor, "... seq_len"]) -> Float[torch.Tensor, "... seq_len d_k"]:
        assert x.size(-1) == self.d_k
        # Note that odd is becuase it's odd for 1-based indexing.
        x_i_odd = x[..., 0::2]
        x_i_even = x[..., 1::2]

        # Expand the token_positions to shape of [..., seq_len, d_k/2] for slicing.
        # Note that expand means we simply duplicate the last int element to an int vector of d_k/2 length.
        index = token_positions.unsqueeze(-1)  # Shape [..., seq_len, 1]
        # Shape [..., seq_len, d_k_half]
        index = index.expand(*token_positions.shape, self.d_k//2)

        # Expand the cos_by_i_k and sin_by_i_k to shape [..., max_seq_len, d_k/2] to match the slicing dimension.
        if token_positions.dim() == 1:
            # "..." is empty, no need to change exapnd shape.
            cos_by_i_k = self.cos_by_i_k
            sin_by_i_k = self.sin_by_i_k
        else:
            cos_by_i_k = self.cos_by_i_k.unsqueeze(0)
            cos_by_i_k = cos_by_i_k.expand(*token_positions.shape, self.d_k//2)
            sin_by_i_k = self.sin_by_i_k.unsqueeze(0)
            sin_by_i_k = sin_by_i_k.expand(*token_positions.shape, self.d_k//2)
        # Slice along the seq_len dimension. Since the last dimension is just the same integer repeated d_k/2 times, the same index will be used for the entire vector of d_k/2 lengh.
        reordered_cos_by_i_k = torch.gather(cos_by_i_k, dim=-2, index=index)
        reordered_sin_by_i_k = torch.gather(sin_by_i_k, dim=-2, index=index)

        # Element-wise multiplication. Shape: [..., seq_len, d_k/2].
        x_odd_cos_i_k = x_i_odd * reordered_cos_by_i_k
        x_odd_sin_i_k = x_i_odd * reordered_sin_by_i_k
        x_even_cos_i_k = x_i_even * reordered_cos_by_i_k
        x_even_sin_i_k = x_i_even * reordered_sin_by_i_k
        # New odd and even parts after applying rotary positional embedding. Shape: [..., seq_len, d_k/2].
        new_x_odd = x_odd_cos_i_k - x_even_sin_i_k
        new_x_even = x_odd_sin_i_k + x_even_cos_i_k
        # Reconstruct the original tensor shape by interleaving odd and even parts. Shape: [..., seq_len, d_k/2, 2].
        stacked_new_x_odd_even = torch.stack((new_x_odd, new_x_even), dim=-1)
        # Merge the last two dimensions by alternatingly taking elements from each. Shape: [..., seq_len, d_k].
        return rearrange(
            stacked_new_x_odd_even, "... seq_len d_k_half two -> ... seq_len (d_k_half two)")
