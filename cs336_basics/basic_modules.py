import math
import torch
from torch import nn, Tensor
from einops import einsum, rearrange
from jaxtyping import Float, Int, Bool


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
        self.linear1 = Linear(d_model, d_ff, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, dtype=dtype)
        self.linear3 = Linear(d_model, d_ff, dtype=dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        gate_pre_activation: Float[Tensor, "... d_ff"] = self.linear1(x)
        swish_linear_unit = gate_pre_activation * \
            torch.sigmoid(gate_pre_activation)
        linear_pre_activation: Float[Tensor, "... d_ff"] = self.linear3(x)
        return self.linear2(swish_linear_unit * linear_pre_activation)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        # d_k is the dimension of the key or query vectors
        assert d_k % 2 == 0
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # A list of thetas [theta_0, theta_1, ..., theta_(d_k/2 - 1)].
        # Note that this is not the same as the textbook PDF. But turns out this is the one that unit test uses and it's the conanical one.
        thetas_by_k = theta ** (-1 * (2 * torch.arange(d_k//2)) / d_k)
        # A list of positions [0, 1, 2, ..., max_seq_len - 1].
        positions_by_i = torch.arange(max_seq_len)
        # thetas_by_i_k is a matrix of shape (max_seq_len, d_k/2). It's an index
        thetas_by_i_k = einsum(positions_by_i, thetas_by_k,
                               "max_seq_len, d_k_half -> max_seq_len d_k_half")
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
        seq_len = x.size(-2)
        assert seq_len <= self.max_seq_len

        # Note that odd is becuase it's odd for 1-based indexing.
        x_i_odd = x[..., 0::2]
        x_i_even = x[..., 1::2]

        # Expand the token_positions to shape of [..., seq_len, d_k/2] for slicing.
        # Note that expand means we simply duplicate the last int element to an int vector of d_k/2 length.
        index = token_positions.unsqueeze(-1)  # Shape [..., seq_len, 1]
        # Shape [..., seq_len, d_k_half]
        index = index.expand(*token_positions.shape, self.d_k//2)

        # The cos_by_i_k is a cache for max_seq_len. The actual input usually has a shorter seq_len.
        # So we only need to utilize part of the cache.
        cos_by_i_k = self.cos_by_i_k[:seq_len, ...]
        sin_by_i_k = self.sin_by_i_k[:seq_len, ...]
        # Expand the cos_by_i_k and sin_by_i_k to shape [..., max_seq_len, d_k/2] to match the slicing dimension.
        if token_positions.dim() > 1:
            cos_by_i_k = cos_by_i_k.unsqueeze(0)
            cos_by_i_k = cos_by_i_k.expand(*token_positions.shape, self.d_k//2)
            sin_by_i_k = sin_by_i_k.unsqueeze(0)
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


def softmax(x: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    x = x - x.amax(dim, keepdim=True)
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.size(-1)
    QK = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    if mask is not None:
        # mask=True means the attention should be propogated, i.e. don't fill with -inf.
        QK = QK.masked_fill(~mask, float("-inf"))
    similarity_weight = softmax(QK / (d_k)**0.5, dim=-1)
    return einsum(similarity_weight, V, "... queries keys, ... keys d_v -> ... queries d_v")


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float | None = None, max_seq_len: int | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.linear_Q = Linear(self.d_model, num_heads * self.d_k)
        self.linear_K = Linear(self.d_model, num_heads * self.d_k)
        self.linear_V = Linear(self.d_model, num_heads * self.d_v)
        self.linear_O = Linear(num_heads * self.d_v, self.d_model)

        self.positional_embedding = None
        if theta and max_seq_len:
            self.positional_embedding = RotaryPositionalEmbedding(
                theta, self.d_k, max_seq_len)

    def forward(self, x: Float[Tensor, " ... seq_len d_model"], token_positions: Int[torch.Tensor, "... seq_len"] | None = None) -> Float[Tensor, " ... seq_len d_model"]:
        assert x.size(-1) == self.d_model
        seq_len = x.size(-2)

        # Recover the Q, K, V from the concatenated matrix.
        Q = rearrange(self.linear_Q(
            x), "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        K = rearrange(self.linear_K(
            x), "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        V = rearrange(self.linear_V(
            x), "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)

        if token_positions is not None:
            assert self.positional_embedding is not None
            Q = self.positional_embedding(Q, token_positions)
            K = self.positional_embedding(K, token_positions)

        # causal_mask allows Query token to attend to itself and all tokens before it.
        causal_mask = torch.tril(torch.ones(
            seq_len, seq_len, dtype=torch.bool), diagonal=0)
        causal_mask.unsqueeze(0)
        causal_mask = causal_mask.expand(*K.shape[:-2], seq_len, seq_len)

        attentioned_value = scaled_dot_product_attention(Q, K, V, causal_mask)
        concatenated_attentioned_value = rearrange(
            attentioned_value, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")
        return self.linear_O(concatenated_attentioned_value)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int):
        super().__init__()
        self.attention_pre_norm = RMSNorm(d_model)
        self.multi_head_self_attention = CausalMultiHeadSelfAttention(
            d_model, num_heads, theta, max_seq_len)
        self.feed_forward_pre_norm = RMSNorm(d_model)
        self.swi_gated_linear_unit = SwiGLU(d_model, d_ff)

    def forward(self, in_features: Float[Tensor, " ... seq_len d_model"]) -> Float[Tensor, " ... seq_len d_model"]:
        seq_len = in_features.size(-2)
        normalized_in_features = self.attention_pre_norm(in_features)

        # Construct the natural-ordered token_positions matrix.
        token_positions = torch.arange(seq_len)
        token_positions = token_positions.unsqueeze(0)
        token_positions = token_positions.expand(*in_features.shape[:-1])

        attention_output = self.multi_head_self_attention(
            normalized_in_features, token_positions)
        summed_attention_output = in_features + attention_output
        normalized_summed_attention_output = self.feed_forward_pre_norm(
            summed_attention_output)
        feed_forward_output = self.swi_gated_linear_unit(
            normalized_summed_attention_output)
        return summed_attention_output + feed_forward_output


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, rope_theta: float, context_length: int, num_layers: int) -> None:
        super().__init__()
        self.context_length = context_length
        self.embedding = Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(
            d_model, num_heads, d_ff, theta=rope_theta, max_seq_len=context_length) for _ in range(num_layers)])
        self.final_rms_norm = RMSNorm(d_model)
        self.final_linear = Linear(d_model, vocab_size)

    def forward(self, in_indices: Int[Tensor, " ... seq_len"]):
        assert in_indices.size(-1) <= self.context_length

        x = self.embedding(in_indices)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.final_rms_norm(x)
        return self.final_linear(x)
