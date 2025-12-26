import math
import torch
from torch import nn, Tensor
from collections.abc import Callable, Iterable
from typing import Optional, Any, BinaryIO, IO
from einops import einsum, rearrange
from jaxtyping import Float, Int, Bool, jaxtyped
from typeguard import typechecked
import numpy.typing as npt
import numpy as np
import os


@jaxtyped(typechecker=typechecked)
@typechecked
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

    def forward(self, x: Float[Tensor, "*batch d_in"]) -> Float[Tensor, "*batch d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


@jaxtyped(typechecker=typechecked)
@typechecked
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(
            num_embeddings, embedding_dim, dtype=dtype))
        nn.init.trunc_normal_(self.weight, 0, 1, -3, 3)

    def forward(self, token_ids: Int[Tensor, "*batch seq_len"]) -> Float[Tensor, "*batch seq_len d_model"]:
        return self.weight[token_ids]


@jaxtyped(typechecker=typechecked)
@typechecked
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(
            d_model, dtype=dtype))

    def forward(self, x: Float[Tensor, "*batch d_model"]) -> Float[Tensor, "*batch d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms: Float[Tensor, "*batch 1"] = torch.rsqrt(torch.sum(x**2, -1, keepdim=True) /
                                                     self.d_model + self.eps)
        result = einsum(x, self.gain, rms,
                        "... d_model, d_model, ... one -> ... d_model")
        return result.to(in_dtype)


def silu(x: Float[Tensor, "*batch"]) -> Float[Tensor, "*batch"]:
    return x * torch.sigmoid(x)


@jaxtyped(typechecker=typechecked)
@typechecked
class SwiGLU(nn.Module):
    """
    Postion-wise Feed-Forward Network with SwiGLU activation.
    """

    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, dtype=dtype)
        self.linear3 = Linear(d_model, d_ff, dtype=dtype)

    def forward(self, x: Float[Tensor, "*batch d_model"]) -> Float[Tensor, "*batch d_model"]:
        return self.linear2(silu(self.linear1(x)) * self.linear3(x))


@jaxtyped(typechecker=typechecked)
@typechecked
class SiLU(nn.Module):
    """
    Postion-wise Feed-Forward Network with SiLU activation.
    """

    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff, dtype=dtype)
        self.linear2 = Linear(d_ff, d_model, dtype=dtype)

    def forward(self, x: Float[Tensor, "*batch d_model"]) -> Float[Tensor, "*batch d_model"]:
        return self.linear2(silu(self.linear1(x)))


@jaxtyped(typechecker=typechecked)
@typechecked
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        # d_k is the dimension of the key or query vectors
        if d_k % 2 != 0:
            raise ValueError(f"d_k is not even: {d_k}.")
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

    def forward(self, x: Float[torch.Tensor, "*batch seq_len d_k"], token_positions: Int[torch.Tensor, "*batch seq_len"]) -> Float[torch.Tensor, "*batch seq_len d_k"]:
        # Check the token_positions can be broadcast to x.shape[-1]. Usually token_positions.shape:
        #   1. Has the exact same shape as x.shape[-1] so every batch may have their own token_positions; OR
        #   2. All batches share the same token_positions. So it's 1-dim, or 2-dim where first dim is 1.
        try:
            torch.broadcast_shapes(x.shape[:-1], token_positions.shape)
        except RuntimeError:
            raise ValueError(
                f"token_positions {token_positions.shape} cannot be broadcast to x {x.shape[:-1]}")

        if x.size(-1) != self.d_k:
            raise ValueError(
                f"Input x.size(-1)={x.size(-1)} doesn't match d_k={self.d_k}.")
        seq_len = x.size(-2)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input x sequence length={seq_len} exceeds max sequence length={self.max_seq_len}.")

        # Note that odd is becuase it's odd for 1-based indexing.
        x_i_odd = x[..., 0::2]
        x_i_even = x[..., 1::2]

        reordered_cos_by_i_k = self.cos_by_i_k[token_positions]
        reordered_sin_by_i_k = self.sin_by_i_k[token_positions]

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


@jaxtyped(typechecker=typechecked)
@typechecked
def softmax(x: Float[Tensor, "*batch"], dim: int, temperature: float = 1.0) -> Float[Tensor, "*batch"]:
    scaled_x = x / temperature
    scaled_x = scaled_x - scaled_x.amax(dim, keepdim=True)
    exp_x = torch.exp(scaled_x)
    return exp_x / exp_x.sum(dim, keepdim=True)


@jaxtyped(typechecker=typechecked)
@typechecked
def scaled_dot_product_attention(
    Q: Float[Tensor, "*batch queries d_k"],
    K: Float[Tensor, "*batch keys d_k"],
    V: Float[Tensor, "*batch keys d_v"],
    mask: Bool[Tensor, "*batch queries keys"] | None = None,
) -> Float[Tensor, "*batch queries d_v"]:
    d_k = Q.size(-1)
    QK = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    if mask is not None:
        # mask=True means the attention should be propogated, i.e. don't fill with -inf.
        QK = QK.masked_fill(~mask, float("-inf"))
    similarity_weight = softmax(QK / (d_k)**0.5, dim=-1)
    return einsum(similarity_weight, V, "... queries keys, ... keys d_v -> ... queries d_v")


@jaxtyped(typechecker=typechecked)
@typechecked
class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float | None = None, max_seq_len: int | None = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} cannot be fully divided by num_heads={num_heads}.")
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

    def forward(self, x: Float[Tensor, " *batch seq_len d_model"], token_positions: Int[torch.Tensor, "*batch seq_len"] | None = None) -> Float[Tensor, " *batch seq_len d_model"]:
        if x.size(-1) != self.d_model:
            raise ValueError(
                f"Input x.size(-1)={x.size(-1)} doesn't match d_model={self.d_model}.")
        seq_len = x.size(-2)

        # Recover the Q, K, V from the concatenated matrix.
        Q = rearrange(self.linear_Q(
            x), "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        K = rearrange(self.linear_K(
            x), "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        V = rearrange(self.linear_V(
            x), "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)

        if token_positions is not None:
            if self.positional_embedding is None:
                raise ValueError(
                    "token_positions is specified but the positional_embedding is not initialized.")
            Q = self.positional_embedding(Q, token_positions)
            K = self.positional_embedding(K, token_positions)

        # causal_mask allows Query token to attend to itself and all tokens before it.
        causal_mask = torch.tril(torch.ones(
            seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=0)
        causal_mask.unsqueeze(0)
        causal_mask = causal_mask.expand(*K.shape[:-2], seq_len, seq_len)

        attentioned_value = scaled_dot_product_attention(Q, K, V, causal_mask)
        concatenated_attentioned_value = rearrange(
            attentioned_value, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")
        return self.linear_O(concatenated_attentioned_value)


@jaxtyped(typechecker=typechecked)
@typechecked
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int):
        super().__init__()
        self.attention_pre_norm = RMSNorm(d_model)
        self.multi_head_self_attention = CausalMultiHeadSelfAttention(
            d_model, num_heads, theta, max_seq_len)
        self.feed_forward_pre_norm = RMSNorm(d_model)
        self.swi_gated_linear_unit = SwiGLU(d_model, d_ff)

        token_positions = torch.arange(max_seq_len)
        self.register_buffer(
            "token_positions", token_positions, persistent=False)
        self.token_positions: torch.Tensor

    def forward(self, in_features: Float[Tensor, " *batch seq_len d_model"]) -> Float[Tensor, " *batch seq_len d_model"]:
        seq_len = in_features.size(-2)
        normalized_in_features = self.attention_pre_norm(in_features)

        # Construct the natural-ordered token_positions matrix.
        token_positions = self.token_positions[:seq_len]

        attention_output = self.multi_head_self_attention(
            normalized_in_features, token_positions)
        summed_attention_output = in_features + attention_output
        normalized_summed_attention_output = self.feed_forward_pre_norm(
            summed_attention_output)
        feed_forward_output = self.swi_gated_linear_unit(
            normalized_summed_attention_output)
        return summed_attention_output + feed_forward_output


@jaxtyped(typechecker=typechecked)
@typechecked
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

    def forward(self, in_indices: Int[Tensor, " *batch seq_len"]) -> Float[Tensor, "*batch seq_len vocab_size"]:
        if in_indices.size(-1) > self.context_length:
            raise ValueError(
                f"in_indices sequence length={in_indices.size(-1)} exceeds max context_length={self.context_length}.")

        x = self.embedding(in_indices)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.final_rms_norm(x)
        return self.final_linear(x)


@jaxtyped(typechecker=typechecked)
@typechecked
def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    max_logits = inputs.amax(dim=-1, keepdim=True)
    loss: Float[Tensor, " batch_size one"] = -torch.gather(inputs, dim=-1, index=targets.unsqueeze(
        -1).long()) + max_logits + torch.log(torch.exp(inputs - max_logits).sum(dim=-1, keepdim=True))
    return loss.mean()


@jaxtyped(typechecker=typechecked)
@typechecked
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, betas: tuple[float, float], weight_decay: float = 0.1, eps: float = 1e-8):
        defaults = {"lr": lr, "betas": betas,
                    "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)  # t starts from 1
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                grad = p.grad.data
                m = betas[0] * m + (1.0-betas[0]) * grad
                v = betas[1] * v + (1.0-betas[1]) * grad ** 2
                alpha_t = lr * \
                    math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)
                p.data.sub_(alpha_t * m / (torch.sqrt(v) + eps))
                p.data.mul_(1 - lr * weight_decay)

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v

        return loss


@jaxtyped(typechecker=typechecked)
@typechecked
def get_lr_cosine_schedule(it: int,
                           max_learning_rate: float,
                           min_learning_rate: float,
                           warmup_iters: int,
                           cosine_cycle_iters: int,) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)))
    return min_learning_rate


@jaxtyped(typechecker=typechecked)
@typechecked
@torch.no_grad()
def clip_gradient(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> Float[Tensor, ""]:
    """
    Clip gradient in-place without GPU-CPU sync.

    The l2_norm is computed as if all paramters concatenates to a single vector and then compute its l2_norm.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)

    total_l2_norm_square = torch.tensor(0.0, device=grads[0].device)
    for grad in grads:
        total_l2_norm_square += grad.detach().norm(p=2) ** 2
    total_l2_norm = total_l2_norm_square.sqrt()

    scale_factor = (max_l2_norm / (total_l2_norm + eps)).clamp(max=1.0)

    for param in parameters:
        if param.grad is not None:
            param.grad.mul_(scale_factor)

    return total_l2_norm


@jaxtyped(typechecker=typechecked)
@typechecked
def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    starting_positions = np.random.randint(
        low=0, high=dataset.size - context_length, size=batch_size)
    # Use boradcast rule to construct idx of shape [batch_size, context_length]
    idx = starting_positions[:, None] + np.arange(context_length)[None, :]
    data = dataset[idx]
    label = dataset[idx + 1]
    if "cuda" not in device:
        return (torch.from_numpy(data).to(device), torch.from_numpy(label).to(device))
    # Otherwise, pin the memory to speed up the transfer to GPU.
    data_cpu = torch.from_numpy(data).pin_memory()
    label_cpu = torch.from_numpy(label).pin_memory()
    return (data_cpu.to(device, non_blocking=True), label_cpu.to(device, non_blocking=True))


@jaxtyped(typechecker=typechecked)
@typechecked
def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: str | os.PathLike | BinaryIO | IO[bytes],
                    wandb_run_id: str | None = None,
                    config: dict[str, Any] | None = None) -> None:
    state = {"iteration": iteration,
             "model_state": model.state_dict(),
             "optimizer_state": optimizer.state_dict(),
             "wandb_run_id": wandb_run_id,
             "config": config}
    torch.save(state, out)


@jaxtyped(typechecker=typechecked)
@typechecked
def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None,) -> dict[str, Any]:
    state: dict[str, Any] = torch.load(src)
    model.load_state_dict(state["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state"])
    return {"iteration": state["iteration"], "wandb_run_id": state.get("wandb_run_id", None), "config": state.get("config", None)}


@jaxtyped(typechecker=typechecked)
@typechecked
def top_p_filter(probs: Float[Tensor, "*batch_size vocab_size"], p: float) -> Float[Tensor, "*batch_size vocab_size"]:
    # Sort probabilities in descending order
    # sorted_probs: the values, sorted_indices: their original positions
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Create a mask for tokens to remove
    # We want to keep tokens where the cumulative probability (up to that point)
    # is <= p. Everything after the first token that crosses 'p' gets masked.
    # We shift the mask by one to ensure the first token that exceeds 'p' is KEPT.
    removed_mask = cumulative_probs > p
    removed_mask[..., 1:] = removed_mask[..., :-1].clone()
    removed_mask[..., 0] = False

    # Zero out the probabilities of the masked tokens
    sorted_probs[removed_mask] = 0.0

    # Re-normalize the remaining probabilities
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # Scatter the values back to their original indices
    # We create a zero tensor and "put back" the remaining probabilities
    output = torch.zeros_like(probs)
    output.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)

    return output
