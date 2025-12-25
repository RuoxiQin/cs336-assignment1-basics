"""
Run Transformer LM training.

uv run python cs336_basics/training_main.py \
--training_token_file_path="data/TinyStoriesV2-GPT4-train-tokens.dat" \
--testing_token_file_path="data/TinyStoriesV2-GPT4-valid-tokens.dat" \
"""
import logging
import argparse
from pathlib import Path
import numpy as np
from torch import Tensor
from jaxtyping import Float, Int
from einops import rearrange
import torch
import os
from dotenv import load_dotenv
import wandb
from basic_modules import get_lr_cosine_schedule, get_batch, TransformerLM, cross_entropy, AdamW, clip_gradient, save_checkpoint, load_checkpoint

load_dotenv()  # Load environment variables from .env file if present.
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,                # Minimum level to log
        format="%(asctime)s [%(levelname)s] %(message)s",  # Log format
        handlers=[
            logging.StreamHandler(),       # Logs to console
            logging.FileHandler("data/training.log")  # Logs to file
        ]
    )

    parser = argparse.ArgumentParser(
        description="A script to train Transformer LM.")
    parser.add_argument("--training_token_file_path", type=str,
                        help="The path to the training token file.")
    parser.add_argument("--testing_token_file_path", type=str,
                        help="The path to the testing token file.")
    parser.add_argument("--load_model_checkpoint_path", type=str)
    parser.add_argument("--batch_size", type=int, default=48,)
    parser.add_argument("--vocab_size", type=int, default=10000,)
    parser.add_argument("--context_length", type=int, default=256,)
    parser.add_argument("--d_model", type=int, default=512,)
    parser.add_argument("--num_heads", type=int, default=16,)
    parser.add_argument("--d_ff", type=int, default=1344,)  # 8/3 * d_model
    parser.add_argument("--rope_theta", type=float, default=10000.0,)
    parser.add_argument("--num_layers", type=int, default=4,)
    parser.add_argument("--device", type=str, default="cuda",)
    parser.add_argument("--max_learning_rate", type=float, default=6e-4,)
    parser.add_argument("--min_learning_rate", type=float,
                        default=6e-5,)  # 10% of max_learning_rate
    parser.add_argument("--max_iters", type=int, default=50000,)
    parser.add_argument("--warmup_iters", type=int,
                        default=2500,)  # 5% of max_iters
    parser.add_argument("--cosine_cycle_iters", type=int,
                        default=50000,)  # 100% of max_iters
    parser.add_argument("--beta1", type=float, default=0.9,)
    parser.add_argument("--beta2", type=float, default=0.98,)
    parser.add_argument("--weight_decay", type=float, default=0.1,)

    args = parser.parse_args()

    # Load training tokens.
    training_tokens_mmap = np.memmap(
        args.training_token_file_path,
        dtype=np.int32,
        mode='r'
    )
    testing_tokens_mmap = np.memmap(
        args.testing_token_file_path,
        dtype=np.int32,
        mode='r'
    )

    transformer_lm = TransformerLM(vocab_size=args.vocab_size, d_model=args.d_model, num_heads=args.num_heads,
                                   d_ff=args.d_ff, rope_theta=args.rope_theta, context_length=args.context_length, num_layers=args.num_layers).to(args.device)

    adamw_optimizer = AdamW(transformer_lm.parameters(), lr=args.max_learning_rate, betas=(
        args.beta1, args.beta2), weight_decay=args.weight_decay)

    if args.load_model_checkpoint_path:
        logger.info(
            f"Loading model and optimizer state from {args.load_model_checkpoint_path}.")
        checkpoint = load_checkpoint(args.load_model_checkpoint_path, transformer_lm,
                                               adamw_optimizer)
        logger.info(
            f"Resuming training from iteration {checkpoint['iteration']}.")
        start_iteration = checkpoint["iteration"] + 1
        wandb_run_id = checkpoint.get("wandb_run_id", None)
    else:
        start_iteration = 0
        wandb_run_id = None

    wandb_run = None
    if os.getenv("WANDB_API_KEY"):
        wandb.login()

        wandb_run = wandb.init(
            project="cs336-basics-transformer-lm-tinystories",
            config={
                "training_token_file_path": args.training_token_file_path,
                "testing_token_file_path": args.testing_token_file_path,
                "batch_size": args.batch_size,
                "vocab_size": args.vocab_size,
                "context_length": args.context_length,
                "d_model": args.d_model,
                "num_heads": args.num_heads,
                "d_ff": args.d_ff,
                "rope_theta": args.rope_theta,
                "num_layers": args.num_layers,
                "device": args.device,
                "max_learning_rate": args.max_learning_rate,
                "min_learning_rate": args.min_learning_rate,
                "max_iters": args.max_iters,
                "warmup_iters": args.warmup_iters,
                "cosine_cycle_iters": args.cosine_cycle_iters,
                "beta1": args.beta1,
                "beta2": args.beta2,
                "weight_decay": args.weight_decay,
            },
            id = wandb_run_id,
            resume = "allow",
        )
        # Hide the 'iter' plot itself (it's just a diagonal line)
        wandb_run.define_metric("iter", hidden=True)
        wandb_run.define_metric("train/*", step_metric="iter")
        wandb_run.define_metric("val/*", step_metric="iter")
        wandb_run.define_metric("lr", step_metric="iter")

    for iteration in range(start_iteration, args.max_iters + 1):
        # Calculate the new learning rate for this iteration
        current_lr = get_lr_cosine_schedule(
            it=iteration,
            max_learning_rate=args.max_learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters
        )
        # Apply the new learning rate to the optimizer
        for param_group in adamw_optimizer.param_groups:
            param_group['lr'] = current_lr

        # Zero gradients from previous iteration.
        adamw_optimizer.zero_grad()

        # Sample a batch of token sequences.
        training_token_sequences: Int[Tensor, "batch_size context_length"]
        label_token_sequences: Int[Tensor, "batch_size context_length"]
        training_token_sequences, label_token_sequences = get_batch(
            training_tokens_mmap, batch_size=args.batch_size, context_length=args.context_length, device=args.device)

        # Model predictions.
        predicted_logits: Float[Tensor, "batch_size context_length vocab_size"] = transformer_lm(
            training_token_sequences)
        # Compute cross-entropy loss.
        predicted_logits = rearrange(
            predicted_logits, "batch_size context_length vocab_size -> (batch_size context_length) vocab_size")
        label_token_sequences = rearrange(
            label_token_sequences, "batch_size context_length -> (batch_size context_length)")
        loss: Float[Tensor, ""] = cross_entropy(
            predicted_logits, label_token_sequences)

        # Compute grad from back propagation.
        loss.backward()

        # Gradient clipping.
        clip_gradient(transformer_lm.parameters(), max_l2_norm=1.0)

        # Update weights.
        adamw_optimizer.step()

        # Print training loss.
        if iteration % 10 == 0:
            logger.info(
                f"Training loss at iteration {iteration}: {loss.item():.4f}")
            if wandb_run is not None:
                wandb_run.log({
                    "iter": iteration,
                    "train/loss": loss.item(),
                    "lr": current_lr,
                })

        # Compute testing loss.
        if iteration % 500 == 0 and iteration > 0:
            # Evaluate on testing set.
            testing_token_sequences: Int[Tensor, "batch_size context_length"]
            testing_label_token_sequences: Int[Tensor,
                                               "batch_size context_length"]
            # Reduce batch size for testing to save memory.
            testing_token_sequences, testing_label_token_sequences = get_batch(
                testing_tokens_mmap, batch_size=args.batch_size//4, context_length=args.context_length, device=args.device)

            with torch.no_grad():
                predicted_test_logits: Float[Tensor, "batch_size context_length vocab_size"] = transformer_lm(
                    testing_token_sequences)
                predicted_test_logits = rearrange(
                    predicted_test_logits, "batch_size context_length vocab_size -> (batch_size context_length) vocab_size")
                testing_label_token_sequences = rearrange(
                    testing_label_token_sequences, "batch_size context_length -> (batch_size context_length)")
                test_loss: Float[Tensor, ""] = cross_entropy(
                    predicted_test_logits, testing_label_token_sequences)
            logger.info(
                f"Testing loss at iteration {iteration}: {test_loss.item():.4f}")

            if wandb_run is not None:
                wandb_run.log({
                    "iter": iteration,
                    "val/loss": test_loss.item(),
                })

        # Save model checkpoint.
        if iteration % 5000 == 0 and iteration > 0:
            training_text_path = Path(args.training_token_file_path)
            model_checkpoint_path = training_text_path.with_stem(
                training_text_path.stem + f"_iter{iteration}").with_suffix(".pt")
            save_checkpoint(transformer_lm, adamw_optimizer,
                            iteration, model_checkpoint_path)

    if wandb_run is not None:
        wandb_run.finish()
