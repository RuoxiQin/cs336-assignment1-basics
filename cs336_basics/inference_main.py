"""
Run Transformer LM inference.

uv run python cs336_basics/inference_main.py \
--training_text_file_path="data/TinyStoriesV2-GPT4-train.txt" \
--load_model_checkpoint_path="data/TinyStoriesV2-GPT4-train-tokens_iter45000.pt" \
"""
import logging
import argparse
import torch
from dotenv import load_dotenv
from basic_modules import TransformerLM, load_checkpoint, softmax
from cs336_basics.tokenizer import BPETokenizer, get_merges_pkl_file_path, get_vocab_pkl_file_path


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
        description="A script to run inference of Transformer LM.")
    parser.add_argument("--training_text_file_path", type=str,
                        help="The text file that is used to train the Tokenizer.")
    parser.add_argument("--special_tokens", nargs="*", type=str,
                        default=["<|endoftext|>"], help="A list of special tokens str.")
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
    parser.add_argument("--max_output_tokens", type=int, default="512",)

    args = parser.parse_args()

    # Initialize tokenizer.
    vocab_path = get_vocab_pkl_file_path(args.training_text_file_path)
    merges_path = get_merges_pkl_file_path(args.training_text_file_path)
    tokenizer_args = {
        'vocab_path': vocab_path,
        'merges_path': merges_path,
        'special_tokens': args.special_tokens
    }
    tokenizer = BPETokenizer.from_files(
        tokenizer_args['vocab_path'],
        tokenizer_args['merges_path'],
        tokenizer_args['special_tokens']
    )

    # Initialize Transformer LM model.
    transformer_lm = TransformerLM(vocab_size=args.vocab_size, d_model=args.d_model, num_heads=args.num_heads,
                                   d_ff=args.d_ff, rope_theta=args.rope_theta, context_length=args.context_length, num_layers=args.num_layers).to(args.device)
    checkpoint = load_checkpoint(args.load_model_checkpoint_path, transformer_lm,
                                 optimizer=None)
    logger.info(
        f"Using checkpoint of iteration {checkpoint['iteration']}.")

    while True:
        prompt = input("Enter prompt (or 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break
        
        num_generated_tokens = 0
        while num_generated_tokens < args.max_output_tokens and prompt[-13:] != "<|endoftext|>":
            # Tokenize the prompt (including previously generated tokens)
            prompt_tokens: list[int] = tokenizer.encode(prompt)
            prompt_token_tensor = torch.tensor(prompt_tokens, dtype=torch.int32).unsqueeze(
                0).to(args.device)  # Shape: (1, seq_len)

            predicted_logits = transformer_lm(prompt_token_tensor)  # Shape: (1, seq_len, vocab_size)
            next_token_probablities = softmax(
                predicted_logits[-1, -1, :], dim=-1)  # Shape: (vocab_size,)
            next_token = torch.multinomial(
                next_token_probablities, num_samples=1)  # Shape: (1,)
            prompt_tokens += next_token.tolist()
            num_generated_tokens += 1
            prompt = tokenizer.decode(prompt_tokens)

        print(f"Generated story: {prompt}\n")
