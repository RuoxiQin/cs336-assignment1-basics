"""
Run tokenizer.

uv run python cs336_basics/run_tokenizer_main.py \
--training_text_file_path="data/TinyStoriesV2-GPT4-train.txt" \
--load_tokenizer_from_pkl \
--file_to_tokenize="data/TinyStoriesV2-GPT4-valid.txt" \
--verify_tokens
"""
import logging
import argparse
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections.abc import Iterable
import concurrent.futures
import multiprocessing
from cs336_basics.tokenizer import train_bpe_on_data, BPETokenizer, get_merges_pkl_file_path, get_vocab_pkl_file_path, find_chunk_boundaries

logger = logging.getLogger(__name__)


def get_tmp_output_file_path(file_path: str, chunk_i: int) -> str:
    input_path = Path(args.file_to_tokenize)
    output_tokens_path = input_path.with_stem(
        input_path.stem + f"-tokens-{chunk_i}").with_suffix(".dat")
    return str(output_tokens_path)


def tokenize_chunk(file_path: str, start: int, end: int, tokenizer_args: dict, output_tokens_path: str, queue) -> None:
    """Worker function to tokenize a specific byte range of a file."""
    # Re-initialize tokenizer inside the process
    tokenizer = BPETokenizer.from_files(
        tokenizer_args['vocab_path'],
        tokenizer_args['merges_path'],
        tokenizer_args['special_tokens']
    )

    with open(output_tokens_path, 'wb') as f_out, open(file_path, 'rb') as f_in:
        f_in.seek(start)
        bytes_remaining = end - start

        # Use a generator to stream read within the byte limits
        def chunk_line_generator() -> Iterable[str]:
            nonlocal bytes_remaining
            while bytes_remaining > 0:
                # Read line by line, but don't over-read the boundary
                line = f_in.readline(bytes_remaining)
                if not line:
                    break
                bytes_remaining -= len(line)
                queue.put(len(line))
                # pbar.update(len(line))
                yield line.decode('utf-8')

        # Use the tokenizer's iterable encoder
        for tokens in tokenizer.encode_iterable(chunk_line_generator()):
            if tokens:
                np.array(tokens, dtype=np.int32).tofile(f_out)


if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,                # Minimum level to log
        format="%(asctime)s [%(levelname)s] %(message)s",  # Log format
        handlers=[
            logging.StreamHandler(),       # Logs to console
            logging.FileHandler("data/app.log")  # Logs to file
        ]
    )

    parser = argparse.ArgumentParser(
        description="A script to train Tokenizer or encode file with it.")
    parser.add_argument("--training_text_file_path", type=str,
                        help="The text file to train the Tokenizer.")
    parser.add_argument("--vocab_size", type=int,  default=10000,
                        help="Vocabulary size for the trainig.",)
    parser.add_argument("--special_tokens", nargs="*", type=str,
                        default=["<|endoftext|>"], help="A list of special tokens str.")
    parser.add_argument("--load_tokenizer_from_pkl", action="store_true",
                        help="If true, load the trained tokenizer from pkl file.")
    parser.add_argument("--file_to_tokenize", type=str,
                        help="The path to the text file to tokenize.")
    parser.add_argument("--verify_tokens", action="store_true",
                        help="If True, verify the tokens by decoding the generated tokens back to text and compare against original text.")

    args = parser.parse_args()

    if not args.load_tokenizer_from_pkl:
        train_bpe_on_data(args.training_text_file_path,
                          args.vocab_size, args.special_tokens)

    if args.file_to_tokenize:
        logger.info(
            f"Loading vocab and merges trained on {args.training_text_file_path}.")
        vocab_path = get_vocab_pkl_file_path(args.training_text_file_path)
        merges_path = get_merges_pkl_file_path(args.training_text_file_path)
        tokenizer_args = {
            'vocab_path': vocab_path,
            'merges_path': merges_path,
            'special_tokens': args.special_tokens
        }

        input_path = Path(args.file_to_tokenize)
        output_tokens_path = input_path.with_stem(
            input_path.stem + "-tokens").with_suffix(".dat")
        input_file_size = os.path.getsize(input_path)

        # Split the input text file and determine boundaries.
        num_workers = multiprocessing.cpu_count()
        with open(args.file_to_tokenize, "rb") as f:
            # Split by b'<|endoftext|>\n'
            boundaries = find_chunk_boundaries(
                f, num_workers, b"<|endoftext|>\n")
        # Create (start, end) pairs
        tasks = [(boundaries[i], boundaries[i+1])
                 for i in range(len(boundaries)-1)]

        # Tokenize text chunks in parallel.
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        all_chunks_tokens = []
        with tqdm(total=input_file_size, unit='B', unit_scale=True, desc="Tokenizing") as pbar, concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Map the tasks to the worker function
            futures = [
                executor.submit(tokenize_chunk, args.file_to_tokenize, s, e,
                                tokenizer_args, get_tmp_output_file_path(args.file_to_tokenize, i), progress_queue)
                for i, (s, e) in enumerate(tasks)
            ]

            processed_file_bytes = 0
            while processed_file_bytes < input_file_size:
                newly_processed_bytes = progress_queue.get()
                processed_file_bytes += newly_processed_bytes
                pbar.update(newly_processed_bytes)
        
        [future.result() for future in futures]

        # Merge the files from above parallel process outputs.
        with open(output_tokens_path, 'wb') as f_final:
            for i in range(len(tasks)):
                temp_file_path = get_tmp_output_file_path(
                    args.file_to_tokenize, i)
                with open(temp_file_path, 'rb') as f_temp:
                    f_final.write(f_temp.read())
                os.remove(temp_file_path)

        logger.info(
            f"Finished tokenizing {args.file_to_tokenize} and written to {output_tokens_path}.")

        if args.verify_tokens:
            # Decode the token to text and compare it against the original file content.
            tokens_mmap = np.memmap(
                output_tokens_path,
                dtype=np.int32,
                mode='r'
            )
            tokenizer = BPETokenizer.from_files(
                tokenizer_args["vocab_path"], tokenizer_args["merges_path"], tokenizer_args["special_tokens"])
            decoded_text = tokenizer.decode(tokens_mmap.tolist())

            with open(input_path, "r") as f:
                # Assume the original_contents can fit to memory as verification usually runs on small dataset.
                original_contents = f.read()
            if original_contents == decoded_text:
                logger.info("Successfully verified the decoded text matches the original content.")
            else:
                logger.error("Verification failed! Decoded text does NOT match the original content!")
