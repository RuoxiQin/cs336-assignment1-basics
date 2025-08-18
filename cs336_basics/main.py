

import logging
from tests.adapters import run_train_bpe

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,                # Minimum level to log
        format="%(asctime)s [%(levelname)s] %(message)s",  # Log format
        handlers=[
            logging.StreamHandler(),       # Logs to console
            logging.FileHandler("data/app.log") # Logs to file
        ]
    )
    # with open(input_path, "rb") as f:
    #     num_processes = 8
    #     boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    #     # The following is a serial implementation, but you can parallelize this
    #     # by sending each start/end pair to a set of processes.
    #     for start, end in zip(boundaries[:-1], boundaries[1:]):
    #         f.seek(start)
    #         print("Chunk 1:", f.read(end - start))
    #         chunk = f.read(end - start).decode("utf-8", errors="ignore")
    vocab, merges = run_train_bpe("data/owt_valid.txt", 32000, ["<|endoftext|>"])
    logger.info("BPE training completed.")
    with open("data/owt-valid-vocab.txt", "w", encoding="utf-8") as f:
        f.write(str(vocab))
    with open("data/owd-valid-merges.txt", "w", encoding="utf-8") as f:
        f.write(str(merges))
