import logging
from cs336_basics.tokenizer import train_bpe_on_tinystoriesv2_train, load_vocab_and_merges, BPETokenizer

logger = logging.getLogger(__name__)


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

    # train_bpe_on_tinystoriesv2_train()
    vocab, merges = load_vocab_and_merges("data/TinyStoriesV2-GPT4-train.txt")
    tokenizer = BPETokenizer.from_files(
        "data/TinyStoriesV2-GPT4-train-vocab.pkl", "data/TinyStoriesV2-GPT4-train-merges.pkl")
    encoding_result = tokenizer.encode("🙃")
    print(encoding_result)
    print(tokenizer.decode(encoding_result))
