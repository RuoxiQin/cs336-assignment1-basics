import logging
from cs336_basics.tokenizer import train_bpe_on_tinystoriesv2_train, load_vocab_and_merges, BPETokenizer, train_bpe_on_tinystoriesv2_valid

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
    # train_bpe_on_tinystoriesv2_valid()

    vocab, merges = load_vocab_and_merges("data/TinyStoriesV2-GPT4-valid.txt")
    # print(vocab)
    tokenizer = BPETokenizer.from_files(
        "data/TinyStoriesV2-GPT4-valid-vocab.pkl", "data/TinyStoriesV2-GPT4-valid-merges.pkl", ["<|endoftext|><|endoftext|>", "<|endoftext|>"])
    test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"
    ids_together = tokenizer.encode(test_string)
    ids = []
    for id in tokenizer.encode_iterable(test_string):
        ids.append(id)
    tokenized_string = [tokenizer.decode([x]) for x in ids]
    print(tokenized_string)
    print(ids_together == ids)
