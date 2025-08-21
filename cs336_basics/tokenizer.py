from collections import OrderedDict, defaultdict
import heapq
import logging
import os
from pathlib import Path
import pickle
import regex as re
from typing import BinaryIO

logger = logging.getLogger(__name__)


def train_bpe_on_tinystoriesv2_train():
    return train_bpe_on_data("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])


def train_bpe_on_tinystoriesv2_valid():
    return train_bpe_on_data("data/TinyStoriesV2-GPT4-valid.txt", 10000, ["<|endoftext|>"])


def train_bpe_on_owt_train():
    return train_bpe_on_data("data/owt_train.txt", 32000, ["<|endoftext|>"])


def train_bpe_on_owt_valid():
    return train_bpe_on_data("data/owt_valid.txt", 32000, ["<|endoftext|>"])


def train_bpe_on_data(path_str: str, vocab_size: int, special_tokens: list[str]):
    logger.info(f"Started BPE training for {path_str}.")
    vocab, merges = train_bpe(
        path_str, vocab_size, special_tokens)
    logger.info(f"Completed BPE training for {path_str}.")
    path = Path(path_str)
    with open(path.with_stem(path.stem + "-vocab").with_suffix(".pkl"), "wb") as f:
        pickle.dump(vocab, f)
    with open(path.with_stem(path.stem + "-merges").with_suffix(".pkl"), "wb") as f:
        pickle.dump(merges, f)


def load_vocab_and_merges(training_data_path: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    path = Path(training_data_path)
    with open(path.with_stem(path.stem + "-vocab").with_suffix(".pkl"), "rb") as f:
        vocab: dict[int, bytes] = pickle.load(f)
    with open(path.with_stem(path.stem + "-merges").with_suffix(".pkl"), "rb") as f:
        merges: list[tuple[bytes, bytes]] = pickle.load(f)
    return vocab, merges


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token,
                      bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    logger.info("Tokenizer started.")
    # Initialize vocab with 256 bytes and special tokens.
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, special_token in enumerate(special_tokens):
        vocab[len(vocab) + i] = special_token.encode("utf-8")

    # Split the input text by special_tokens.
    with open(input_path, "r", encoding="utf-8") as f:
        input_text = f.read()
    split_pattern = "|".join(re.escape(special_token)
                             for special_token in special_tokens)
    text_parts = re.split(split_pattern, input_text)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    words_frequency: dict[str, int] = defaultdict(int)
    for text_part in text_parts:
        # Pre-tokenization.
        pre_tokenized_words = re.findall(PAT, text_part)
        for word in pre_tokenized_words:
            # Build words frequency table.
            words_frequency[word] += 1
    logger.info("Pre-tokenization completed.")
    # Count byte-pair frequency.
    # Double-linked list of the words bytes sequence. Only the left-most bytes of a merged bytes are valid.
    # Note that this can be further optimized by storing word_frequency, words, words_bytes in a single object.
    words: list[str] = []
    words_bytes: list[list[bytes]] = []
    next: list[list[int]] = []
    prev: list[list[int]] = []

    # BytesPair that supports inversed order. Used in min-heap.
    class BytesPairWithInvseredOrder:
        bytes_pair: tuple[bytes, bytes]

        def __init__(self, bytes_pair: tuple[bytes, bytes]):
            self.bytes_pair = bytes_pair

        def __lt__(self, other):
            return self.bytes_pair > other.bytes_pair

    # Stores the bytes-pair to occurance (word_index, byte_position_index). Use OrderedDict to preserve insertion order so it can be
    # popped in order. The value of OrderedDict is a dummy None. (Ideally we'd like OrderedSet)
    bytes_pair_occurance: dict[tuple[bytes, bytes], OrderedDict[tuple[int, int], None]] = defaultdict(
        OrderedDict)
    bytes_pair_frequency: dict[tuple[bytes, bytes], int] = defaultdict(int)
    bytes_pair_frequency_heap: list[tuple[int,
                                          BytesPairWithInvseredOrder]] = []

    def increase_bytes_pair_frequency(bytes_pair: tuple[bytes, bytes], count: int):
        bytes_pair_frequency[bytes_pair] += count
        updated_count = bytes_pair_frequency[bytes_pair]
        heapq.heappush(bytes_pair_frequency_heap, (-updated_count,
                       BytesPairWithInvseredOrder(bytes_pair)))

    def get_most_frequent_bytes_pair():
        while bytes_pair_frequency_heap and bytes_pair_frequency[bytes_pair_frequency_heap[0][1].bytes_pair] != -bytes_pair_frequency_heap[0][0]:
            heapq.heappop(bytes_pair_frequency_heap)
        if not bytes_pair_frequency_heap:
            return None
        return bytes_pair_frequency_heap[0][1].bytes_pair

    def reduce_bytes_pair_frequency(bytes_pair: tuple[bytes, bytes], count):
        frequency = bytes_pair_frequency[bytes_pair]
        assert frequency >= count, f"{bytes_pair} frequency {frequency} is lower than {count}."
        if frequency > count:
            bytes_pair_frequency[bytes_pair] -= word_frequency
            updated_frequency = bytes_pair_frequency[bytes_pair]
            heapq.heappush(bytes_pair_frequency_heap,
                           (-updated_frequency, BytesPairWithInvseredOrder(bytes_pair)))
        else:
            del bytes_pair_frequency[bytes_pair]

    def delete_bytes_pair_occurance(bytes_pair: tuple[bytes, bytes], occurance: tuple[int, int]):
        bytes_pair_occurance[bytes_pair].pop(occurance)
        if not bytes_pair_occurance[bytes_pair]:
            del bytes_pair_occurance[bytes_pair]

    merges = []
    for word_i, (word, count) in enumerate(words_frequency.items()):
        words.append(word)
        word_bytes = word.encode("utf-8")
        words_bytes.append([word_bytes[i:i+1] for i in range(len(word_bytes))])
        next.append([i for i in range(1, len(word_bytes))] + [-1])
        prev.append([-1] + [i for i in range(len(word_bytes) - 1)])
        for i in range(len(word_bytes) - 1):
            bytes_pair = (bytes([word_bytes[i]]), bytes([word_bytes[i+1]]))
            increase_bytes_pair_frequency(bytes_pair, count)
            bytes_pair_occurance[bytes_pair][(word_i, i)] = None
    logger.info("Tokenizer initial frequency tables creation completed.")

    # BPE merge loop.
    for vocab_i in range(len(vocab), vocab_size):
        if vocab_i % 100 == 0:
            logger.info(f"Starting BPE merge loop {vocab_i}.")
        if not bytes_pair_frequency:
            break
        # Get the most frequent bytes_pair. If there is a tie, further sort lexicographically by bytes_pairs.
        most_frequent_bytes_pair = get_most_frequent_bytes_pair()
        if not most_frequent_bytes_pair:
            logger.warning("Empty most_frequent_bytes_pair!!!")
            break
        merges.append(most_frequent_bytes_pair)
        merged_bytes = most_frequent_bytes_pair[0] + \
            most_frequent_bytes_pair[1]
        vocab[vocab_i] = merged_bytes

        # Update the merged byte-pairs and adjacent byte-pairs frequency tables.
        while bytes_pair_occurance[most_frequent_bytes_pair]:
            word_i, bytes_i = bytes_pair_occurance[most_frequent_bytes_pair].popitem(last=False)[
                0]
            second_bytes_i = next[word_i][bytes_i]
            word_bytes = words_bytes[word_i]
            word_frequency = words_frequency[words[word_i]]
            assert word_bytes[bytes_i] == most_frequent_bytes_pair[0]
            assert word_bytes[second_bytes_i] == most_frequent_bytes_pair[1]
            # Update the bytes before the merged pair (if exists).
            before_bytes_i = prev[word_i][bytes_i]
            if before_bytes_i >= 0:
                before_bytes = word_bytes[before_bytes_i]
                reduce_bytes_pair_frequency(
                    (before_bytes, most_frequent_bytes_pair[0]), word_frequency)
                delete_bytes_pair_occurance(
                    (before_bytes, most_frequent_bytes_pair[0]), (word_i, before_bytes_i))
                increase_bytes_pair_frequency(
                    (before_bytes, merged_bytes), word_frequency)
                bytes_pair_occurance[(before_bytes, merged_bytes)][(
                    word_i, before_bytes_i)] = None
            # Update the bytes after the merged pair (if exists).
            after_bytes_i = next[word_i][second_bytes_i]
            if after_bytes_i >= 0:
                next_bytes = word_bytes[after_bytes_i]
                reduce_bytes_pair_frequency(
                    (most_frequent_bytes_pair[1], next_bytes), word_frequency)
                delete_bytes_pair_occurance(
                    (most_frequent_bytes_pair[1], next_bytes), (word_i, second_bytes_i))
                increase_bytes_pair_frequency((
                    merged_bytes, next_bytes), word_frequency)
                bytes_pair_occurance[(merged_bytes, next_bytes)
                                     ][(word_i, bytes_i)] = None
            # Update double-linked list pointers
            next[word_i][bytes_i] = after_bytes_i
            if after_bytes_i >= 0:
                prev[word_i][after_bytes_i] = bytes_i
            # Update word_bytes with the merged bytes and delete itself from the frequency tables.
            word_bytes[bytes_i] = merged_bytes
            reduce_bytes_pair_frequency(
                most_frequent_bytes_pair, word_frequency)
            # No need to remove from bytes_pair_occurance because it's already popped at beginning of the loop.
        del bytes_pair_occurance[most_frequent_bytes_pair]

    return vocab, merges
