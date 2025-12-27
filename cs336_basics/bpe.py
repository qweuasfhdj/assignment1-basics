import os.path

import regex as re
from collections import defaultdict
from typing import Iterable, Iterator
import json
import sys

from torchgen.api.types import boolT
from tqdm.contrib.concurrent import process_map
import pathlib
# GPT2's token split
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def initialize_vocab(special_tokens):
    vocab = {token: bytes([token]) for token in range(256)}
    for i, token in enumerate(special_tokens):
        vocab[i + 256] = token.encode('utf-8')
    return vocab


def word2bytes(word):
    word_list = list(word.encode('utf-8'))
    return tuple(bytes([w]) for w in word_list)


def merge_dicts(dicts: defaultdict):
    merged_dict = defaultdict(int)
    for dict in dicts:
        for k, v in dict.items():
            merged_dict[k] += v
    return merged_dict


def count_words(text):
    "Split text into word bytes using GPT2 pattern and count word bytes frequency."
    word_count = defaultdict(int)
    for match in re.finditer(PAT, text):
        word = match.group(0)
        word_bytes = word2bytes(word)
        if len(word_bytes) >= 2:
            word_count[word_bytes] += 1
    return word_count


def get_max_pair(pair_count):
    if len(pair_count) == 0:
        return None
    max_pair, _ = max(pair_count.items(), key=lambda pair: (pair[1], pair[0]))
    return max_pair


def split_by_special(text, special_tokens, drop_special=True):
    if not special_tokens:
        return [text]
    special_tokens = sorted(special_tokens, key=len, reverse=True)

    # 2. 转义所有特殊字符 如 ., \
    escaped_tokens = [re.escape(token) for token in special_tokens]

    # 3. 构建分割模式：(token1|token2|...)
    pattern_str = "|".join(escaped_tokens)

    # 3. 根据drop_special决定是否加括号
    # re.split(pattern, text)：分隔符不保留
    # re.split((pattern), text)：分隔符保留在结果中
    if not drop_special:
        # 加括号的话，分隔符就会出现在分割结果里
        pattern_str = f"({pattern_str})"
    else:
        pattern_str = pattern_str
    pattern = re.compile(pattern_str)
    chunks = pattern.split(text)
    # 去除空字节
    return [c for c in chunks if c]


def count_pair(word_counts):
    pair_count = defaultdict(int)
    for word_bytes, count in word_counts.items():
        for pair in zip(word_bytes[:-1], word_bytes[1:]):
            pair_count[pair] += count
    return pair_count


def apply_merge(word_bytes, merge_pair):
    """
    replace original word bytes with merged word bytes.
    "world", "wo" --> "wo" "r" "l" "d"
    :param word_bytes:
    :param merge_pair:
    :return:
    """
    a, b = merge_pair
    merged = merge_pair[0]+merge_pair[1]
    result = []
    n = len(word_bytes)
    i = 0
    while i < n:
        if word_bytes[i] == a and i + 1 < n and word_bytes[i + 1] == b:
            result.append(merged)
            i = i + 2
        else:
            result.append(word_bytes[i])
            i = i + 1
    return tuple(result)

def update_count(word_count, pair_count, merge_pair):
    new_word_count = defaultdict(int)
    new_pair_count = defaultdict(int, pair_count) # copy old pair count
    for word_bytes, count in word_count.items():
        old_pairs = list(zip(word_bytes[:-1], word_bytes[1:]))
        if merge_pair not in old_pairs:
            new_word_count[word_bytes] += count
            continue

        # use updated key to replace the separated two code
        new_word = apply_merge(word_bytes, merge_pair)
        new_word_count[new_word] += count

        # decrease all old pair counts
        for pair in old_pairs:
            new_pair_count[pair] -= count
            if new_pair_count[pair] == 0:
                del new_pair_count[pair]

        # count new pair in new word
        new_pairs = list(zip(new_word[: -1], new_word[1:]))
        for new_pair in new_pairs:
            new_pair_count[new_pair] += count

    return new_word_count, new_pair_count


def train_bpe(input_path, vocab_size, special_tokens):
    """
    :param input_path:
    :param vocab_size:
    :param special_tokens:
    :return:
    """
    text = read_text(input_path)
    chunks = split_by_special(text, special_tokens)
    if len(chunks) < 4:
        word_dicts = list(map(count_words, chunks))
    else:
        # 返回一个List of result. concurrent 操作。
        word_dicts = process_map(count_words, chunks, chunksize=1000)

    word_count = merge_dicts(word_dicts)
    pair_count = count_pair(word_count)

    vocab = initialize_vocab(special_tokens)
    initial_vocab_size = len(vocab)
    n_merges = vocab_size - initial_vocab_size
    merges = []
    for i in range(n_merges):
        max_pair = get_max_pair(pair_count)
        if max_pair is None:
            continue
        vocab[initial_vocab_size + i] = max_pair[0] + max_pair[1]
        merges.append(max_pair)
        # update pair count
        word_count, pair_count = update_count(word_count, pair_count, max_pair)

    return vocab, merges

def split_to_words(text):
    return PAT.findall(text)

def apply_merges(word_bytes, merges, vocab_to_id):
    word_bytes = list(word_bytes)
    while True:
        min_token_id = float('inf')
        best_pair_idx = -1
        merged = None

        for i in range(len(word_bytes) - 1):
            pair = (word_bytes[i], word_bytes[i + 1])
            if pair in merges:
                combined = pair[0] + pair[1]
                token_id = vocab_to_id[combined]
                if token_id is not None and token_id < min_token_id:
                    # BPE 的训练过程中，先学到的合并规则优先级更高
                    # 词汇表通常是按 token_id 排序的，早期学到的合并有更小的 ID
                    # 这确保了合并顺序与训练时一致
                    min_token_id = min(min_token_id, token_id)
                    best_pair_idx = i
                    merged = combined

        if best_pair_idx == -1:
            break

        word_bytes = (
            word_bytes[:best_pair_idx] +
            [merged] +
            word_bytes[best_pair_idx + 2:]
        )

    return tuple(word_bytes)

def encode_merged(text, merges, vocab_to_id):
    """
    text -> word_list -> word bytes -> find merges and map to merged tokens
    :param text:
    :param merges:
    :param vocab_to_id:
    :return:
    """
    word_list = split_to_words(text)
    tokens = []
    for word in word_list:
        word_bytes = word2bytes(word)
        merged_word_bytes = apply_merges(word_bytes, merges, vocab_to_id)
        tokens.extend(vocab_to_id[i] for i in merged_word_bytes)
    return tokens

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = set(merges)
        self.special_tokens = special_tokens if special_tokens else []
        self.vocab_to_id = {v:k for k,v in self.vocab.items()}

        for token_bytes in self.special_tokens:
            if token_bytes not in self.vocab_to_id:
                new_id = len(self.vocab)
                self.vocab_to_id[token_bytes] = new_id
                self.vocab[new_id] = token_bytes

    @classmethod
    def from_file(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)
            # Optional: convert keys to int if stored as strings
            vocab = {int(k): bytes(v, 'latin1') if isinstance(v, str) else bytes(v)
                     for k, v in vocab_data.items()}
        # load merges
        with open(merges_filepath, "r", encoding="utf-8") as mf:
            lines = mf.readlines()
            merge_pairs = [tuple(line.strip().split()) for line in lines if not line.startswith('#') and line.strip()]
            # Convert to byte-pairs
            merges = [(a.encode('utf-8'), b.encode('utf-8')) for a, b in merge_pairs]
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        chunks = split_by_special(text, self.special_tokens, drop_special=False)
        tokens = []
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                tokens.append(self.vocab_to_id[chunk.encode('utf-8')])
            else:
                tokens.extend(encode_merged(chunk, self.merges, self.vocab_to_id))
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        return b''.join(self.vocab[t] for t in ids).decode('utf-8', errors='replace')


if __name__ == "__main__":
    test_path = "/Users/a1/Desktop/cs336n/assignment1-basics/tests/fixtures/local_test.txt"
    vocab, merges = train_bpe(test_path,
                              vocab_size=500,
                              special_tokens=["<|endoftext|>"])
    print(vocab)
    print(merges)