import os.path

import regex as re
from collections import defaultdict
import sys
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
        word_dicts = process_map(count_words, chunks, chunksize=10)

    word_count = merge_dicts(word_dicts)
    pair_count = count_pair(word_count)

    vocab = initialize_vocab(special_tokens)
    initial_vocab_size = len(vocab)
    n_merges = vocab_size - initial_vocab_size
    merges = []
    for i in range(n_merges):
        max_pair = get_max_pair(pair_count)
        vocab[initial_vocab_size + i] = max_pair[0] + max_pair[1]
        merges.append(max_pair)
        # update pair count
        word_count, pair_count = update_count(word_count, pair_count, max_pair)

    return vocab, merges


if __name__ == "__main__":
    test_path = "/Users/a1/Desktop/cs336n/assignment1-basics/tests/fixtures/local_test.txt"
    vocab, merges = train_bpe(test_path,
                              vocab_size=500,
                              special_tokens=["<|endoftext|>"])
    print(vocab)
    print(merges)