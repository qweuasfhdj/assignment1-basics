from idlelib.iomenu import errors

import yaml

import bpe
import os

curr_path = os.path.dirname(os.path.abspath(__file__))
# tiny_story_path = os.path.join(curr_path, '../data/tinystories_sample.txt')
# tiny_story_path = os.path.join(curr_path, '../data/tinystories_sample_5M.txt')
tiny_story_path = os.path.join(curr_path, '../data/TinyStoriesV2-GPT4-train.txt')

save_path = os.path.join(curr_path, '../data/tiny_stories_result.txt')

def save_vocab(vocab, merges, save_path):
    """
    Save the vocabulary into a file
    :param vocab:
    :return:
    """
    encoded_data = {k: v.decode('utf-8', errors="replace") for k, v in vocab.items()}
    merges_data = [(a.decode('utf-8', errors="replace"), b.decode('utf-8', errors="replace")) for a, b in merges]

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump({
            "vocab": encoded_data,
            "merges": merges_data
        }, f, allow_unicode=True)
    return


if __name__ == '__main__':
    input_path = tiny_story_path
    vocab, merges = bpe.train_bpe(
        input_path=input_path,
        vocab_size=10000,
        special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"],
    )
    save_vocab(vocab, merges, save_path)
    # sorted_vocab = sorted(vocab.items(), key=lambda x: len(x[1]), reverse=True)
    # print(sorted_vocab)