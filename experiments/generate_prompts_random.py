'''beam search for prompts that can probe the bias'''

import time
import logging
import argparse
import torch
import numpy as np
from utils import *
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--vocab_file",
    default='data/wiki_words_5000.txt',
    type=str,
    help="Path to the file that stores the vocabulary of the prompts",
)

parser.add_argument(
    "--model_name_or_path",
    default="bert-base-uncased",
    type=str,
    help="Path to pretrained model or model identifier from huggingface.co/models",
)

parser.add_argument(
    "--BS",
    default=1000,  # 1000
    type=int,
    help="batch size of the data fed into the model",
)

parser.add_argument(
    "--PL",
    default=5,
    type=int,
    help="maximun length of the generated prompts",
)

parser.add_argument(
    "--K",
    default=10,  # 100
    type=int,
    help="top K prompts to be selected in the beam search",
)


if __name__ == "__main__":
    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    vocab = load_wiki_word_list(args.vocab_file)
    vocab = clean_word_list(vocab, tokenizer)  # vocabulary in prompts
    current_prompts = vocab

    f = open('data/prompts_random_{}'.format(time.time()), 'w')
    f_200 = open('data/prompts_random_top200_{}'.format(time.time()), 'w')
    top_200_prompts = []
    for m in range(args.PL):
        np.random.shuffle(current_prompts)
        top_k_prompts = np.array(current_prompts)[:args.K]
        print(top_k_prompts)

        if m == 0:
            top_200_prompts = np.array(current_prompts)[:200]
            for k_1 in top_200_prompts:
                f_200.write(k_1)
                f_200.write("\n")
            f_200.close()

        for p in top_k_prompts:
            f.write(p)
            f.write("\n")
        new_prompts = []
        for tkp in top_k_prompts:
            for v in top_200_prompts:
                new_prompts.append(tkp + " " + v)
        current_prompts = new_prompts
        print("search space size:", len(current_prompts))
    f.close()
