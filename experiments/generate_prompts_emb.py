'''beam search for prompts that can probe the bias'''

import time
import logging
import random
import pickle
import argparse
import torch
import numpy as np
import  math
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaModel
from transformers import AlbertTokenizer, AlbertForMaskedLM, AlbertModel
from def_sent_utils import get_def_pairs
from tqdm import tqdm, trange


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--debias_type",
    default='gender',
    type=str,
    choices=['gender', 'race'],
    help="Choose from ['gender','race']",
)

parser.add_argument(
    "--model_name_or_path",
    default="roberta-base",
    type=str,
    help="Path to pretrained model or model identifier from huggingface.co/models. Choose from ['bert-base-uncased','roberta-base','albert-base-v2']",
)

parser.add_argument(
    "--load_path",
    default="",
    type=str,
    help="Path to pretrained model or model identifier from huggingface.co/models. Choose from ['bert-base-uncased','roberta-base','albert-base-v2']",
)


parser.add_argument(
    "--model_type",
    default="roberta",
    type=str,
    help="Choose from ['bert','roberta','albert']",
)

parser.add_argument(
    "--vocab_file",
    default='data/wiki_words_5000.txt',
    type=str,
    help="Path to the file that stores the vocabulary of the prompts",
)

parser.add_argument(
    "--BS",
    default=128,  # 1000
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
    default=20,  # 100
    type=int,
    help="top K prompts to be selected in the beam search",
)

parser.add_argument(
    "--def_pairs_name",
    default="all",
    type=str,
    help="Name of definitional sentence pairs.",
)


def get_cos(tar1_sen, tar2_sen, model):
    cos_list = []
    tar_input1 = tokenizer(tar1_sen, padding=True, truncation=True, max_length=128, return_tensors="pt")
    tar_input2 = tokenizer(tar2_sen, padding=True, truncation=True, max_length=128, return_tensors="pt")
    for i in range(tar_input1['input_ids'].shape[0] // args.BS + 1):
        tar_input_id1 = {}
        tar_input_id2 = {}
        try:
            for key in tar_input1.keys():
                tar_input_id1[key] = tar_input1[key][args.BS * i:args.BS * (i + 1)].cuda()
                tar_input_id2[key] = tar_input2[key][args.BS * i:args.BS * (i + 1)].cuda()

        except IndexError:
            for key in tar_input1.keys():
                tar_input_id1[key] = tar_input1[key][args.BS * i:].cuda()
                tar_input_id2[key] = tar_input2[key][args.BS * i:].cuda()
        output_a = model(**tar_input_id1).pooler_output
        output_b = model(**tar_input_id2).pooler_output
        cos_matrix = torch.nn.functional.cosine_similarity(output_a, output_b, dim=1).detach().cpu().numpy()
        cos_list.extend(list(cos_matrix))
        torch.cuda.empty_cache()
    return cos_list


def get_prompt(example_a, example_b, prompts):
    tar1_sen = []
    tar2_sen = []
    for j in range(len(prompts)):
        tar1_sen.append(example_a + ' ' + prompts[j])
        tar2_sen.append(example_b + ' ' + prompts[j])
    return tar1_sen, tar2_sen


def get_prompt_cos(prompts, model, def_examples):
    cos_word_list = []
    for (i, example) in enumerate(tqdm(def_examples)):
        tar1_sen, tar2_sen = get_prompt(example.text_a, example.text_b, prompts)
        cos_list = get_cos(tar1_sen, tar2_sen, model)
        cos_word_list.append(cos_list)
    cos_word_list = np.array(cos_word_list)
    return np.mean(cos_word_list, axis=0)


def findpos_new(words, sent):
    sent_list = sent.split(' ')
    for i, token in enumerate(sent_list):
        if token in words:
           return token
    return None


def get_def_examples_new(def_pairs, words_1, words_2):
    '''Construct definitional examples from definitional pairs.'''
    def_examples = []
    for group_id in def_pairs:
        def_group = def_pairs[group_id]
        f_sents = def_group['f']
        m_sents = def_group['m']
        m_refer = set(words_1)
        f_refer = set(words_2)
        for sent_id, (sent_a, sent_b) in enumerate(zip(f_sents, m_sents)):
            word_a = findpos_new(f_refer, sent_a)
            word_b = findpos_new(m_refer, sent_b)
            if word_a!=None and word_b!=None:
                def_examples.append(InputExample(guid='{}-{}'.format(group_id, sent_id),
                                                 text_a=sent_a, text_b=sent_b, label=None, key_a=word_a, key_b=word_b))
    return def_examples


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None,key_a=None,key_b=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.key_a = key_a # the bias word position
        self.key_b = key_b # the bias word position


if __name__ == "__main__":
    args = parser.parse_args()

    if args.model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        model = BertModel.from_pretrained(args.model_name_or_path)
    elif args.model_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.load_path or args.model_name_or_path)
        model = RobertaModel.from_pretrained(args.load_path or args.model_name_or_path)
    elif args.model_type == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained(args.model_name_or_path)
        model = AlbertModel.from_pretrained(args.model_name_or_path)
    else:
        raise NotImplementedError("not implemented!")

    device_ids = []
    model = nn.DataParallel(model, device_ids=device_ids)
    model.eval()
    model.cuda()

    jsd_model = JSD(reduction='none')

    if args.debias_type == 'gender':
        words_1 = ["man","boy","he","father","son", "guy","male","his","himself","John"]
        words_2 = ["woman","girl","she","mother","daughter","gal","female","her","herself","Mary"]

    elif args.debias_type == 'race':
        words_2 = ["europe", "european", "russian", "italian", "spanish", "irish", "british", "french", "greek",
                   "german", "romanian", "polish", "portuguese", "bosnian", "albanian", "caucasian", "caucasian",
                   "caucasian", "caucasian", "caucasian", "caucasian", "caucasian", "caucasian", "caucasian", "caucasian",
                   "caucasian", "caucasian", "caucasian", "caucasian", "caucasian"]
        words_1 = ["african", "african", "african", "african", "african", "african", "african", "african", "african",
                   "african", "african", "african", "african", "african", "african", "african",
                   "nigerian", "ethiopian", "sudanese", "africa", "somali", "ghanaian", "moroccan", "eritrean",
                   "kenyan", "liberian", "cameroonian", "mexican", "south-african", "egyptian"]

    def_pairs = get_def_pairs(args.def_pairs_name, args.debias_type)
    def_examples = get_def_examples_new(def_pairs, words_1, words_2)  # 1D list where 2i and 2i+1 are a pair

    vocab = load_wiki_word_list(args.vocab_file)
    vocab = clean_word_list(vocab, tokenizer)  # vocabulary in prompts

    current_prompts = vocab

    f = open('data/{}/prompts_cos_{}_{}_{}'.format(args.model_type, args.model_name_or_path, args.debias_type, time.time()), 'w')
    f_200 = open('data/{}/prompts_cos_top200_{}'.format(args.model_type, time.time()), 'w')
    top_200_prompts = []
    for m in range(args.PL):
        if args.debias_type == 'gender':
            current_prompts_cos = get_prompt_cos(current_prompts, model, def_examples)
        elif args.debias_type == 'race':
            current_prompts_cos = get_prompt_cos(current_prompts, model, def_examples)
            print(current_prompts_cos)
        top_k_prompts = np.array(current_prompts)[np.argsort(current_prompts_cos)[:args.K]]
        print(top_k_prompts)
        if m == 0:
            top_200_prompts = np.array(current_prompts)[np.argsort(current_prompts_cos)[:200]]
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
