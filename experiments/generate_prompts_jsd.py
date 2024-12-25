'''beam search for prompts that can probe the bias'''

import time
import os
import logging
import random
import pickle
import argparse
import torch
import numpy as np
from utils import *
from transformers import BertTokenizer, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import AlbertTokenizer, AlbertForMaskedLM
from def_sent_utils import get_def_pairs
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)

FEMALES =["woman","girl","she","mother","daughter","gal","female","her","herself","Mary"]
MALES = ["man","boy","he","father","son", "guy","male","his","himself","John"]

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
    default="albert-base-v2",
    type=str,
    help="Path to pretrained model or model identifier from huggingface.co/models",
)

parser.add_argument(
    "--model_type",
    default="albert",
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
    default=256,  # 1000
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

parser.add_argument(
    "--def_pairs_name",
    default="all",
    type=str,
    help="Name of definitional sentence pairs.",
)

def send_to_cuda(tar1_tokenized, tar2_tokenized):
    for key in tar1_tokenized.keys():
        tar1_tokenized[key] = tar1_tokenized[key].cuda()
        tar2_tokenized[key] = tar2_tokenized[key].cuda()
    return tar1_tokenized, tar2_tokenized


def get_tokenized_ith_prompt(prompts, text_a, text_b, tokenizer):
    tar1_sen_i = []
    tar2_sen_i = []

    for i in range(len(prompts)):
        tar1_sen_i.append(text_a + ' ' + prompts[i])
        tar2_sen_i.append(text_b + ' ' + prompts[i])
    tar1_tokenized = tokenizer(tar1_sen_i, padding=True, truncation=True, return_tensors="pt")
    tar2_tokenized = tokenizer(tar2_sen_i, padding=True, truncation=True, return_tensors="pt")
    tar1_mask_index = np.where(tar1_tokenized['input_ids'].numpy() == tokenizer.mask_token_id)[1]
    tar2_mask_index = np.where(tar2_tokenized['input_ids'].numpy() == tokenizer.mask_token_id)[1]
    print([tokenizer.convert_ids_to_tokens(int(id_)) for id_ in tar1_tokenized['input_ids'][0]])
    assert tar1_mask_index.shape[0] == tar1_tokenized['input_ids'].shape[0]
    return tar1_tokenized, tar2_tokenized, tar1_mask_index, tar2_mask_index


def run_model(model, inputs, mask_index, ster_words):
    predictions = model(**inputs)
    predictions_logits = predictions.logits[np.arange(inputs['input_ids'].size(0)), mask_index][:, ster_words]

    return predictions_logits


def get_JSD(tar1_tokenized, tar2_tokenized, tar1_mask_index, tar2_mask_index, model, ster_words_male, ster_words_female):
    jsd_list = []
    tar1_tokenized, tar2_tokenized = send_to_cuda(tar1_tokenized, tar2_tokenized)
    for k in range(tar1_tokenized['input_ids'].shape[0] // args.BS + 1):
        tar1_inputs = {}
        tar2_inputs = {}
        try:
            for key in tar1_tokenized.keys():
                tar1_inputs[key] = tar1_tokenized[key][args.BS * k:args.BS * (k + 1)]
                tar2_inputs[key] = tar2_tokenized[key][args.BS * k:args.BS * (k + 1)]

            tar1_local_mask_index = tar1_mask_index[args.BS * k:args.BS * (k + 1)]
            tar2_local_mask_index = tar2_mask_index[args.BS * k:args.BS * (k + 1)]
        except IndexError:
            for key in tar1_tokenized.keys():
                tar1_inputs[key] = tar1_tokenized[key][args.BS * (k + 1):]
                tar2_inputs[key] = tar2_tokenized[key][args.BS * (k + 1):]

            tar1_local_mask_index = tar1_mask_index[args.BS * (k + 1):]
            tar2_local_mask_index = tar2_mask_index[args.BS * (k + 1):]

        # stopped here
        tar1_predictions_logits = run_model(model, tar1_inputs, tar1_local_mask_index, ster_words_male)
        tar2_predictions_logits = run_model(model, tar2_inputs, tar2_local_mask_index, ster_words_female)

        jsd = jsd_model(tar1_predictions_logits, tar2_predictions_logits)
        jsd_np = jsd.detach().cpu().numpy()
        jsd_np = np.sum(jsd_np, axis=1)
        jsd_list += list(jsd_np)
        del tar1_predictions_logits, tar2_predictions_logits, jsd
    return jsd_list


def get_prompt_jsd(tar1_words, tar2_words, prompts, model, ster_words_male, ster_words_female, def_examples):
    jsd_word_list = []
    with open('prompt.txt','a') as f:
        assert len(tar1_words) == len(tar2_words)
        for (i, example) in enumerate(tqdm(def_examples)):
            tar1_tokenized_i, tar2_tokenized_i, tar1_mask_index_i, tar2_mask_index_i = get_tokenized_ith_prompt(prompts,
                                                                                                                example.text_a,
                                                                                                                example.text_b,
                                                                                                                tokenizer)
            print("tokenized input shape", tar1_tokenized_i['input_ids'].shape)
            jsd_list = get_JSD(tar1_tokenized_i, tar2_tokenized_i, tar1_mask_index_i, tar2_mask_index_i, model, ster_words_male, ster_words_female)
            jsd_word_list.append(jsd_list)
            print("got the jsd for the example_{}".format(i), example.text_a)
            f.write('\n')
            f.write(example.text_a)
            f.write('\n')
            f.write(example.text_b)
        jsd_word_list = np.array(jsd_word_list)
        print("jsd for every prompt, every word has shape", jsd_word_list.shape)
        f.close()
        return np.mean(jsd_word_list, axis=0)


def findpos(words, sent):
    sent_list = sent.split(' ')
    for i, token in enumerate(sent_list):
        if token in words:
            sent_list[i] = tokenizer.mask_token
            return token, ' '.join(sent_list)
    return None, None


def findpos_new(words, sent):
    sent_list = sent.split(' ')
    for i, token in enumerate(sent_list):
        if token in words:
           return token
    return None


def get_def_examples(def_pairs):
    '''Construct definitional examples from definitional pairs.'''
    def_examples = []
    for group_id in def_pairs:
        def_group = def_pairs[group_id]
        f_sents = def_group['f']
        m_sents = def_group['m']

        f_refer = set(FEMALES)
        m_refer = set(MALES)
        for sent_id, (sent_a, sent_b) in enumerate(zip(f_sents, m_sents)):
            word_a, sent_a_1 = findpos(f_refer, sent_a)
            word_b, sent_b_1 = findpos(m_refer, sent_b)
            if word_a!=None and word_b!=None:
                def_examples.append(InputExample(guid='{}-{}'.format(group_id, sent_id),
                                                 text_a=sent_a_1, text_b=sent_b_1, label=None, key_a=word_a, key_b=word_b))
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
        model = BertForMaskedLM.from_pretrained(args.model_name_or_path)
    elif args.model_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        model = RobertaForMaskedLM.from_pretrained(args.model_name_or_path)
    elif args.model_type == 'albert':
        tokenizer = AlbertTokenizer.from_pretrained(args.model_name_or_path)
        model = AlbertForMaskedLM.from_pretrained(args.model_name_or_path)
    else:
        raise NotImplementedError("not implemented!")

    device_ids = []
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.eval()
    model.cuda()

    jsd_model = JSD(reduction='none')

    if args.debias_type == 'gender':
        male_words = ["man","boy","he","father","son", "guy","male","his","himself","John"]
        female_words = ["woman","girl","she","mother","daughter","gal","female","her","herself","Mary"]

    elif args.debias_type == 'race':
        race1 = ["black", "african", "black", "africa", "africa", "africa", "black people", "african people",
                 "black people", "the africa"]
        race2 = ["caucasian", "caucasian", "white", "america", "america", "europe", "caucasian people",
                 "caucasian people", "white people", "the america"]

    ster_words_ = ["man","boy","he","father","son", "guy","male","his","himself","John","woman","girl","she","mother","daughter","gal","female","her","herself","Mary"]
    ster_words_male = male_words
    ster_words_female = female_words
    ster_words_ = tokenizer.convert_tokens_to_ids(ster_words_)  # stereotype words
    ster_words_male = tokenizer.convert_tokens_to_ids(ster_words_male)  # stereotype words
    ster_words_female = tokenizer.convert_tokens_to_ids(ster_words_female)  # stereotype words

    def_pairs = get_def_pairs(args.def_pairs_name, args.debias_type)
    def_examples = get_def_examples(def_pairs)  # 1D list where 2i and 2i+1 are a pair


    vocab = load_wiki_word_list(args.vocab_file)
    vocab = clean_word_list(vocab, tokenizer)  # vocabulary in prompts

    current_prompts = vocab

    f = open('data/prompts_{}_{}_{}'.format(args.model_name_or_path, args.debias_type, time.time()), 'w')
    f_100 = open('data/prompts_top200_{}'.format(time.time()), 'w')
    top_100_prompts = []
    for m in range(args.PL):
        if args.debias_type == 'gender':
            current_prompts_jsd = get_prompt_jsd(male_words, female_words, current_prompts, model, ster_words_male, ster_words_female, def_examples)
        elif args.debias_type == 'race':
            current_prompts_jsd = get_prompt_jsd(race1, race2, current_prompts, model, ster_words_male, ster_words_female)
        top_k_prompts = np.array(current_prompts)[np.argsort(current_prompts_jsd)[::-1][:args.K]]
        print(top_k_prompts)
        if m == 0:
            top_100_prompts = np.array(current_prompts)[np.argsort(current_prompts_jsd)[::-1][:200]]
            for k in top_100_prompts:
                f_100.write(k)
                f_100.write("\n")
            f_100.close()

        for p in top_k_prompts:
            f.write(p)
            f.write("\n")
        new_prompts = []
        for tkp in top_k_prompts:
            for v in top_100_prompts:
                new_prompts.append(tkp + " " + v)
        current_prompts = new_prompts
        print("search space size:", len(current_prompts))
    f.close()