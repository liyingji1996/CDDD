from __future__ import absolute_import, division, print_function

# standard library
import argparse
import csv
import logging
import os
import random
import sys
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange
from collections import OrderedDict
from def_sent_utils import get_def_pairs
from utils import *
import gc
from bert.mi_class import CLUB, FilterMU_Unite
from bert.modeling_bert import BertForSequenceClassificationMU
from bert.tokenization_bert import BertTokenizer
from albert.modeling_albert import AlbertForSequenceClassificationMU
from albert.tokenization_albert import AlbertTokenizer
from roberta.modeling_roberta import RobertaForSequenceClassificationMU, RobertaModelMU
from roberta.tokenization_roberta import RobertaTokenizer


logger = logging.getLogger(__name__)

FEMALES =["woman","girl","she","mother","daughter","gal","female","her","herself","Mary"]
MALES = ["man","boy","he","father","son", "guy","male","his","himself","John"]
RACE_2 = ["european", "british", "german", "polish", "russian", "europe", "italian", "portuguese", "french",
          "romanian", "greek", "irish", "spanish", "bosnian", "albanian", "caucasian", "caucasian", "caucasian",
          "caucasian", "caucasian", "caucasian", "caucasian", "caucasian","caucasian", "caucasian", "caucasian",
          "caucasian", "caucasian", "caucasian", "caucasian"]
RACE_1 = ["african", "african", "african", "african", "african", "african", "african", "african", "african", "african",
          "african", "african", "african", "african", "african","african", "nigerian", "ethiopian", "africa",
          "ghanaian", "kenyan", "egyptian", "somali", "liberian","moroccan", "cameroonian", "mexican",
          "eritrean", "sudanese","south-african"]


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


class DualInputFeatures(object):
    """A single set of dual features of data."""

    def __init__(self, input_ids_a, input_ids_b, mask_a, mask_b, segments_a, segments_b):
        self.input_ids_a = input_ids_a
        self.input_ids_b = input_ids_b
        self.mask_a = mask_a
        self.mask_b = mask_b
        self.segments_a = segments_a
        self.segments_b = segments_b


class BertEncoder(object):
    def __init__(self, model, device):
        self.device = device
        self.bert = model

    def encode(self, input_ids, token_type_ids=None, attention_mask=None, word_level=False, remove_bias=False):
        embeddings = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                               word_level=word_level, remove_bias=remove_bias, encode_only=True)
        return embeddings


class AlBertEncoder(object):
    def __init__(self, model, device):
        self.device = device
        self.albert = model

    def encode(self, input_ids, token_type_ids=None, attention_mask=None, word_level=False, remove_bias=False):
        self.albert.eval()
        embeddings = self.albert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                 word_level=False, remove_bias=remove_bias, encode_only=True)
        return embeddings


class RoBertaEncoder(object):
    def __init__(self, model, device):
        self.device = device
        self.roberta = model

    def encode(self, input_ids, token_type_ids=None, attention_mask=None, word_level=False, remove_bias=False):
        # self.roberta.eval()
        embeddings = self.roberta(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                 word_level=False, remove_bias=remove_bias, encode_only=True)
        return embeddings


def load_filter(model, filter_model):
    filter_state = filter_model.state_dict()
    model_state = model.state_dict()
    filter_state_ori = OrderedDict()
    for key in filter_state:
        filter_state_ori['filter_model.' + key] = filter_state[key]
    model_state.update(filter_state_ori)
    model.load_state_dict(model_state)
    return model,filter_state_ori


def train_eval_filter(args, filter_model, all_embeddings, gender_words, gender_embeddings, device):
    word_idx = list(range(len(gender_words)))
    word_refer = dict(zip(word_idx, gender_words))
    emb_a = all_embeddings[:len(all_embeddings)//2]
    emb_b = all_embeddings[len(all_embeddings)//2:]
    emb_pairs = np.concatenate([emb_a, emb_b], axis=1)

    word_idx_a = word_idx[:len(word_idx)//2]
    word_idx_b = word_idx[len(word_idx)//2:]

    data = TensorDataset(torch.from_numpy(emb_pairs), torch.tensor(word_idx_a), torch.tensor(word_idx_b))
    dataloader = DataLoader(data, batch_size=args.mu_size, shuffle=True)

    filter_model.train()
    optimizer = torch.optim.Adam(filter_model.module.parameters(), lr=args.mu_lr)

    current_step = 0
    for epoch in range(args.mu_epoch):
        for step, batch in enumerate(tqdm(dataloader)):
            current_step += 1
            sent_emb, worda, wordb = batch
            length = sent_emb.shape[1] // 2
            sent_emb_a, sent_emb_b = sent_emb[:,:length],sent_emb[:, length:]

            word_emb_a = torch.cat([torch.unsqueeze(gender_embeddings[word_refer[wd.item()].lower()], dim=0) for wd in worda], dim=0)
            word_emb_b = torch.cat([torch.unsqueeze(gender_embeddings[word_refer[wd.item()].lower()], dim=0) for wd in wordb], dim=0)

            if (device != None):
                sent_emb_a = sent_emb_a.cuda()
                word_emb_a = word_emb_a.cuda()
                sent_emb_b = sent_emb_b.cuda()
                word_emb_b = word_emb_b.cuda()

            loss = filter_model(sent_emb_a, word_emb_a, sent_emb_b, word_emb_b, args.mu_lam)
            loss = torch.sum(loss, dim=-1)
            print(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter_model.module.parameters(), 5)
            optimizer.step()
            optimizer.zero_grad()

        model_to_save = filter_model.module if hasattr(filter_model, 'module') else filter_model  # Only save the model it-self
        if args.add_prompt:
            # If we save using the predefined names, we can load using `from_pretrained`
            weights_name = "epoch{}_step{}_{}*{}_{}_0920_lr1-6".format(epoch, current_step, args.PL, args.k, args.task_name)
            output_filter_file = os.path.join(args.output_prompt_filter_dir, args.bias_type, weights_name)
            torch.save(model_to_save.state_dict(), output_filter_file)
        else:
            weights_name = "epoch{}_step_{}_debias_filter_{}".format(epoch, current_step, args.task_name)
            output_filter_file = os.path.join(args.output_filter_dir, weights_name)
            torch.save(model_to_save.state_dict(), output_filter_file)

def train_filter_old(args, filter_model, all_embeddings, gender_words, gender_embeddings, device):
    word_idx = list(range(len(gender_words)))
    word_refer = dict(zip(word_idx, gender_words))
    emb_a = all_embeddings[:len(all_embeddings)//2]
    emb_b = all_embeddings[len(all_embeddings)//2:]
    emb_pairs = np.concatenate([emb_a, emb_b], axis=1)

    word_idx_a = word_idx[:len(word_idx)//2]
    word_idx_b = word_idx[len(word_idx)//2:]

    data = TensorDataset(torch.from_numpy(emb_pairs), torch.tensor(word_idx_a), torch.tensor(word_idx_b))
    dataloader = DataLoader(data, batch_size=args.mu_size, shuffle=True)

    filter_model.train()
    optimizer = torch.optim.Adam(filter_model.module.parameters(), lr=args.mu_lr)
    if args.bert_model:
        club = CLUB(args.filter_input_size, args.mu_hidden_size)
    elif args.albert_model:
        club = CLUB(args.filter_input_size, args.word_size)
    elif args.roberta_model:
        club = CLUB(args.filter_input_size, args.mu_hidden_size)
    if (device!=None):
        club = club.cuda()
    club_optimizer = torch.optim.Adam(club.parameters(), lr=args.mu_lr)

    current_step = 0
    for epoch in range(args.mu_epoch):
        for step, batch in enumerate(tqdm(dataloader)):
            current_step += 1
            sent_emb, worda, wordb = batch
            length = sent_emb.shape[1] // 2
            sent_emb_a, sent_emb_b = sent_emb[:,:length],sent_emb[:, length:]

            word_emb_a = torch.cat([torch.unsqueeze(gender_embeddings[word_refer[wd.item()].lower()], dim=0) for wd in worda], dim=0)
            word_emb_b = torch.cat([torch.unsqueeze(gender_embeddings[word_refer[wd.item()].lower()], dim=0) for wd in wordb], dim=0)

            if (device != None):
                sent_emb_a = sent_emb_a.cuda()
                word_emb_a = word_emb_a.cuda()
                sent_emb_b = sent_emb_b.cuda()
                word_emb_b = word_emb_b.cuda()

            debiased_sent_emb_a = filter_model.module.encode(sent_emb_a)
            debiased_sent_emb_b = filter_model.module.encode(sent_emb_b)
            for j in range(5):
                club_loss = -club.loglikeli(debiased_sent_emb_a.detach(), word_emb_a.detach()) - club.loglikeli(debiased_sent_emb_b.detach(), word_emb_b.detach())
                club_loss.backward()
                club_optimizer.step()
                club_optimizer.zero_grad()

            loss = filter_model(sent_emb_a, word_emb_a, sent_emb_b, word_emb_b, args.mu_lam, club)
            loss = torch.sum(loss, dim=-1)
            print(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter_model.module.parameters(), 5)
            optimizer.step()
            optimizer.zero_grad()

        model_to_save = filter_model.module if hasattr(filter_model, 'module') else filter_model  # Only save the model it-self
        weights_name = "epoch{}_step{}_debias_filter_{}".format(epoch, current_step, args.task_name)
        output_filter_file = os.path.join(args.output_filter_dir, weights_name)
        torch.save(model_to_save.state_dict(), output_filter_file)


def extract_embeddings_pair(args, encoder, tokenizer, examples, max_seq_length, device,
        load, task, label_list, output_mode, norm, word_level=False):
    '''Encode paired examples into BERT embeddings in batches.
       Used in the computation of gender bias direction.
       Save computed embeddings under saved_embs/.
    '''
    if args.add_prompt:
        emb_loc_a = 'saved_embs_new/prompt_{}_{}*{}_num{}_a_{}.pkl'.format(args.bias_type, args.PL, args.k, len(examples), task)
        emb_loc_b = 'saved_embs_new/prompt_{}_{}*{}_num{}_b_{}.pkl'.format(args.bias_type, args.PL, args.k, len(examples), task)
    else:
        emb_loc_a = 'saved_embs/no-prompt_num{}_a_{}.pkl'.format(len(examples), task)
        emb_loc_b = 'saved_embs/no-prompt_num{}_b_{}.pkl'.format(len(examples), task)
    all_embeddings_a = []
    all_embeddings_b = []

    if os.path.isfile(emb_loc_a) and os.path.isfile(emb_loc_b) and load:
        with open(emb_loc_a, 'rb') as f:
            while 1:
                try:
                    all_embeddings_a.append(pickle.load(f))
                except EOFError:
                    break
        with open(emb_loc_b, 'rb') as f:
            while 1:
                try:
                    all_embeddings_b.append(pickle.load(f))
                except EOFError:
                    break
        print ('preprocessed embeddings loaded from:', emb_loc_a, emb_loc_b)

    else:
        if args.add_prompt:
            features_file = os.path.join("runs/pretrain_roberta", "pretrain_features_{}.pkl".format(args.bias_type))
        else:
            features_file = os.path.join("runs/pretrain_roberta", "pretrain_features_{}_no-prompt.pkl".format(args.bias_type))
        if not os.path.exists(features_file):
            features = convert_examples_to_dualfeatures(examples, label_list, max_seq_length, tokenizer, output_mode)
            print("--------------write features------------")
            all_inputs_a = torch.tensor([f.input_ids_a for f in features], dtype=torch.long)
            all_mask_a = torch.tensor([f.mask_a for f in features], dtype=torch.long)
            all_segments_a = torch.tensor([f.segments_a for f in features], dtype=torch.long)
            all_inputs_b = torch.tensor([f.input_ids_b for f in features], dtype=torch.long)
            all_mask_b = torch.tensor([f.mask_b for f in features], dtype=torch.long)
            all_segments_b = torch.tensor([f.segments_b for f in features], dtype=torch.long)
            with open(features_file, 'wb') as wf:
                pickle.dump(all_inputs_a, wf)
                pickle.dump(all_mask_a, wf)
                pickle.dump(all_segments_a, wf)
                pickle.dump(all_inputs_b, wf)
                pickle.dump(all_mask_b, wf)
                pickle.dump(all_segments_b, wf)
                wf.close()
        else:
            print("------------------read features---------------")
            with open(features_file, 'rb') as rf:
                all_inputs_a = pickle.load(rf)
                all_mask_a = pickle.load(rf)
                all_segments_a = pickle.load(rf)
                all_inputs_b = pickle.load(rf)
                all_mask_b = pickle.load(rf)
                all_segments_b = pickle.load(rf)
                rf.close()

        data = TensorDataset(all_inputs_a, all_inputs_b, all_mask_a, all_mask_b, all_segments_a, all_segments_b)
        del all_inputs_a, all_inputs_b, all_mask_a, all_mask_b, all_segments_a, all_segments_b  # 释放cpu内存
        gc.collect()
        dataloader = DataLoader(data, batch_size=1000, shuffle=False)
        print('preprocessed embeddings.')
        for step, batch in enumerate(tqdm(dataloader)):  # 进度条
            inputs_a, inputs_b, mask_a, mask_b, segments_a, segments_b = batch
            if (device != None):
                inputs_a = inputs_a.cuda()
                mask_a = mask_a.cuda()
                segments_a = segments_a.cuda()
                inputs_b = inputs_b.cuda()
                mask_b = mask_b.cuda()
                segments_b = segments_b.cuda()
            with torch.no_grad():
                embeddings_a = encoder.encode(input_ids=inputs_a, token_type_ids=segments_a, attention_mask=mask_a, word_level=False)
                embeddings_b = encoder.encode(input_ids=inputs_b, token_type_ids=segments_b, attention_mask=mask_b, word_level=False)

            embeddings_a = torch.mean(embeddings_a, dim=1)
            embeddings_b = torch.mean(embeddings_b, dim=1)
            embeddings_a /= torch.norm(embeddings_a, dim=-1, keepdim=True)
            embeddings_b /= torch.norm(embeddings_b, dim=-1, keepdim=True)

            if not torch.isnan(embeddings_a).any() and not torch.isnan(embeddings_b).any():
                embeddings_a = embeddings_a.cpu().detach().numpy()
                embeddings_b = embeddings_b.cpu().detach().numpy()
            if step == 0:
                with open(emb_loc_a, 'wb') as f_a:
                    pickle.dump(embeddings_a, f_a, protocol=4)
                    f_a.close()
                with open(emb_loc_b, 'wb') as f_b:
                    pickle.dump(embeddings_b, f_b, protocol=4)
                    f_b.close()
            else:
                with open(emb_loc_a, 'ab') as f_a:
                    pickle.dump(embeddings_a, f_a, protocol=4)
                    f_a.close()
                with open(emb_loc_b, 'ab') as f_b:
                    pickle.dump(embeddings_b, f_b, protocol=4)
                    f_b.close()
            print('preprocessed embeddings saved to:', emb_loc_a, emb_loc_b)
        if os.path.isfile(emb_loc_a) and os.path.isfile(emb_loc_b) and load:
            with open(emb_loc_a, 'rb') as f:
                while 1:
                    try:
                        all_embeddings_a.append(pickle.load(f))
                    except EOFError:
                        break
            with open(emb_loc_b, 'rb') as f:
                while 1:
                    try:
                        all_embeddings_b.append(pickle.load(f))
                    except EOFError:
                        break
            print('preprocessed embeddings loaded from:', emb_loc_a, emb_loc_b)
    all_embeddings_a = np.concatenate(all_embeddings_a, axis=0)
    all_embeddings_b = np.concatenate(all_embeddings_b, axis=0)
    all_embeddings = np.concatenate([all_embeddings_a, all_embeddings_b], axis=0)
    return all_embeddings


def bias_word_embeddings(args, encoder, tokenizer, examples, device, load, task):
    '''Encode paired examples into BERT embeddings in batches.
       Used in the computation of gender bias direction.
       Save computed embeddings under saved_embs/.
    '''
    if args.add_prompt:
        emb_loc = 'saved_word_embs_new/prompt_{}*{}_num{}_{}_{}.pkl'.format(args.PL, args.k, len(examples), args.bias_type, task)
    else:
        emb_loc = 'saved_word_embs/no-prompt_num{}_{}_{}.pkl'.format(len(examples), args.bias_type, task)

    if os.path.isfile(emb_loc) and load:
        with open(emb_loc, 'rb') as f:
            # a list of gender words and a dictionary
            words, words_embeddings = pickle.load(f)
            for k in words_embeddings:
                words_emb =torch.from_numpy(words_embeddings[k])
                if device!=None:
                    words_emb = words_emb.cuda()
                words_embeddings[k] = words_emb
        print('preprocessed words embeddings loaded from:', emb_loc)
    else:
        words = get_words_info(examples)
        if args.bias_type=='gender':
            all_words = [wd.lower() for wd in FEMALES + MALES]
            word_ids = tokenizer.convert_tokens_to_ids(all_words)
            word_ids_tensor = torch.Tensor(word_ids).long()
            if device != None:
                word_ids_tensor = word_ids_tensor.cuda()
            if args.bert_model:
                all_embeddings = encoder.bert.module.bert.embeddings.word_embeddings(word_ids_tensor) if hasattr(encoder.bert, 'module') else encoder.bert.bert.embeddings.word_embeddings
            elif args.albert_model:
                all_embeddings = encoder.albert.module.albert.embeddings.word_embeddings(word_ids_tensor) if hasattr(encoder.albert, 'module') else encoder.albert.albert.embeddings.word_embeddings
            elif args.roberta_model:
                all_embeddings = encoder.roberta.module.embeddings.word_embeddings(word_ids_tensor) if hasattr(encoder.roberta, 'module') else encoder.roberta.embeddings.word_embeddings
            embeddings_a = all_embeddings[:len(FEMALES)]
            embeddings_a /= torch.norm(embeddings_a, dim=-1, keepdim=True)
            embeddings_b = all_embeddings[len(FEMALES):]
            embeddings_b /= torch.norm(embeddings_b, dim=-1, keepdim=True)
        elif args.bias_type=='race':
            all_words = [wd.lower() for wd in RACE_2 + RACE_1]
            word_ids = tokenizer.convert_tokens_to_ids(all_words)
            word_ids_tensor = torch.Tensor(word_ids).long()
            if device != None:
                word_ids_tensor = word_ids_tensor.cuda()
            if args.bert_model:
                all_embeddings = encoder.bert.bert.embeddings.word_embeddings(word_ids_tensor)
            elif args.albert_model:
                all_embeddings = encoder.albert.albert.embeddings.word_embeddings(word_ids_tensor)
            elif args.roberta_model:
                all_embeddings = encoder.roberta.module.embeddings.word_embeddings(word_ids_tensor)
            embeddings_a = all_embeddings[:len(RACE_1)]
            embeddings_a /= torch.norm(embeddings_a, dim=-1, keepdim=True)
            embeddings_b = all_embeddings[len(RACE_1):]
            embeddings_b /= torch.norm(embeddings_b, dim=-1, keepdim=True)

        means = (embeddings_a + embeddings_b) / 2.0
        embeddings_a -= means
        embeddings_b -= means

        all_embeddings = torch.cat([embeddings_a,embeddings_b], axis=0)
        word_embeddings_tosave = all_embeddings.cpu().detach().numpy()
        words_embeddings = {}
        for i, keyword in enumerate(all_words):
            words_embeddings[keyword] = word_embeddings_tosave[i]

        with open(emb_loc, 'wb') as f:
            pickle.dump([words, words_embeddings], f, protocol=4)
        print('preprocessed words embeddings saved to:', emb_loc)
        # pdb.set_trace()
    return words, words_embeddings


def findpos(words, sent):
    sent_list = sent.split(' ')
    for i, token in enumerate(sent_list):
        if token in words:
            return token
    return None


def get_def_examples_prompt(args, def_pairs, prompts):
    '''Construct definitional examples from definitional pairs.'''
    def_examples = []
    for prompt_id in range(len(prompts)):
        for group_id in def_pairs:
            def_group = def_pairs[group_id]
            f_sents = def_group['f']
            m_sents = def_group['m']

            if args.bias_type == 'gender':
                f_refer = set(FEMALES)
                m_refer = set(MALES)
            elif args.bias_type == 'race':
                f_refer = set(RACE_2)
                m_refer = set(RACE_1)
            for sent_id, (sent_a, sent_b) in enumerate(zip(f_sents, m_sents)):
                word_a = findpos(f_refer, sent_a)
                word_b = findpos(m_refer, sent_b)
                assert word_a is not None
                assert word_b is not None
                sent_a = sent_a + " " + prompts[prompt_id] + "."
                sent_b = sent_b + " " + prompts[prompt_id] + "."
                def_examples.append(InputExample(guid='{}-{}-{}'.format(group_id, sent_id, prompt_id),
                                                 text_a=sent_a, text_b=sent_b, label=None, key_a=word_a, key_b=word_b))
    return def_examples


def get_def_examples(args, def_pairs):
    '''Construct definitional examples from definitional pairs.'''
    def_examples = []
    for group_id in def_pairs:
        def_group = def_pairs[group_id]
        f_sents = def_group['f']
        m_sents = def_group['m']

        if args.bias_type == 'gender':
            f_refer = set(FEMALES)
            m_refer = set(MALES)
        elif args.bias_type == 'race':
            f_refer = set(RACE_2)
            m_refer = set(RACE_1)
        for sent_id, (sent_a, sent_b) in enumerate(zip(f_sents, m_sents)):
            word_a = findpos(f_refer, sent_a)
            word_b = findpos(m_refer, sent_b)
            assert word_a is not None
            assert word_b is not None
            def_examples.append(InputExample(guid='{}-{}'.format(group_id, sent_id),
                                             text_a=sent_a, text_b=sent_b, label=None,key_a=word_a,key_b=word_b))
    return def_examples


def compute_gender_filter(args, device, tokenizer, bert_encoder, def_pairs, max_seq_length, load, task, word_level=False, keepdims=False):
    '''Compute gender debias autoencoder weights.'''
    if args.add_prompt:
        searched_prompts = load_word_list(args.data_path + args.prompts_file)
        def_examples = get_def_examples_prompt(args, def_pairs, searched_prompts)  # add prompt
    else:
        def_examples = get_def_examples(args, def_pairs)  # no prompt
    all_embeddings = extract_embeddings_pair(args, bert_encoder, tokenizer, def_examples, max_seq_length, device, load, task, label_list=None, output_mode=None, norm=True, word_level=word_level)
    gender_words, gender_embeddings = bias_word_embeddings(args, bert_encoder, tokenizer, def_examples, device,
                                                           load, task)
    filter_model = FilterMU_Unite(all_embeddings.shape[1], args.filter_encode_size, args.mu_hidden_size, args.mode)
    device_ids = [0]
    filter_model = torch.nn.DataParallel(filter_model, device_ids=device_ids)
    filter_model = filter_model.cuda()
    if args.add_prompt:
        train_eval_filter(args, filter_model, all_embeddings, gender_words, gender_embeddings, device)
    else:
        train_filter_old(args, filter_model, all_embeddings, gender_words, gender_embeddings, device)


def get_words_info(examples):
    word1 = []
    word2 = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        word1.append(example.key_a)
        if example.text_b:
            word2.append(example.key_b)

    word_info = word1 + word2
    return word_info


def convert_examples_to_dualfeatures(examples, label_list, max_seq_length, tokenizer, output_mode):
    """Loads a data file into a list of dual input features."""
    '''
    output_mode: classification or regression
    '''
    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)
        # truncate length
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
        tokens_a = [tokenizer.cls_token] + tokens_a + [tokenizer.sep_token]
        segments_a = [0] * len(tokens_a)
        input_ids_a = tokenizer.convert_tokens_to_ids(tokens_a)
        mask_a = [1] * len(input_ids_a)
        padding_a = [0] * (max_seq_length - len(input_ids_a))
        input_ids_a += padding_a
        mask_a += padding_a
        segments_a += padding_a
        assert(len(input_ids_a) == max_seq_length)
        assert(len(mask_a) == max_seq_length)
        assert(len(segments_a) == max_seq_length)


        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            if len(tokens_b) > max_seq_length - 2:
                tokens_b = tokens_b[:(max_seq_length - 2)]

            tokens_b = [tokenizer.cls_token] + tokens_b + [tokenizer.sep_token]
            segments_b = [0] * len(tokens_b)
            input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)
            mask_b = [1] * len(input_ids_b)
            padding_b = [0] * (max_seq_length - len(input_ids_b))
            input_ids_b += padding_b
            mask_b += padding_b
            segments_b += padding_b
            assert(len(input_ids_b) == max_seq_length)
            assert(len(mask_b) == max_seq_length)
            assert(len(segments_b) == max_seq_length)

        else:
            input_ids_b = None
            mask_b = None
            segments_b = None

        features.append(
                DualInputFeatures(input_ids_a=input_ids_a,
                                  input_ids_b=input_ids_b,
                                  mask_a=mask_a,
                                  mask_b=mask_b,
                                  segments_a=segments_a,
                                  segments_b=segments_b))
    return features


def parse_args():
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default="", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--albert_model", default="", type=str,
                        help="AlBert pre-trained model selected in the list: albert-base-v1," 
                        "albert-large-v1, albert-xlarge-v1, albert-xxlarge-v1, albert-base-v2, " 
                        "albert-large-v2, albert-xlarge-v2, albert-xxlarge-v2.")
    parser.add_argument("--roberta_model", default="roberta-base", type=str,
                        help="roberta-base")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        help="The name of the task to train.")
    # Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--resume_model_path",
                        type=str,
                        default="",
                        help="Whether to resume from a model.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--normalize",
                        action='store_true',
                        help="Set this flag if you want embeddings normalized.")
    parser.add_argument("--debias",
                        action='store_true',
                        help="Set this flag if you want embeddings debiased.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--def_pairs_name", default="all", type=str,
                        help="Name of definitional sentence pairs.")

    # filter parameters
    parser.add_argument("--output_filter_dir",
                        default="./saved_filter",
                        type=str,
                        help="The output directory where the debias fitler will be written.")
    parser.add_argument("--output_prompt_filter_dir",
                        default="./saved_filter_new",
                        type=str,
                        help="The output directory where the debias fitler will be written when you add prompt.")
    parser.add_argument("--mode",
                        default="club_infonce",
                        type=str,
                        help="Encoding way to debias.")
    parser.add_argument("--mu_lr", type=float, default=1e-5,
                        help="learning rate to train the filter")
    parser.add_argument("--mu_size", type=int, default=128,
                        help="batch size to train the filter")
    parser.add_argument("--mu_epoch", type=int, default=50,
                        help="batch size to train the filter")
    parser.add_argument("--mu_lam", type=float, default=0.2,
                        help="lambda to balance the recon loss and club loss")
    parser.add_argument("--filter_input_size", type=int, default=768,
                        help="dimension of the input size for filter")
    parser.add_argument("--filter_encode_size", type=int, default=768,
                        help="dimension of the output size for filter")
    parser.add_argument("--mu_hidden_size", type=int, default=768,
                        help="dimension of the hidden size of filter")
    parser.add_argument("--word_size", type=int, default=128,
                        help="dimension of the hidden size of filter")
    parser.add_argument("--sim_temp", type=float, default=1.0,
                        help="temperature for simclr loss")
    # prompt need
    parser.add_argument("--data_path", default="data/", type=str,
                        help="data path to put the target/attribute word list")
    parser.add_argument("--prompts_file", default="roberta/prompts_cos_roberta-base_gender_20_5", type=str,
                        help="the name of the file that stores the prompts, by default it is under the data_path")
    parser.add_argument("--bias_type", default='gender', type=str,
                        help="Choose from ['gender','race','religion']")
    parser.add_argument("--add_prompt", action='store_true',
                        help="Whether to add prompt augmentation.")
    parser.add_argument("--k", type=int, default=20,
                        help="batch size to train the filter")
    parser.add_argument("--PL", type=int, default=5,
                        help="batch size to train the filter")

    args = parser.parse_args()
    return args


def prepare_model_and_bias(args, device):
    '''Return model and gender direction (computed by resume_model_path)'''
    if args.bert_model:
        model_weights_path = args.bert_model if (args.resume_model_path == "") else args.resume_model_path
        tokenizer = BertTokenizer.from_pretrained(model_weights_path, do_lower_case=args.do_lower_case)
        # a. load pretrained model and compute gender direction
        logger.info("Initialize model with {}".format(model_weights_path))
        model = BertForSequenceClassificationMU.from_pretrained(model_weights_path)
        device_ids = []
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.cuda()
        if (args.debias):
            bert_encoder = BertEncoder(model, device)
            def_pairs = get_def_pairs(args.def_pairs_name, args.bias_type)
            compute_gender_filter(args, device, tokenizer, bert_encoder, def_pairs, args.max_seq_length, load=True, task=args.task_name)
    elif args.albert_model:
        model_weights_path = args.albert_model if (args.resume_model_path == "") else args.resume_model_path
        tokenizer = AlbertTokenizer.from_pretrained(model_weights_path, do_lower_case=args.do_lower_case)
        # a. load pretrained model and compute gender direction
        logger.info("Initialize model with {}".format(model_weights_path))
        model = AlbertForSequenceClassificationMU.from_pretrained(model_weights_path).cuda()
        if (args.debias):
            albert_encoder = AlBertEncoder(model, device)
            def_pairs = get_def_pairs(args.def_pairs_name, args.bias_type)
            compute_gender_filter(args, device, tokenizer, albert_encoder, def_pairs, args.max_seq_length,  load=True, task=args.task_name)
    elif args.roberta_model:
        model_weights_path = args.roberta_model if (args.resume_model_path == "") else args.resume_model_path
        tokenizer = RobertaTokenizer.from_pretrained(model_weights_path, do_lower_case=args.do_lower_case)
        # a. load pretrained model and compute gender direction
        logger.info("Initialize model with {}".format(model_weights_path))
        model = RobertaModelMU.from_pretrained(model_weights_path)
        device_ids = []
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.cuda()
        if (args.debias):
            roberta_encoder = RoBertaEncoder(model, device)
            def_pairs = get_def_pairs(args.def_pairs_name, args.bias_type)
            compute_gender_filter(args, device, tokenizer, roberta_encoder, def_pairs, args.max_seq_length, load=True, task=args.task_name)


def main():
    '''Fine-tune BERT on the specified task and evaluate on dev set.'''
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    n_gpu = torch.cuda.device_count()
    device = torch.device("cuda:2" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    prepare_model_and_bias(args, device)


if __name__ == "__main__":
    main()






