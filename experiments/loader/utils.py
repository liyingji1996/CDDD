from collections import OrderedDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from .common import convert_examples_to_features
from collections import Counter
import re
import os
import torch
import json
import yaml
import numpy as np
import string
import pandas as pd
import traceback

CONFIG_NAME = "config.json"     # TODO: do multiple config to separate model from framework
WEIGHTS_NAME = "pytorch_model.bin"
PHASE_NAMES = ['normal', 'correcting', 'debiasing']
MAX_LINE_WIDTH = 150


def load_text_as_feature(args, processor, tokenizer, dataset, output_mode='classification'):
    valid_choices = ['train', 'test', 'eval']
    assert dataset in valid_choices, 'Invalid dataset is given: [{}], valid choices {}'.format(dataset, valid_choices)
    if dataset == 'train':
        examples = processor.get_train_examples(args.data_dir)
    elif dataset == 'eval':
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)
    features = convert_examples_to_features(examples, args.max_seq_length, tokenizer, output_mode)
    return features, examples


def compute_metrics(labels, pred_labels):
    assert len(pred_labels) == len(labels), \
        'Unmatched length between predictions [{}] and ground truth [{}]'.format(len(pred_labels), len(labels))
    return accuracy_score(labels, pred_labels)


def save_model_new(args, model, epoch):
    target_dir = args.output_dir
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # model.save_pretrained(target_dir)
    torch.save(model.state_dict(), os.path.join(target_dir, 'ck.pth'))
    # torch.save(model_to_save.state_dict(), output_model_file)
    model.config.to_json_file(os.path.join(target_dir, 'epoch_{}_config.json'.format(epoch)))
    # tokenizer.save_vocabulary(target_dir)
    f = open(os.path.join(target_dir, 'args.json'), 'w')
    json.dump(args.__dict__, f, indent=4)
    f.close()


def seconds2hms(s, get_int=True):
    h = s//3600
    m = (s % 3600) // 60
    s = s % 60
    if get_int:
        return int(h), int(m), int(s)
    return h, m, s


class DescStr:
    def __init__(self):
        self._desc = ''

    def write(self, instr):
        self._desc += re.sub('\n|\x1b.*|\r', '', instr)

    def read(self):
        ret = self._desc
        self._desc = ''
        return ret

    def flush(self):
        pass
