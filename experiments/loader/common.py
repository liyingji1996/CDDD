import json, os, csv, sys, logging
import torch
from torch.utils.data.dataloader import default_collate
import numpy as np
import pickle
import traceback
logger = logging.getLogger(__name__)

class DotDict:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)

def dotdict_collate(batch):
    return DotDict(**default_collate(batch))


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label, group_male, group_female, group_white, group_black):
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
        self.guid = guid  # unused
        self.text = text
        self.label = label
        self.group_male = group_male
        self.group_female = group_female
        self.group_white = group_white
        self.group_black = group_black


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, male_id, female_id, white_id, black_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.male_id = male_id
        self.female_id = female_id
        self.white_id = white_id
        self.black_id = black_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


def convert_examples_to_features(examples, max_seq_length, tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        try:
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            tokens = tokenizer.tokenize(example.text)
            if len(tokens) > max_seq_length - 2:
                tokens = tokens[:(max_seq_length - 2)]
            tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            segment_ids = [0] * len(input_ids)
            input_mask = [1] * len(input_ids)
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            if output_mode == "classification":
                label_id = example.label
                male_id = example.group_male
                female_id = example.group_female
                white_id = example.group_white
                black_id = example.group_black
            elif output_mode == "regression":
                label_id = float(example.label)
            else:
                raise KeyError(output_mode)
            features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                                          label_id=label_id, male_id=male_id, female_id=female_id, white_id=white_id, black_id=black_id))

        except TypeError:
            traceback.print_exc()
            continue

    return features


def write_features(fs, file):
    print("--------------write features------------")
    input_ids = np.array([f.input_ids for f in fs])
    input_mask = np.array([f.input_mask for f in fs])
    segment_ids = np.array([f.segment_ids for f in fs])
    label_ids = np.array([f.label_id for f in fs])
    male_ids = np.array([f.male_id for f in fs])
    female_ids = np.array([f.female_id for f in fs])
    white_ids = np.array([f.white_id for f in fs])
    black_ids = np.array([f.black_id for f in fs])
    with open(file, 'wb') as wf:
        pickle.dump(input_ids, wf)
        pickle.dump(input_mask, wf)
        pickle.dump(segment_ids, wf)
        pickle.dump(label_ids, wf)
        pickle.dump(male_ids, wf)
        pickle.dump(female_ids, wf)
        pickle.dump(white_ids, wf)
        pickle.dump(black_ids, wf)


def unpack_features(file, output_mode='classification'):
    print("------------------read features---------------")
    with open(file, 'rb') as rf:
        input_ids = pickle.load(rf)
        input_mask = pickle.load(rf)
        segment_ids = pickle.load(rf)
        label_ids = pickle.load(rf)
        male_ids = pickle.load(rf)
        female_ids = pickle.load(rf)
        white_ids = pickle.load(rf)
        black_ids = pickle.load(rf)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.long)
    segment_ids = torch.tensor(segment_ids, dtype=torch.long)

    if output_mode == 'regression':
        label_ids = torch.tensor(label_ids, dtype=torch.float)
    else:
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        male_ids = torch.tensor(male_ids, dtype=torch.long)
        female_ids = torch.tensor(female_ids, dtype=torch.long)
        white_ids = torch.tensor(white_ids, dtype=torch.long)
        black_ids = torch.tensor(black_ids, dtype=torch.long)
    return input_ids, input_mask, segment_ids, label_ids, male_ids, female_ids, white_ids, black_ids