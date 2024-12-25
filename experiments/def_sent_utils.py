# standard library
from itertools import combinations
import numpy as np
import os, sys
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

np.random.seed(42)
words1 = [["african","european"], ["african","british"], ["african","german"], ["african","polish"],
          ["african","russian"], ["african","europe"], ["african","italian"], ["african","portuguese"],
          ["african","french"], ["african","romanian"], ["african","greek"], ["african","irish"],
          ["african","spanish"], ["african","bosnian"], ["african","albanian"],
          ["african","caucasian"], ["nigerian","caucasian"], ["ethiopian", "caucasian"], ["africa","caucasian"],
          ["ghanaian","caucasian"], ["kenyan","caucasian"], ["somali","caucasian"],
          ["liberian","caucasian"], ["moroccan","caucasian"], ["cameroonian","caucasian"], ["eritrean","caucasian"],
          ["egyptian","caucasian"], ["sudanese","caucasian"], ["mexican","caucasian"], ["south-african","caucasian"]]
words2 = [["woman", "man"], ["girl", "boy"], ["she", "he"], ["mother", "father"], ["daughter", "son"], ["gal", "guy"], ["female", "male"], ["her", "his"], ["herself", "himself"], ["Mary", "John"]]

DIRECTORY = '../text_corpus/'

GENDER = 0
RACE = 1


def match(a,L):
    for b in L:
        if a == b:
            return True
    return False


def replace(a, new, L):
    Lnew = []
    for b in L:
        if a == b:
            Lnew.append(new)
        else:
            Lnew.append(b)
    return ' '.join(Lnew)


def template1(words, sent, sent_list, all_pairs):
    for i, (race1, race2) in enumerate(words):
        if match(race1, sent_list):
            sent_r1 = sent
            sent_r2 = replace(race1, race2, sent_list)
            all_pairs[i]['m'].append(sent_r1)
            all_pairs[i]['f'].append(sent_r2)
        if match(race2, sent_list):
            sent_r1 = replace(race2, race1, sent_list)
            sent_r2 = sent
            all_pairs[i]['m'].append(sent_r1)
            all_pairs[i]['f'].append(sent_r2)
    return all_pairs


def template2(words, sent, sent_list, all_pairs):
    for i, (female, male) in enumerate(words):
        if match(female, sent_list):
            sent_f = sent
            sent_m = replace(female, male, sent_list)
            all_pairs[i]['f'].append(sent_f)
            all_pairs[i]['m'].append(sent_m)
        if match(male, sent_list):
            sent_f = replace(male, female, sent_list)
            sent_m = sent
            all_pairs[i]['f'].append(sent_f)
            all_pairs[i]['m'].append(sent_m)
    return all_pairs


def get_pom(bais_type):
    all_pairs1 = defaultdict(lambda: defaultdict(list))
    all_pairs2 = defaultdict(lambda: defaultdict(list))
    pom_loc = os.path.join(DIRECTORY, 'POM/')

    for file in os.listdir(pom_loc):
        if file.endswith(".txt"):
            f = open(os.path.join(pom_loc, file), 'r')
            data = f.read()
            for sent in data.lower().split('.'):
                sent = sent.strip()
                sent_list = sent.split(' ')
                if bais_type == 'race':
                    all_pairs = template1(words1, sent, sent_list, all_pairs1)
                elif bais_type == 'gender':
                    all_pairs = template2(words2, sent, sent_list, all_pairs2)
    return all_pairs


def get_rest(filename, bais_type):
    all_pairs1 = defaultdict(lambda: defaultdict(list))
    all_pairs2 = defaultdict(lambda: defaultdict(list))

    f = open(os.path.join(DIRECTORY, filename), 'r')
    data = f.read()
    for sent in data.lower().split('\n'):
        sent = sent.strip()
        sent_list = sent.split(' ')
        if bais_type == 'race':
            all_pairs = template1(words1, sent, sent_list, all_pairs1)
        elif bais_type == 'gender':
            all_pairs = template2(words2, sent, sent_list, all_pairs2)

    print(filename, len(all_pairs))  # print: reddit.txt 9
    return all_pairs


def get_sst(bais_type):
    all_pairs1 = defaultdict(lambda: defaultdict(list))
    all_pairs2 = defaultdict(lambda: defaultdict(list))

    for sent in open(os.path.join(DIRECTORY, 'sst.txt'), 'r'):
        try:
            num = int(sent.split('\t')[0])
            sent = sent.split('\t')[1:]
            sent = ' '.join(sent)
        except:
            pass
        sent = sent.lower().strip()
        sent_list = sent.split(' ')
        if bais_type == 'race':
            all_pairs = template1(words1, sent, sent_list, all_pairs1)
        elif bais_type == 'gender':
            all_pairs = template2(words2, sent, sent_list, all_pairs2)
    return all_pairs


def check_bucket_size(D):
    n = 0
    for i in D:
        for key in D[i]:
            n += len(D[i][key])
            break
    return n


# domain: news, reddit, sst, pom, wikitext
def get_single_domain(domain, bais_type):
    if (domain == "pom"):
        def_pairs = get_pom(bais_type)
    elif (domain == "sst"):
        def_pairs = get_sst(bais_type)
    else:
        def_pairs = get_rest("{}.txt".format(domain), bais_type)
    return def_pairs

def get_all(bais_type):
    domains = ["reddit", "sst", "wikitext", "pom", "meld"]
    print("Get data from {}".format(domains))
    all_data = defaultdict(lambda: defaultdict(list))
    for domain in domains:
        bucket = get_single_domain(domain, bais_type)
        bucket_size = check_bucket_size(bucket)
        print("{} has {} pairs of templates".format(domain, bucket_size))
        for i in bucket:
            for term in bucket[i]:
                all_data[i][term].extend(bucket[i][term])
    total_size = check_bucket_size(all_data)
    print("{} pairs of templates in total".format(total_size))
    return all_data

def get_def_pairs(def_pairs_name, bais_type):
    # all 5 sets
    if (def_pairs_name == "all"):
        return get_all(bais_type)
    else:
        raise Exception("invalid defining pairs name")

