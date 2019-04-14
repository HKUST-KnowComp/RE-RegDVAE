import random

import numpy as np
import pickle
import copy

from definitions import settings
from definitions.OieExample import HoldoutDataset
from processing.OiePreprocessor  import FeatureLexiconCompact
from collections import Counter, OrderedDict, namedtuple


# Todo: the whole data is translated to structured array, currently not working.
def holdout_by_freq(freq_ranks):
    """
    holdout relations from origin_relation_lex by the freq_ranks list
    return: holdout relations, new relation lexicon
    """

    origin_relation_lex = relation_lex
    origin_en_lex = entity_lex

    relation_list = list(origin_relation_lex.id2Freq.items())

    holdout_relations = set()
    freq_ranks = np.array(freq_ranks)
    freq_ranks.sort()
    freq_ranks = freq_ranks[::-1]
    for i in freq_ranks:
        item = relation_list.pop(i)
        holdout_relations.add(item[0])

    trans_rllex = FeatureLexiconCompact()
    trans_rllex.init_by_freq_dict(dict(relation_list))
    return holdout_relations, trans_rllex


def holdout_randomly(k, subset):
    relation_list = set(relation_lex.id2Freq.keys())
    relation_list -= subset
    relation_list = list(relation_list)
    random.shuffle(relation_list)

    holdout_relations = subset
    holdout_relations = set.union(
        holdout_relations, set(relation_list[:k - len(subset)]))
    trans_rllex = FeatureLexiconCompact()
    for item in relation_list[k - len(subset):]:
        idx = trans_rllex.getOrAdd(item)
        trans_rllex.id2Freq[idx] = relation_lex.id2Freq[item]

    return holdout_relations, trans_rllex


def gen_holdout_dataset(holdout_relations, trans_rllex):
    """ split labeled_data by holdout_realtions
    Tr: relations that are not in holdout_relations, are used to train external kb
    translated by trans_enlex
    Tr2: all data that both arg1 and arg2 are in labeled data. translated by tran_enlex.
    Te: in holdout_relations, used to test oie model
    Te1: dev,
    Te2: test
    """
    trans_enlex = FeatureLexiconCompact()
    Tr = []
    Te = []
    for item in labeled_data:
        arg1 = trans_enlex.getOrAdd(item.arg1)
        arg2 = trans_enlex.getOrAdd(item.arg2)
        if item.relation in trans_rllex.str2ID:

            rl = trans_rllex.str2ID[item.relation]
            Tr.append([item.id, arg1, arg2, rl])
        else:
            Te.append(item)
    Tr2 = prune_sentences_by_entity(trans_enlex, oiedata.dataset.train)

    print("Tr: {} Te: {} Tr2: {} Entity in Tr:{}".format(
        len(Tr), len(Te), len(Tr2), trans_enlex.count))
    Te1, Te2 = split_list(Te, valid_ratio,)
    return HoldoutDataset(holdout_relations, trans_rllex, trans_enlex, Tr, Tr2, Te1, Te2)


def prune_sentences_by_entity(trans_enlex, oie_data):
    pruned = []
    id2Freq = Counter()
    for item in oie_data:
        if item.arg1 in trans_enlex.str2ID and item.arg2 in trans_enlex.str2ID:
            item = copy.deepcopy(item)
            item.arg1 = trans_enlex.str2ID[item.arg1]
            item.arg2 = trans_enlex.str2ID[item.arg2]
            if item.arg1 > 17376 or item.arg2 > 17376:
            pruned.append(item)
            id2Freq[item.arg1] += 1
            id2Freq[item.arg2] += 1
    trans_enlex.id2Freq = id2Freq
    return pruned


def save_data(data, output_dir):
    with open(output_dir, 'wb') as pklfile:
        pickle.dump(data, pklfile, protocol=settings.pkl_protocol)
