"""
    Preprocessor,
"""
import json
import argparse
import logging
import logging.config
import os
import pickle
import sys
from collections import Counter
from typing import List, Callable, AnyStr
from multiprocessing import Pool

import numpy as np

from definitions import OieExample
from definitions import OieFeatures
from definitions import settings, tools
from definitions.OieExample import DataSet
from definitions import log_dict


class FeatureLexiconCompact:
    def __init__(self):
        # Todo: use sklearnsklearn.preprocessing.LabelEncoder
        self.count = 0
        self.id2Str = {}
        self.str2ID = {}
        self.id2Freq = Counter()

    def init_by_freq_dict(self, freq):
        self.count = len(freq)
        self.id2Str = list(freq.keys())
        self.str2ID = dict(zip(freq.keys(), np.arange(len(freq))))
        self.id2Freq = dict(zip(np.arange(len(freq)), freq.values()))

    @classmethod
    def init_by_list(cls, lst):
        freq = Counter(lst)
        obj = cls()
        obj.init_by_freq_dict(freq)
        return obj

    def getOrAdd(self, s):
        if s is None:
            return None
        if s not in self.str2ID:
            self.id2Str[self.count] = s
            self.str2ID[s] = self.count
            self.count += 1
        idx = self.str2ID[s]
        self.id2Freq[idx] += 1
        return idx

    def __getitem__(self, key):
        # Todo: ambiguous, to be removed
        if isinstance(key, int):
            return self.getStr(key)
        else:
            return self.getId(key)

    def getId(self, s):
        """ get id by string, return None if not exist
        """
        return self.str2ID.get(s)

    def getStr(self, idx):
        return self.id2Str[idx]

    def getFreq(self, idx):
        return self.id2Freq.get(idx, 0)

    def get_dimensionality(self):
        return self.count

    def neg_sample_cum(self, neg_power):
        sorted_freqs = sorted(self.id2Freq.items())
        arg_samp_freqs = np.array([i for _, i in sorted_freqs])
        arg_samp_freqs_powered = np.power(arg_samp_freqs, neg_power)
        negSamplingDistr = tools.normalize(arg_samp_freqs_powered, ord=1)
        negSamplingCum = np.cumsum(negSamplingDistr)

        return negSamplingDistr, negSamplingCum

    def __len__(self):
        return len(self.str2ID)

    def save_txt(self, output_path, overwrite=False):
        if os.path.exists(output_path):
            logger = logging.getLogger(__name__)
            logger.warning("File exists {}".format(output_path))
            if not overwrite:
                return

        with open(output_path, 'w') as f:
            for s, i in self.str2ID.items():
                freq = self.id2Freq[i]
                f.write("{}\t{}\t{}\n".format(s, i, freq))

    @classmethod
    def load_txt(cls, input_path):
        obj = cls()
        with open(input_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            s, i, f = line.strip().split('\t')
            i = int(i)
            obj.str2ID[s] = int(i)
            obj.id2Str[i] = s
            obj.id2Freq[i] = int(f)
        obj.count = len(obj)

        return obj


class LoadExamplesIter:
    def __init__(self, file_name):
        self.count_line = 0
        self.fin = open(file_name, 'r')

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self.fin)
        # can't strip before split, else it will remove last \t
        line = line.split('\t')
        if len(line) == 0:
            raise IOError
        assert len(line) == 9, "a problem with the file format (# fields is wrong) len is " \
                               + str(len(line)) + " instead of 9:" + str(line)
        self.count_line += 1
        return [str(self.count_line - 1)] + line


def write_stats(data, p):
    logger = logging.getLogger(__name__)
    p = os.path.join(p, 'stats.json')
    if os.path.exists(p):
        with open(p, 'r') as f:
            stats = json.load(f)
    else:
        stats = {}
    stats.update(data)
    with open(p, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info('Write stats {} to {}'.format(str(stats), p))


def getFeatures(feat_freq, featureExs: List[Callable], info, arg1=None, arg2=None,
                expand: bool = False, threshold: int = -1):
    """
    :param featureExs:       list of feature extractors, all should return list
    :param info:             features
    :param arg1:
    :param arg2:
    :param expand:           condition that lexicon always contains the feature
    :param threshold:        threshold, -1 for no threshold
    :return:                 list of features
    """
    feats = []
    for f in featureExs:
        res = f(info, arg1, arg2)
        for el in res:
            feat_str = f.__name__ + "#" + el
            feat_freq[feat_str] += 1
            feats.append(feat_str)
    return feats


def prepareArgParser():
    return parser


@tools.print_func_name
def loadExamples(fileName: str) -> List[AnyStr]:
    """
    load examples
    :param fileName:
    :return:                 [token, feature1,..., feature9], all str
    """
    count = 0
    if __debug__:
        print(fileName)
    with open(fileName, 'r', encoding='utf-8') as fp:
        relationExamples = []
        for line in fp:
            # line = line.strip()
            if len(line) == 0 or len(line.split()) == 0:
                raise IOError

            else:
                fields = line.split('\t')
                assert len(fields) == 9, "a problem with the file format (# fields is wrong) len is " \
                                         + str(len(fields)) + "instead of 9" \
                                         + line
                # this will be 10
                relationExamples.append([str(count)] + fields)
                count += 1

    return relationExamples


def load_self_extracted_examples(file_lines):
    relationExamples = []
    for count, line in enumerate(file_lines):
        line = line.split('\t')
        if len(line) <= 1:
            relationExamples.append([str(count), ] + [None, ] * 8)
        elif len(line) == 8:
            line[-1] = line[-1].strip()
            fields = line
            # this will be 10
            relationExamples.append([str(count)] + fields + [None])
        else:
            relationExamples.append([str(count)] + fields)
    return relationExamples


def load_examples_to_np(file_name: str):
    logger = logging.getLogger(__name__)
    logger.debug(file_name)
    with open(file_name, 'r', encoding='utf-8') as fp:
        relationExamples = fp.readlines()
    relationExamples = [(str(i),) + tuple(item.split('\t')) for i, item in enumerate(relationExamples)]

    dtype = OieExample.RawDataExampleDType
    relationExamples = np.array(relationExamples, dtype=dtype)
    return relationExamples


def get_sentence_id_to_index(index2sid):
    index2sid = np.array(index2sid, dtype=int)
    sid2indx = np.full(np.max(index2sid) + 1, -1, dtype=int)
    for index, sid in enumerate(index2sid):
        sid2indx[sid] = index
    return sid2indx


@tools.print_func_name
def make_dataexampledtype_structure_array(data_list):
    list_of_tuple = [tuple(x) for x in data_list]
    structure = np.array(list_of_tuple, dtype=OieExample.DataExampleDType)
    return structure


def get_datalines(p):
    data_lens = np.load(os.path.join(p, 'dataset.npz'))
    data_lens = {k: len(v) for k, v in data_lens.items()}
    return data_lens


def get_split_feature_lines(features, lens):
    tr = 0
    va = lens['train']
    te = lens['valid'] + va
    return {'train': features[tr:va], 'valid': features[va:te], 'test': features[te:]}


def process_features_to_npz(args):
    if os.path.exists(os.path.join(args.output_dir, 'entity_lexicon.txt')):
        logger.info('found existing entity&relation lexicons')
        entity_lexicon = FeatureLexiconCompact.load_txt(os.path.join(args.output_dir, 'entity_lexicon.txt'))
        relation_lexicon = FeatureLexiconCompact.load_txt(os.path.join(args.output_dir, 'relation_lexicon.txt'))
    else:
        entity_lexicon = FeatureLexiconCompact()
        relation_lexicon = FeatureLexiconCompact()

    feature_extrs = OieFeatures.getBasicCleanFeatures()
    feature_lexicon = FeatureLexiconCompact()
    with open(os.path.join(args.output_dir, 'features.txt'), 'r', encoding='utf-8') as f:
        featurelines = f.readlines()
    data_lens = get_datalines(args.output_dir)
    dataset = get_split_feature_lines(featurelines, data_lens)
    dataset = {k: load_self_extracted_examples(v) for k, v in dataset.items()}
    raw_examples, feature_lexicon, entity_lexicon, relation_lexicon = get_feature_2_freq(dataset['train'],
                                                                                         feature_lexicon,
                                                                                         entity_lexicon,
                                                                                         relation_lexicon,
                                                                                         args.threshold)
    dataids = {}
    for batch_name in ['train', 'valid', 'test']:
        relationExamples = dataset[batch_name]
        if batch_name == 'train':
            data = encode_feature_to_id(raw_examples, feature_lexicon, num_workers=args.num_workers)
        else:
            data = encode_data_to_id(relationExamples, feature_extrs, feature_lexicon, entity_lexicon, relation_lexicon, num_workers=args.num_workers)
        dataids[batch_name] = [x[3] if x[3] is not None else [] for x in data]
        print("{} Processed {:d} examples".format(batch_name, len(data)))
    npz_dir = os.path.join(args.output_dir, 'features_{}.npz'.format(args.threshold))
    logger.info("Output feature to {}".format(npz_dir))
    np.savez(npz_dir, **dataids)
    write_stats({
        'feature_{}_num'.format(args.threshold): len(feature_lexicon)},
        args.output_dir)
    feature_lexicon.save_txt(os.path.join(args.output_dir, 'feature_{}_lexicon.txt'.format(args.threshold)))


# @tools.profile
def process(args):
    logger = logging.getLogger(__name__)
    logger.debug("Loading sentences...")

    if os.path.exists(args.pickled_dataset):
        logger.error("Found existing pickled dataset")
        return

    else:
        logger.info("Using rich features")
        feature_extrs = OieFeatures.getBasicCleanFeatures()

        if os.path.exists(os.path.join(args.output_dir, 'entity_lexicon.txt')):
            logger.info('found existing entity&relation lexicons')
            entity_lexicon = FeatureLexiconCompact.load_txt(os.path.join(args.output_dir, 'entity_lexicon.txt'))
            relation_lexicon = FeatureLexiconCompact.load_txt(os.path.join(args.output_dir, 'relation_lexicon.txt'))
        else:
            entity_lexicon = FeatureLexiconCompact()
            relation_lexicon = FeatureLexiconCompact()

        feature_lexicon = FeatureLexiconCompact()
        processed_dataset = DataSet(None, None, None)

    logger.info("Processing relation Examples")

    # ---------- Train --------------------------------------------------------
    # relationExamples = loadExamples(getattr(args, 'train'))
    dataset = {key: loadExamples(getattr(args, key)) for key in ['train', 'valid', 'test']}

    raw_examples, feature_lexicon, entity_lexicon, relation_lexicon = get_feature_2_freq(dataset['train'],
                                                                                         feature_lexicon,
                                                                                         entity_lexicon,
                                                                                         relation_lexicon,
                                                                                         args.threshold)
    # train_ids = encode_feature_to_id(raw_examples, feature_lexicon)
    # dataids = {}
    for batch_name in ['train', 'valid', 'test']:
        relationExamples = dataset[batch_name]
        data = encode_data_to_id(relationExamples, feature_extrs, feature_lexicon, entity_lexicon, relation_lexicon)
        # dataids[batch_name] = data
        setattr(processed_dataset, batch_name, make_dataexampledtype_structure_array(data))
        print("{} Processed {:d} examples".format(batch_name, len(data)))
        # get_feature_2_freq(relationExamples, feature_lexicon, entity_lexicon, relation_lexicon)

    if args.output_pkl:
        logger.info("Pickling the dataset...")
        pkl_file = open(args.pickled_dataset, 'wb')

        pklProtocol = settings.pkl_protocol
        data = OieExample.AllData(feature_extrs=feature_extrs,
                                  feature_lexicon=feature_lexicon,
                                  entity_lexicon=entity_lexicon,
                                  relation_lexicon=relation_lexicon,
                                  dataset=processed_dataset,
                                  )
        pickle.dump(data, pkl_file, pklProtocol)
    else:
        npz_dir = os.path.join(args.output_dir, 'features_{}.npz'.format(args.threshold))
        logger.info("Output feature to {}".format(npz_dir))
        np.savez(npz_dir,
                 train=processed_dataset.train['feats'],
                 valid=processed_dataset.valid['feats'],
                 test=processed_dataset.test['feats'])
        write_stats({
            'feature_{}_num'.format(args.threshold): len(feature_lexicon)
        },
            args.output_dir
        )
        feature_lexicon.save_txt(os.path.join(args.output_dir, 'feature_{}_lexicon.txt'.format(args.threshold)))
        entity_lexicon.save_txt(os.path.join(args.output_dir, 'entity_lexicon.txt'))
        relation_lexicon.save_txt(os.path.join(args.output_dir, 'relation_lexicon.txt'))


def get_feature_2_freq(train, feature_lexicon=None, entity_lexicon=None, relation_lexicon=None, threshold=-1,
                       ):
    entity_lexicon = FeatureLexiconCompact() if entity_lexicon is None else entity_lexicon
    relation_lexicon = FeatureLexiconCompact() if relation_lexicon is None else relation_lexicon
    feature_lexicon = FeatureLexiconCompact() if feature_lexicon is None else feature_lexicon
    logger = logging.getLogger(__name__)

    # ---------- Count& get examples -----------------------------
    raw_examples = []
    feat2freq = Counter()

    relationExamples = train
    feature_extrs = OieFeatures.getBasicCleanFeatures()
    empty_line_count = 0
    for re_idx, re in enumerate(relationExamples):
        if not re or re[1] is None:
            raw_examples.append((None,) * 6)
            empty_line_count += 1
            continue
        feats = []
        arg1, arg2 = entity_lexicon.getOrAdd(re[2]), entity_lexicon.getOrAdd(re[3])
        relation_label = re[-1]
        if relation_label is not None:
            relation_label.strip().split(' ')[0]
        else:
            relation_label = ''
        if relation_label != '':
            try:
                relation_label_id = relation_lexicon.getOrAdd(relation_label)  # no empty str in relation lexicon
            except:
                import pdb;
                pdb.set_trace()
        else:
            relation_label_id = -1
        re = [str(x).replace('_', ' ') for x in re]
        for f in feature_extrs:

            try:
                res = f([re[1], re[4], re[5], re[7], re[8], re[6]],
                        re[2], re[3])
            except:
                import pdb;
                pdb.set_trace()
            # entity_lexicon.getOrAdd(re[2])
            for el in res:
                feat_str = f.__name__ + "#" + el
                feat2freq[feat_str] += 1
                feats.append(feat_str)
        raw_examples.append([re_idx, arg1, arg2] + [feats] + [relation_label_id, re[5]])

    # -------- threshold -------------------------------
    feat2freq = {k: v for k, v in feat2freq.items() if v > threshold}
    feature_lexicon.init_by_freq_dict(feat2freq)
    logger.info("empty line: %d" % empty_line_count)
    return raw_examples, feature_lexicon, entity_lexicon, relation_lexicon

def _encode_feature_to_id(raw_examples, feature_lexicon: FeatureLexiconCompact):
    str2ID = feature_lexicon.str2ID
    examples = []
    for reIdx, re in enumerate(raw_examples):
        if not re or re[1] is None:
            examples.append((None,) * 7)
            continue
        pure_feats = []
        for feat in re[3]:
            if feat in str2ID:
                pure_feats.append(feature_lexicon.getId(feat))
        ex = OieExample.DataExample(re[0], re[1], re[2], pure_feats,
                                    re[4], None, re[5]
                                    )
        examples.append(ex)
    return examples

    setattr(dataset, "train", make_dataexampledtype_structure_array(examples))

    print("Processed {:d} training examples".format(len(examples)))

    # ---------- Valid & Test -----------------------------
    for batch_name in ('valid', 'test'):
        relationExamples = loadExamples(getattr(args, batch_name))
        pass

def encode_feature_to_id(raw_examples, feature_lexicon: FeatureLexiconCompact, num_workers=4):
    if num_workers == 1:
        return _encode_feature_to_id(raw_examples, feature_lexicon)
    else:
        examples = []
        total = len(raw_examples)
        num_per_worker = total // num_workers + 1
        pool = Pool()
        results = []
        for i in range(num_workers):
            results.append(pool.apply_async(_encode_feature_to_id, args=(raw_examples[i*num_per_worker:min((i+1)*num_per_worker, total)], feature_lexicon)))
        pool.close()
        pool.join()
        for i in range(num_workers):
            examples.extend(results[i].get())
        return examples

def _encode_data_to_id(relationExamples, feature_extrs, feature_lexicon=None, entity_lexicon=None, relation_lexicon=None,
                      test_mode=False):
    examples = []
    for re_idx, re in enumerate(relationExamples):
        if re_idx % 1000 == 0:
            print(".", end="")
        if re_idx % 10000 == 0:
            print(re_idx, end="")

        feats = []
        if not re or re[1] is None:
            examples.append((None,) * 7)
            continue
        relation_label = re[-1]
        if relation_label is not None:
            relation_label.strip().split(' ')[0]
        else:
            relation_label = ''
        if test_mode and not relation_label:
            pdb.set_trace()
            examples.append((None,) * 7)
            continue
        relation_label_id = relation_lexicon[relation_label]
        arg1, arg2 = entity_lexicon[re[2]], entity_lexicon[re[3]]
        if arg1 is None or arg2 is None:
            # print(re_idx, re, "contains external arguments", arg1, arg2, end=" ")
            examples.append((None,) * 7)
            continue

        re = [str(x).replace('_', ' ') for x in re]
        for f in feature_extrs:

            res = f([re[1], re[4], re[5], re[7], re[8], re[6]],
                    re[2], re[3])

            for el in res:
                feat_str = f.__name__ + "#" + el
                if feat_str in feature_lexicon:
                    feats.append(feature_lexicon[feat_str])
        # ex = OieExample.DataExample(re_idx, arg1, arg2, feats,
        ex = (re_idx, arg1, arg2, feats,
              relation_label_id, None, re[5]
              )
        examples.append(ex)
    return examples

def encode_data_to_id(relationExamples, feature_extrs, feature_lexicon=None, entity_lexicon=None, relation_lexicon=None,
                      test_mode=False, num_workers=4):
    if num_workers == 1:
        return _encode_data_to_id(relationExamples, feature_extrs, feature_lexicon, entity_lexicon, relation_lexicon, test_mode)
    else:
        examples = []
        total = len(relationExamples)
        num_per_worker = total // num_workers + 1
        pool = Pool()
        results = []
        for i in range(num_workers):
            results.append(pool.apply_async(_encode_data_to_id, args=(relationExamples[i*num_per_worker:min((i+1)*num_per_worker, total)], feature_extrs, feature_lexicon, entity_lexicon, relation_lexicon, test_mode)))
        pool.close()
        pool.join()
        for i in range(num_workers):
            examples.extend(results[i].get())
        return examples


if __name__ == '__main__':
    print("Parameters: " + str(sys.argv[1::]))
    parser = argparse.ArgumentParser(description='Processes an Oie file and add its representations '
                                                 'to a Python pickled file.')

    parser.add_argument('--train', help='Train input file in the Yao format')
    parser.add_argument('--valid', help='Valid input file in the Yao format')
    parser.add_argument('--test', help='Test input file in the Yao format')

    parser.add_argument('--pickled_dataset', default='', help='pickle file to be used to store output '
                                                              '(created if empty)')
    parser.add_argument('--input_dir', default=None)
    parser.add_argument('--output_dir', default='')
    parser.add_argument('--features', default="basic", nargs="?", help='features (basic vs ?)')
    parser.add_argument('--threshold', default="5", nargs="?", type=int, help='minimum feature frequency')

    parser.add_argument('--test_mode', action='store_true',
                        help='only keeps examples with annotation in valid and test set')
    parser.add_argument('--output_pkl', action='store_true',
                        help='output a whole pkl but not separate files')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='number of processers will be used')
    args = parser.parse_args()
    target_path = args.input_dir
    for key in ('train', 'valid', 'test'):
        setattr(args, key, os.path.join(target_path, key + '.txt'))
    # setattr(args, 'output_dir', os.path.join(target_path, ))
    logfilename = os.path.join(args.output_dir, '.log')
    logging.config.dictConfig(log_dict.logging_config_dict(2, logfilename))
    logger = logging.getLogger(__name__)
    logger.info(args)

    # process(args)
    process_features_to_npz(args)
    print("$End Parsed params: " + str(args))
