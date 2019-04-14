import json
import logging
import os
import numpy as np
from abc import ABC, abstractmethod

from definitions import tools, settings
from definitions.arguments import LearningArguments
from definitions.log_dict import ArgLog
from processing.OiePreprocessor import FeatureLexiconCompact


# base class for data feeder
class Feeder(ABC):
    def __init__(self, data):
        self.logger = logging.getLogger()
        self.logger.info(ArgLog('DataFeeder', self.__class__.__name__))
        if isinstance(data, str) and os.path.exists(data):
            self.data = self.load_data(data)
            self.logger.info(ArgLog(self.__class__.__name__ + '-Load_From', data))
        else:
            self.data = data

    @abstractmethod
    def process_batch(self, key, indices):
        pass

    def get_func(self, key):
        if key in ('train', 'valid', 'test'):
            return lambda x: self.process_batch(key, x)
        else:
            return None

    def __getitem__(self, item):
        return self.data[item]

    @staticmethod
    def load_data(dir):
        d = np.load(dir)
        data = {}
        for key in ['train', 'valid', 'test']:
            data[key] = d[key]
        return data


class TripletsFeeder(Feeder):
    def __init__(self, data, ):
        super().__init__(data)
        self.data_len = {}
        self.get_unique_triplets(self.data)

    def process_batch(self, key, indices):
        data = self[key]
        raw_batch = data[indices]
        feed_dict = []
        if raw_batch.shape[1] == 3:
            args1, args2, rels = np.split(raw_batch, 3, 1)
        else:
            uid, args1, args2, rels = np.split(raw_batch, 4, 1)
            feed_dict += [('uid', uid)]
        feed_dict += [('args1', args1), ('args2', args2), ('relation_input', np.squeeze(rels, 1))]
        return feed_dict

    def get_func(self, key):
        if key in ('train', 'valid', 'test', 'valid-decoder', 'test-decoder'):
            return lambda x: self.process_batch(key, x)
        else:
            return None

    def get_unique_triplets(self, data):
        for key in ['valid', 'test']:
            # self.data[key + '-decoder'] = np.unique(data[key][:, [1, 2, 3]], axis=0)
            self.data[key + '-decoder'] = data[key]

    def get_relation_labels(self, key):
        return self.data[key][:, -1]

    def get_data_len(self, data_key=None):
        if len(self.data_len) != len(self.data):
            for key in self.data:
                self.data_len[key] = len(self.data[key])
        if data_key is None:
            return self.data_len
        else:
            return self.data_len[data_key]


class LookupIDFeeder(Feeder):
    def __init__(self, data, lookup_dict):
        super().__init__(None)
        self.data = {'train': self.annotate_data(data, lookup_dict)}
        self.lookup_dict = lookup_dict

    def process_batch(self, key, indices):
        data = self[key]
        return {'relation_lookup_id': data[indices]}

    def get_func(self, key):
        if key != 'train':
            return None
        return super().get_func(key)

    @staticmethod
    def annotate_data(data, lookup_dict):
        ids = []
        for _, a1, a2, _ in data['train']:
            ids.append(lookup_dict[(a1, a2)])
        return np.array(ids, dtype=int)


class ArgsTranslateFeeder(TripletsFeeder):
    def __init__(self, data, translator, ):
        super().__init__(data, )
        self.translator = translator
        if isinstance(translator, str):
            self.translator = self.load_translator(translator)

    def process_batch(self, key, indices):
        feed_list = super().process_batch(key, indices)
        a1, a2 = feed_list[-3][1], feed_list[-2][1]
        e1, e2 = self.translator[a1], self.translator[a2]
        mask = np.squeeze(np.logical_and(e1 != -1, e2 != -1), axis=1)
        e1[np.logical_not(mask)] = 0
        e2[np.logical_not(mask)] = 0
        # todo: control graph if mask is all False

        feed_list += [('kbarg1', e1),
                      ('kbarg2', e2),
                      ('reg_mask', mask)
                      ]
        return feed_list

    @staticmethod
    def load_translator(p):
        return np.load(p)


class NegSampleFeeder(Feeder):
    def __init__(self, entity_lex, neg_num, neg_dist_power):
        super().__init__(None)
        self.entity_lex = entity_lex
        self.neg_num = neg_num
        self.neg_dist_power = neg_dist_power
        self.neg_sample_cum = None
        if np.abs(neg_dist_power - 1) > 1e-5:
            self.neg_sample_cum = entity_lex.neg_sample_cum(neg_dist_power)[1]

    def __getitem__(self, item):
        return None

    def get_func(self, key):
        if key != 'train':
            return None
        return super().get_func(key)

    def process_batch(self, key, indices):
        if not self.neg_num:
            return 0  # cannot be None because feed_dict takes number as input
        batch_size = len(indices)
        shape = (2, batch_size, self.neg_num)
        if np.abs(self.neg_dist_power - 1) < 1e-5:
            neg = np.random.randint(0, len(self.entity_lex), size=shape)
        else:
            cutoffs = self.neg_sample_cum
            neg = np.searchsorted(cutoffs, np.random.uniform(0, cutoffs[-1], size=shape))

        return [('neg1', neg[0]),
                ('neg2', neg[1])
                ]

    @staticmethod
    def load_data(dir):
        return None


class FeaturesFeeder(Feeder):
    def __init__(self, features, feature_num, ):
        super().__init__(features)
        self.feature_num = feature_num

    def process_batch(self, key, indices):
        data = self[key]
        raw_batch = data[indices]
        # sparse feature input
        feat_batch = []
        batch_size = len(indices)
        for i, example in enumerate(raw_batch):
            fl = len(example)
            feat_batch.append(np.stack(
                (np.full(shape=fl, fill_value=i, dtype=int), example),
                axis=1,
            ))

        feat_indices = np.concatenate(feat_batch, 0).astype(np.int64)

        feed_dict = [('feat_indices', feat_indices),
                     ('feat_values', np.ones(len(feat_indices), dtype=float)),
                     ('feat_dense_shape', np.array((batch_size, self.feature_num), dtype=np.int64))
                     ]
        return feed_dict


class SentenceIDFeeder(Feeder):
    def __init__(self, data):
        super().__init__(data)
        data_lens = []
        for key in ['train', 'valid', 'test']:
            data_lens.append(len(self.data[key]))
        self.data_lens = data_lens
        self.data = {'train': np.arange(data_lens[0]),
                     'valid': np.arange(data_lens[1]) + data_lens[0],
                     'test': np.arange(data_lens[2]) + data_lens[1] + data_lens[2]
                     }

    def process_batch(self, key, indices):
        data = self[key]
        return [('sentence_id', data[indices])]


# sentence id

class DataIter:
    def __init__(self, data_getters, data_len, total_epoch, batch_size, shuffle_flag=True, ):
        self.logger = logging.getLogger(__name__)

        self.current_idx = 0
        self.epoch = 0
        self.data_getters = data_getters
        self.batch_size = batch_size
        self.data_len = data_len
        self.total_epoch = total_epoch
        self.shuffle_flag = shuffle_flag

        self.index_order = np.arange(len(self), dtype=int)
        if shuffle_flag:
            np.random.shuffle(self.index_order)

    def __iter__(self):
        self.current_idx = 0
        self.epoch = 0
        return self

    def __next__(self):
        if self.total_epoch and self.epoch >= self.total_epoch:
            self.epoch = 0
            self.current_idx = 0
            raise StopIteration
        # get batch
        feed_dict = []
        batch_size = self.batch_size
        start_idx = self.current_idx % len(self)
        indices = self.index_order[start_idx:start_idx + batch_size]

        for func in self.data_getters:
            feed_dict += func(indices)
        feed_dict = dict(feed_dict)

        batch_size = len(indices)
        # iterator
        self.current_idx += batch_size
        current_epoch = self.current_idx // len(self)
        if self.epoch < current_epoch:
            self.epoch = current_epoch
            if self.shuffle_flag:
                np.random.shuffle(self.index_order)
        return feed_dict

    def __len__(self):
        return self.data_len


data_feeder = {'triplets': TripletsFeeder,
               'feature': FeaturesFeeder,
               'neg': NegSampleFeeder
               }


class DataSetManager:
    """
    load data, load data dir, dataiter
    define iter: data getters, data_len, total_epoch, batch_size, shuffle_flag
    """

    def __init__(self, opts):
        """

        :param input_dir:
        :param kwargs: datatype=dir, e.g: triplets=
        DataFeeders init/ load data
        """
        self.logger = logging.getLogger(__name__)
        self.options = opts
        self.data_feeders = []
        self.data_lens = {}
        self._dict = {}
        self.eval_batch_size = int(7 * 2 ** 30 / (2000 * 32) / 50 / 2 // 100) * 100
        self.extkb_parameters = {}
        self.sentence_vec = None
        self.word_vec = None
        self._load_data()

        self._get_iters()

    def _load_data(self, ):
        opts = self.options
        input_dir = opts.input_dir

        text_dir = os.path.join(input_dir, 'text')
        kb_dir = os.path.join(input_dir, 'kb')
        self.logger.info(ArgLog('input_dir', input_dir))

        with open(os.path.join(text_dir, 'stats.json'), 'r') as f:
            self.text_stats = json.load(f)
        with open(os.path.join(kb_dir, 'stats.json'), 'r') as f:
            self.kb_stats = json.load(f)
        self.text_entity_lex = FeatureLexiconCompact.load_txt(os.path.join(text_dir, 'entity_lexicon.txt'))

    def _get_iters(self, ):
        opts = self.options
        self.data_lens = self.triplets_feeder.get_data_len()
        for key in ['train', 'valid', 'test', 'valid-decoder', 'test-decoder']:
            data_getters = []
            if key == 'train':
                shuffle = True
                batch_size = opts.batch_size
                epochs = opts.epochs
            else:
                shuffle = False
                batch_size = self.eval_batch_size
                epochs = 1
            for feeder in self.data_feeders:
                func = feeder.get_func(key)
                if func is not None:
                    data_getters.append(func)

            self._dict[key] = DataIter(data_getters, self.triplets_feeder.get_data_len(key),
                                       epochs, batch_size, shuffle_flag=shuffle, )
        self.train_iter, self.valid_iter, self.test_iter = self._dict['train'], self._dict['valid'], self._dict['test']

    def _encoder_data_loader(self, encoder, text_dir):
        opts = self.options
        encoder = encoder.split('+')
        feeders = []
        if 'F' in encoder:
            load_feature_dir = os.path.join(text_dir, 'features.npz')
            feeders.append(FeaturesFeeder(load_feature_dir, self.text_stats['feature_num']))
        if 'SE' in encoder:
            feeders.append(SentenceIDFeeder(self.text_triplets))
        if 'EV' in encoder:
            extkb_dir = os.path.join(opts.input_dir, 'kb', 'model', opts.load_ext_kb_dir)
            self.extkb_dir = tools.complete_dir(extkb_dir)
            kb_embedding_key = settings.ENTITY_EMBEDDING_NAME
            self.get_kb_entity_embeddings()
        return feeders

    def _get_data_len(self, triplets):
        for key in ['train', 'valid', 'test']:
            self.data_lens[key] = len(triplets[key])

    def __getitem__(self, key):
        return self._dict[key]

    def get_relation_labels(self, key):
        return self.triplets_feeder.get_relation_labels(key)

    def get_extkb_parameters(self, key):
        if key not in self.extkb_parameters:
            opts = self.options
            extkb_dir = os.path.join(opts.input_dir, 'kb', 'model', opts.load_ext_kb_dir)
            extkb_dir = tools.complete_dir(extkb_dir)
            if key == 'opts':
                ext_kb_options = LearningArguments()
                ext_kb_options.load(os.path.join(extkb_dir, 'options_000_.json'))
                self.extkb_parameters[key] = ext_kb_options
            elif key == 'translator':
                self.extkb_parameters[key] = np.load(os.path.join(opts.input_dir, 'kb', 't2kid.npy'))
            else:
                ext_kb_parameters = np.load(os.path.join(extkb_dir, settings.save_parameters_file_name))
                value = ext_kb_parameters['ext_kb/' + key]
                if key == settings.ENTITY_EMBEDDING_NAME:
                    kbopts = self.get_extkb_parameters('opts')
                    if kbopts.model or kbopts.emb_normalize:
                        value = value / np.linalg.norm(value, axis=-1, keepdims=True)
                self.extkb_parameters[key] = value
        return self.extkb_parameters[key]

    def get_sentence_vec(self):
        if self.sentence_vec is None:
            npz_dir = os.path.join(self.options.input_dir, 'text', 'sentence_embedding.npz')
            vec = []
            if os.path.exists(npz_dir):
                data = np.load(npz_dir)
                for key in ['train', 'valid', 'test']:
                    vec.append(data[key])
            else:
                for key in ['train', 'valid', 'test']:
                    npy_dir = os.path.join(self.options.input_dir, 'text', key + '_sentence_embedding.npy')
                    data = np.load(npy_dir)
                    vec.append(data)
            self.sentence_vec = np.concatenate(vec, axis=0).astype(np.float32)
            # data = np.load(os.path.join(self.options.input_dir, 'text', 'sentence_embedding.npz'))
        return self.sentence_vec

    def get_word_vec(self):
        if self.word_vec is None:
            self.word_vec = np.load(os.path.join(self.options.input_dir, 'text', 'word_vec.npy'))
        return self.word_vec
