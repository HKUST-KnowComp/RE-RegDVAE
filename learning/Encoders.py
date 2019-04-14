import logging
import numpy as np
from abc import ABC, abstractmethod

import tensorflow as tf

from definitions import tools, settings
from definitions.log_dict import ArgLog


class InputPlaceholders:
    def __init__(self):
        self.input_placeholders = {}
        self.init_placeholders = {}
        self.init_feed_dict = {}

    def get(self, name, dtype=None, shape=None, ):
        phs = self.input_placeholders
        if name not in phs:
            phs[name] = tf.placeholder(dtype, shape, name)
        return phs[name]

    def get_var(self, name, initializer, dtype=None, shape=None, ):
        phs = self.input_placeholders
        if name not in phs:
            phs[name] = tf.get_variable(name, initializer=initializer, dtype=dtype, shape=shape,
                                        trainable=False)
        return phs[name]

    def get_init(self, name, dtype, value):
        phs = self.init_placeholders
        if name not in phs:
            ph = tf.placeholder(dtype, value.shape, name)
            phs[name] = ph
            self.init_feed_dict[ph] = value
        return phs[name]


class Encoder(ABC):
    def __init__(self, opts, stats, data, input_phs: InputPlaceholders):
        self.opts = opts
        self.input_phs = input_phs
        self.data = data
        self.relation_num = relation_num = opts.relations_number
        self.logger = logging.getLogger(__name__)
        self.logger.info(ArgLog('encoder', self.__class__.__name__))
        self.logger.info(ArgLog('relation_num', relation_num))

    @abstractmethod
    def get_input_placeholders(self, ):
        pass

    @abstractmethod
    def get_r_scores(self):
        pass


class FeatureEncoder(Encoder):
    def __init__(self, opts, stats, data, input_phs):
        super().__init__(opts, stats, data, input_phs)
        self.feature_dim = stats['feature_num']
        self.logger.info(ArgLog('feature_num', self.feature_dim))

    def get_input_placeholders(self):
        with tf.name_scope("feats_input"):
            feats_indices = self.input_phs.get(dtype=tf.int64, name="feat_indices", shape=None)
            feats_values = self.input_phs.get(dtype=tf.float32, name="feat_values", shape=None)
            feats_dense_shape = self.input_phs.get(dtype=tf.int64, name="feat_dense_shape", shape=None)

        tf.get_collection_ref(tools.FeatureInput).extend([feats_indices, feats_values, feats_dense_shape])
        input_phs = [feats_indices, feats_values, feats_dense_shape]
        return input_phs

    def get_r_scores(self):
        feats_input = self.get_input_placeholders()
        (feats_indices, feats_values, feats_dense_shape) = feats_input
        feature_sparse_input = tf.SparseTensor(indices=feats_indices, values=feats_values,
                                               dense_shape=feats_dense_shape)
        feature_sparse_input_reorder = tf.sparse_reorder(feature_sparse_input,
                                                         name="feature_sparse_input_reorder",
                                                         )

        self.relation_weights = tf.get_variable("relation_weights",
                                                initializer=tf.random_uniform_initializer(-1e-3, 1e-3),
                                                shape=(self.feature_dim, self.relation_num),
                                                collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                             tools.RelationKey,
                                                             tf.GraphKeys.MODEL_VARIABLES,
                                                             tf.GraphKeys.WEIGHTS],
                                                dtype=tf.float32,
                                                )
        tools.norm_summary(self.relation_weights, axis=-1, )

        epsilon = 1e-10
        # [batch, feat] x [feat, rel]  -> [batch, rel]
        relation_scores = tf.sparse_tensor_dense_matmul(feature_sparse_input_reorder,
                                                        self.relation_weights, name='feature_scores') + epsilon
        return relation_scores


class SVEncoder(Encoder):
    def __init__(self, opts, stats, data, input_phs):
        super().__init__(opts, stats, data, input_phs)

    def get_input_placeholders(self):
        input_phs = [self.input_phs.get(dtype=tf.int32, shape=(None,), name='sentence_id'),
                     self.input_phs.get(name='args1', dtype=tf.int32, shape=(None, 1)),
                     self.input_phs.get(dtype=tf.int32, shape=(None, 1), name="args2")
                     ]
        return input_phs

    def get_r_scores(self):
        sentence_id_ph, arg1_ph, arg2_ph = self.get_input_placeholders()
        # arg1_ph, arg2_ph = [tf.squeeze(x, axis=-1) for x in (arg1_ph, arg2_ph)]

        entity_num, entity_dim = self.data.text_stats['entity_num'], 10
        sentence_vectors = self.input_phs.get_init('sentence_vecs_init', value=self.data.get_sentence_vec(),
                                                   dtype=tf.float32, )
        self.sentence_dim = sentence_vectors.shape[1]
        self.logger.info(ArgLog("sentence_dim", self.sentence_dim))
        self.entity_embeddings = tf.get_variable('entity_weights',
                                                 shape=(entity_num, entity_dim),
                                                 collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                              tf.GraphKeys.MODEL_VARIABLES,
                                                              tf.GraphKeys.WEIGHTS]
                                                 )
        self.sentence_vecs = tf.get_variable('sentence_vecs', initializer=sentence_vectors, trainable=False,
                                             dtype=tf.float32)

        vecs = []
        vecs.append(tf.nn.embedding_lookup(self.sentence_vecs, sentence_id_ph))

        # b, 2*k+s
        vecs = tf.concat(vecs, axis=1)

        hidden_lens = []
        output_len = self.relation_num
        input_len = self.sentence_dim + 2 * entity_dim
        vecs = vecs
        # -> b, r
        for i, hl in enumerate(hidden_lens + [output_len]):
            w = tf.get_variable('weight_{}'.format(i), shape=(input_len, hl),
                                collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                             tf.GraphKeys.MODEL_VARIABLES,
                                             tf.GraphKeys.WEIGHTS]
                                )

            preout = tf.matmul(vecs, w)
            if i != len(hidden_lens):
                b = tf.get_variable('bias_{}'.format(i), shape=(hl,),
                                    regularizer=tf.no_regularizer,
                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                 tf.GraphKeys.MODEL_VARIABLES,
                                                 tf.GraphKeys.BIASES]
                                    )
                preout += b
                out = tf.nn.relu(preout)
            else:
                out = preout
            vecs = out
            input_len = hl

        # batch_size , dim
        # (batch_size, dim) . (dim, relation_num) -> (batch_size, relation_num)
        relation_scores = out
        return relation_scores


class EntityVEncoder(Encoder):
    def __init__(self, opts, stats, data, input_phs: InputPlaceholders):
        super().__init__(opts, stats, data, input_phs)
        self.entity_num = stats['entity_num']

    def get_input_placeholders(self, ):
        input_phs = [self.input_phs.get(dtype=tf.int32, shape=(None, 1), name='args1'),
                     self.input_phs.get(dtype=tf.int32, shape=(None, 1), name='args2')
                     ]
        return input_phs

    def get_r_scores(self):
        # entity embeddings
        # normalize input
        entity_embeddings = self.data.get_kb_entity_embeddings()
        entity_embeddings = entity_embeddings / np.linalg.norm(entity_embeddings, axis=-1, keepdims=True)
        k = entity_embeddings.shape[1]
        evecs = np.zeros((self.entity_num, k), dtype=np.float32)  # type: np.ndarray
        translator = self.data.get_extkb_parameters('translator')
        exists_mask = translator > 0
        evecs[exists_mask] = entity_embeddings[translator[exists_mask]]
        evecs.astype(np.float32)

        # placeholder
        # 2* (b, )
        args = [tf.squeeze(x, axis=-1) for x in self.get_input_placeholders()]

        # get parameters
        self.A = tf.get_variable("entity_embeddings",
                                 initializer=evecs,
                                 trainable=False,
                                 dtype=tf.float32
                                 )
        self.weights = tf.get_variable("entity_vec_weights",
                                       shape=(k * 2, self.relation_num),
                                       collections=[tf.GraphKeys.GLOBAL_VARIABLES, tools.RelationKey,
                                                    tf.GraphKeys.MODEL_VARIABLES, tf.GraphKeys.WEIGHTS], )
        tools.norm_summary(self.weights, axis=-1, )
        # (b, 2k)
        embs = [tf.nn.embedding_lookup(self.A, args[i]) for i in (0, 1)]
        # (2b, k)
        e_vec = tf.concat(embs, axis=1, name='entity_vec_input')
        # (b,2 k) . (2k, r) -> b,r
        relation_scores = tf.matmul(e_vec, self.weights, name='entity_v_encoder_score')
        return relation_scores


class AllEncoder:
    def __init__(self, opts, data, input_phs):
        encoder_key = opts.encoder
        encoder_key = encoder_key.split('+')
        self.opts = opts
        self.data = data
        self.encoders = {}
        for key in encoder_key:
            self.encoders[key] = EncoderDict[key](opts, data.text_stats, data, input_phs)

    def get_input_placeholders(self):
        phs = []
        for v in self.encoders.values():
            phs += v.get_input_placeholders()
        return phs

    def build_relation_probabilities(self):
        score = 0
        self.bias = tf.get_variable("relation_bias",
                                    shape=(1, self.opts.relations_number),
                                    initializer=tf.zeros_initializer(),
                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                 tf.GraphKeys.MODEL_VARIABLES,
                                                 tools.RelationKey,
                                                 tf.GraphKeys.BIASES],
                                    regularizer=tf.no_regularizer,
                                    dtype=tf.float32,
                                    )
        for v in self.encoders.values():
            ecd_score = v.get_r_scores()
            tf.add_to_collection(tools.HistogramSummaryTensors, ecd_score)
            score += ecd_score
        score += self.bias
        relation_probabilities = tf.nn.softmax(score, dim=-1, name="relation_probabilities", )
        tf.add_to_collection(tools.HistogramSummaryTensors, relation_probabilities)
        return relation_probabilities


EncoderDict = {'F': FeatureEncoder,
               'SE': SVEncoder,
               'EV': EntityVEncoder,
}
