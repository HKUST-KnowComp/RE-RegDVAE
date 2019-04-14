import logging
import os

from definitions.log_dict import ArgLog
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from definitions import tools, arguments

from definitions.settings import TransE_ENTITY_EMBEDDING_NAME


class Regularizer(ABC):
    def __init__(self, opts: arguments.LearningArguments, data, input_phs):
        self.logger = logging.getLogger(__name__)
        self.logger.info(ArgLog('regularizer', self.__class__.__name__))
        self.opts = opts
        self.g = tf.get_default_graph()
        self.input_placeholders = input_phs
        self.data = data

    @abstractmethod
    def get_input_placeholders(self, ):
        pass

    def get_score(self, relation_probabilities):
        return tf.constant(0.)


class TransERegularizer(Regularizer):
    """
    Use TransE entity embedding to regularize
    """

    def __init__(self, opts: arguments.LearningArguments, data, input_phs, ):
        super().__init__(opts, data, input_phs)
        self._create_entity_mats()

    def get_input_placeholders(self, ):
        kb_arg1 = self.input_placeholders.get(dtype=tf.int32, shape=(None, 1), name='kbarg1')
        kb_arg2 = self.input_placeholders.get(dtype=tf.int32, shape=(None, 1), name='kbarg2')
        # https://www.tensorflow.org/api_docs/python/tf/boolean_mask
        reg_mask = self.input_placeholders.get(dtype=tf.bool, shape=(None,), name='reg_mask')
        return [kb_arg1, kb_arg2, reg_mask]

    def _create_entity_mats(self):
        opts = self.opts
        entity_embeddings = self.data.get_kb_entity_embeddings()
        self.transE_entity_vec = tools.get_or_create_variable(TransE_ENTITY_EMBEDDING_NAME,
                                                              initializer=entity_embeddings,
                                                              trainable=False)

    def get_score(self, relation_probabilities):
        """
        scores without relation labels
        :param relation_probabilities: (b, r)
        :return: ()
        """

        E = self.transE_entity_vec
        kb_arg1, kb_arg2, reg_mask = self.get_input_placeholders()
        kb_arg1 = tf.boolean_mask(tf.squeeze(kb_arg1, 1), reg_mask, axis=0)
        kb_arg2 = tf.boolean_mask(tf.squeeze(kb_arg2, 1), reg_mask, axis=0)

        # (b', r)
        relation_probabilities = tf.boolean_mask(relation_probabilities, reg_mask, axis=0)
        # (b', b')
        r_p_distance = self._get_batch_relation_probability_distance(relation_probabilities)
        tf.add_to_collection(tools.HistogramSummaryTensors, r_p_distance)

        if self.opts.cos_threshold >= 1.0:
            equal_matrix, mask = self._get_batch_equal_matrix(kb_arg1, kb_arg2)
            pair_cnt = tf.reduce_sum(mask, name='pair_cnt')
            tf.add_to_collection(tools.AddToScalarSummaryKey, pair_cnt)

            return self._cal_Equal_score(r_p_distance, equal_matrix, pair_cnt)
        else:
            transE_embeddings_ph = tf.nn.embedding_lookup(E, kb_arg1)
            transE_embeddings_pt = tf.nn.embedding_lookup(E, kb_arg2)
            edges = tf.subtract(transE_embeddings_pt, transE_embeddings_ph)

            # (b', b')
            edge_similarity, mask = self._get_batch_edge_similarity(edges)

            pair_cnt = tf.reduce_sum(mask, name='pair_cnt')
            tf.add_to_collection(tools.AddToScalarSummaryKey, pair_cnt)

            return self._cal_TransE_score(r_p_distance, edge_similarity, pair_cnt)

    def _get_batch_equal_matrix(self, kb_arg1, kb_arg2):
        opts = self.opts
        kb_arg1_matrix = tf.tile(tf.expand_dims(kb_arg1, axis=1),
                                 [1, tf.shape(kb_arg1)[0]])
        kb_arg1_matrix_T = tf.transpose(kb_arg1_matrix, [1, 0])
        kb_arg2_matrix = tf.tile(tf.expand_dims(kb_arg2, axis=1),
                                 [1, tf.shape(kb_arg2)[0]])
        kb_arg2_matrix_T = tf.transpose(kb_arg2_matrix, [1, 0])
        mask1 = tf.logical_and(
            tf.equal(kb_arg1_matrix, kb_arg1_matrix_T),
            tf.equal(kb_arg2_matrix, kb_arg2_matrix_T))
        mask2 = tf.logical_and(
            tf.equal(kb_arg1_matrix, kb_arg2_matrix_T),
            tf.equal(kb_arg2_matrix, kb_arg1_matrix_T))
        mask = tf.cast(tf.logical_or(mask1, mask2), dtype=tf.float32)
        mask = tf.matrix_set_diag(mask, tf.zeros((tf.shape(mask)[0],), dtype=tf.float32))
        if opts.abs_cos:
            equal_matrix = mask
        else:
            # if mask1 = 1 and mask2 = 1, then kbarg1 = kbarg2. 
            # we can easily think it implies "one" relation.
            # that is to say, we set this "similarity" to 1
            equal_matrix = tf.cast(mask1, dtype=tf.float32) - tf.cast(mask2, dtype=tf.float32) + \
                           tf.cast(tf.logical_and(mask1, mask2), dtype=tf.float32)
            equal_matrix = tf.matrix_set_diag(equal_matrix, tf.zeros((tf.shape(equal_matrix)[0],), dtype=tf.float32))
        return equal_matrix, mask

    def _get_batch_edge_similarity(self, edges):
        opts = self.opts
        if opts.adj_cos:
            edges_mean = tf.reduce_mean(edges, axis=0, keepdims=True)
            edges = edges - edges_mean
        edges = tf.nn.l2_normalize(edges, dim=1)
        cos_similarity = tf.matmul(edges, tf.transpose(edges, [1, 0]), name='cosine_similarity')
        tf.add_to_collection(tools.HistogramSummaryTensors, cos_similarity)

        abs_cos_similarity = tf.abs(cos_similarity)
        mask = tf.cast(tf.greater_equal(abs_cos_similarity, opts.cos_threshold), dtype=tf.float32)
        mask = tf.matrix_set_diag(mask, tf.zeros((tf.shape(mask)[0],), dtype=tf.float32))

        if opts.abs_cos:
            return abs_cos_similarity * mask, mask
        else:
            return cos_similarity * mask, mask

    def _get_batch_relation_probability_distance(self, relation_probabilities):
        opts = self.opts
        # b, 1, r ->  b, b, r
        r_p_matrix = tf.tile(tf.expand_dims(relation_probabilities, axis=1),
                            [1, tf.shape(relation_probabilities)[0], 1])
        r_p_matrix_T = tf.transpose(r_p_matrix, [1, 0, 2])
        if opts.distance == 'euclidean':
            self.logger.debug('Prob(R) distance: Euclidean')
            r_p_matrix_distance = tf.squared_difference(r_p_matrix, r_p_matrix_T)
            epsilon = 1e-12
            r_p_distance = tf.sqrt(tf.maximum(tf.reduce_sum(r_p_matrix_distance, 2), epsilon), name='euclidean_distance')
        elif opts.distance == 'JS':
            self.logger.debug('Prob(R) distance: Jensen-Shannon')
            average = (r_p_matrix + r_p_matrix_T) / 2
            r_p_distance = tf.divide(
                tools.tf_cal_kl(r_p_matrix, average) + tools.tf_cal_kl(r_p_matrix_T, average), 
                2., name='JS_divergence')
        elif opts.distance == 'KL':
            self.logger.debug('Prob(R) distance: KL')
            r_p_distance = tools.tf_cal_kl(r_p_matrix, r_p_matrix_T, name='KL_divergence')
        else:
            raise NotImplementedError()
        return r_p_distance

    def _cal_Equal_score(self, relation_probability_distance, mask, pair_cnt):
        return tf.multiply(self.opts.batch_size / (pair_cnt + 1e-6),
                           tf.reduce_sum(relation_probability_distance * mask), name='Equal_score')

    def _cal_TransE_score(self, relation_probability_distance, edge_similarity, pair_cnt):
        transe_score = tf.multiply(self.opts.batch_size / (pair_cnt + 1e-6),
                                   tf.reduce_sum(relation_probability_distance * edge_similarity), name='TransE_score')
        return transe_score


RegularizerDict = {"": Regularizer,
                   "None": Regularizer,
                   "TransE": TransERegularizer,
                   'TD': TransERegularizer,
                   }
