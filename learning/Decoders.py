import logging
from definitions.log_dict import ArgLog
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from definitions import tools, arguments

from definitions.settings import ENTITY_EMBEDDING_NAME, ENTITY_BIAS_NAME, BILINEAR_RELATION_MATRIX_NAME, \
    SP_RELATION_EMBEDDING_NAME, TransE_RELATION_EMBEDDING_NAME


class Decoder(ABC):
    def __init__(self, opts: arguments.LearningArguments, entity_number, relations_number=-1, emb_size=0, data=None):
        self.logger = logging.getLogger(__name__)
        self.logger.info(ArgLog('decoder', self.__class__.__name__))

        self.data = data
        self.embedding_names = [['entity_embeddings', ], ['positive', 'negative'], ['1st', '2nd'], ]
        self.options = opts
        self.relations_number = relations_number if relations_number > 0 else opts.relations_number
        self.entity_number = entity_number
        self.max_norm = 1 if self.options.emb_normalize else None
        self.emb_size = emb_size if emb_size > 0 else opts.embed_size
        self.g = tf.get_default_graph()

        self._create_entity_mats()
        self._create_relation_mats()

    def _create_entity_mats(self):
        if 'Fix' in self.options.model:
            entity_embeddings = self.data.get_kb_entity_embeddings()

            k = entity_embeddings.shape[1]
            evecs = np.random.uniform(-0.01, 0.01, size=(self.entity_number, k), ).astype(np.float32)  # type: np.ndarray
            translator = self.data.get_extkb_parameters('translator')
            exists_mask = translator > 0
            evecs[exists_mask] = entity_embeddings[translator[exists_mask]]
            evecs.astype(np.float32)
            A = tools.get_or_create_variable(ENTITY_EMBEDDING_NAME,
                                             dtype=tf.float32,
                                             initializer=evecs,
                                             collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                          tf.GraphKeys.MODEL_VARIABLES,
                                                          tf.GraphKeys.WEIGHTS],
                                             trainable=False
                                             )
            random_tensor = tools.get_or_create_variable('random_embedding',
                                                         dtype=tf.float32,
                                                         initializer=tf.random_uniform_initializer(-0.01, 0.01, dtype=tf.float32),
                                                         shape=(self.entity_number, self.emb_size),
                                                         collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                                      tf.GraphKeys.MODEL_VARIABLES,],
                                                         )
            A = tf.where(exists_mask, A, random_tensor)
        else:
            A = tools.get_or_create_variable(ENTITY_EMBEDDING_NAME,
                                         dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(-0.01, 0.01, ),
                                         shape=(self.entity_number, self.emb_size),
                                         collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                      tf.GraphKeys.MODEL_VARIABLES,
                                                      tf.GraphKeys.WEIGHTS],
                                         )
        self.A = A
        tools.norm_summary(A, axis=-1, )

    @abstractmethod
    def _create_relation_mats(self):
        pass

    @abstractmethod
    def _create_weighted_relation_mats(self, r_probs):
        pass

    @abstractmethod
    def _cal_decoder_score(self, head_embs, tail_embs, relation_probs=None):
        """

        :param head_embs: (b,m,k)
        :param tail_embs: (b,n,k)
        :return: (b, r, n)
        """
        return 0.

    @abstractmethod
    def _cal_triplet_score(self, head_embs, tail_embs, rl_ids):
        """

        :param head_embs: (a, m, k) b, 1 | b, 1
        :param tail_embs: (c, n, k) 1, n | b, n
        :param r_ids: (b)
        :return: (b, m)
        """
        return 0.

    def get_batch_score(self, p_ids: list, n_ids: list, relation_probabilities, original_score=False):
        """
        scores without relation labels
        :param original_score: apply relation probabilities to relation mat if True
        :param p_ids: positive entity ids, [head, tail]
        :param n_ids: negative, [head, tail]
        :param relation_probabilities: (b, r)
        :return: (b, )
        """
        A = self.A
        args_id = [p_ids, n_ids]
        # [p or n] [ f or s]  (b, m, k)  m = 1 or n
        embeddings = [[tf.nn.embedding_lookup(A, args_id[p_n][f_s],
                                              name=self._get_embedding_name(0, p_n, f_s),
                                              max_norm=self.max_norm) for f_s in (0, 1)
                       ] for p_n in (0, 1,)
                      ]
        (ph, pt), (nh, nt) = embeddings

        # calculate relation probabilities
        predict_r_score = tf.squeeze(self._cal_decoder_score(ph, pt), axis=-1)
        self.relation_probs = tf.nn.softmax(predict_r_score, axis=-1, name='decoder_relation_probabilities')
        self.entropy = tools.cal_entropy(self.relation_probs, 'decoder_entropy')
        self.predictions = tf.argmax(self.relation_probs, axis=-1, name='relation_predictions')
        tf.add_to_collection(tools.HistogramSummaryTensors, self.relation_probs)
        # (b, r, 1)
        cal_kernel_r_p = relation_probabilities if original_score else None
        p_scores = self._cal_decoder_score(ph, pt, cal_kernel_r_p)
        nt_scores = - self._cal_decoder_score(ph, nt, cal_kernel_r_p)
        nh_scores = - self._cal_decoder_score(nh, pt, cal_kernel_r_p)

        scores = [tf.reduce_sum(tf.log_sigmoid(x), axis=-1) for x in (p_scores, nt_scores, nh_scores)]

        # (b, r) or b
        # append an additional p_scores
        batch_score = tf.add_n(scores + [scores[0]], name='r_scores')
        if not original_score:
            batch_score = tf.multiply(relation_probabilities, batch_score, 'weighted_r_scores')

            batch_score = tf.reduce_sum(batch_score, axis=1,
                                        name='batch_scores')
        return batch_score

    def get_batch_score_with_relation_label(self, p_ids, n_ids, r_ids):
        A = self.A
        args_id = [p_ids, n_ids]
        # [p or n] [ f or s]  (b, m, k)  m = 1 or n
        embeddings = [[tf.nn.embedding_lookup(A, args_id[p_n][f_s],
                                              name=self._get_embedding_name(0, p_n, f_s),
                                              max_norm=self.max_norm) for f_s in (0, 1)
                       ] for p_n in (0, 1,)
                      ]
        (ph, pt), (nh, nt) = embeddings
        # (b, 1)
        p_scores = self._cal_triplet_score(ph, pt, r_ids)
        # (b, n)
        nt_scores = - self._cal_triplet_score(ph, nt, r_ids)
        nh_scores = - self._cal_triplet_score(nh, pt, r_ids)
        scores = tf.concat([p_scores, p_scores, nt_scores, nh_scores], axis=1)

        # (b,)
        log_sigmoid_scores = tf.reduce_sum(tf.log_sigmoid(scores, ), axis=[1])
        return log_sigmoid_scores

    def get_score(self, p_ids: list, n_ids: list, relation_probabilities, ods=False):
        if relation_probabilities.dtype.is_floating:
            score = tf.reduce_sum(self.get_batch_score(p_ids, n_ids, relation_probabilities, ods), name='decoder_loss')
        else:
            score = tf.reduce_sum(self.get_batch_score_with_relation_label(p_ids, n_ids, relation_probabilities),
                                  name='decoder_loss')
        return score

    @abstractmethod
    def regularize_by_id(self, oie_decoder, lookup_ids, relation_probabilities):
        return 0

    def _regularize_by_id_score(self, res, name):
        res = tf.reduce_mean(res ** 2) / 2
        return tf.identity(res, name)

    def _link_prediction_graph(self, e_ids, r_ids):
        #
        # embeddings = self.A
        # predict tail score
        #  [1,1, k]
        A = self.A
        # list[ (1, 1, k) ]
        fact_embs = [tf.nn.embedding_lookup(A, e_ids[i],
                                            name="fact_embedding_batch") for i in range(2)]
        all_entity_embeddings = self.A
        if r_ids is not None:
            head_score = tf.identity(self._cal_triplet_score(all_entity_embeddings, fact_embs[1], r_ids),
                                     name='predict_h_score')
            tail_score = tf.identity(self._cal_triplet_score(fact_embs[0], all_entity_embeddings, r_ids),
                                     name='predict_t_score')
            tf.add_to_collection(tools.LinkPredictionScoreKey, head_score)
            tf.add_to_collection(tools.LinkPredictionScoreKey, tail_score)
        else:
            head_score = None
            tail_score = None
        r_score = tf.identity(self._cal_decoder_score(fact_embs[0], fact_embs[1]), name='predict_r_score')
        best_fit_r = tf.argmax(r_score, axis=1, name='best_fit_r')
        tf.add_to_collection(tools.LinkPredictionScoreKey, r_score)
        # output and hand to numpy calculate MRR Hits and MR
        # calculate rank
        return head_score, tail_score, r_score

    def _get_embedding_name(self, e_b, p_n, f_s, ):
        indexes = [e_b, p_n, f_s]
        return "-".join([self.embedding_names[i][idx] for i, idx in enumerate(indexes)])

    @staticmethod
    def normalize_tensor(t, axis, norm_ord='euclidean', name=None, summary_norm=True):
        norm = tf.norm(t, ord=norm_ord if norm_ord is not 'N' else 'euclidean',
                       axis=axis, name=t.op.name + "_Norm_{}".format(norm_ord if norm_ord is not 'N' else 'E'
                                                                     ), keepdims=True)
        if summary_norm:
            tf.add_to_collection(tools.HistogramSummaryTensors, norm)
        if norm_ord is "N":
            return t
        name = t.op.name + "_normalize" if name is None else name
        epsilon = 1e-12
        t = tf.divide(t, tf.maximum(norm, epsilon), name)
        return t


class Bilinear(Decoder):
    def __init__(self, opts: arguments.LearningArguments, entity_number, relations_number=-1, emb_size=0, data=None):
        super().__init__(opts, entity_number, relations_number, emb_size, data)

    def _create_relation_mats(self, ):
        super()._create_relation_mats()
        opts = self.options
        C = tools.get_or_create_variable(BILINEAR_RELATION_MATRIX_NAME,
                                         dtype=tf.float32,
                                         regularizer=(tools.regularizer_l1_l2(opts.l1_regularization,
                                                                              opts.l2_regularization
                                                                              ) if opts.extended_reg else None),
                                         initializer=tf.random_normal_initializer(stddev=np.sqrt(0.1)),
                                         shape=(
                                             self.relations_number, self.emb_size,
                                             self.emb_size,),
                                         collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                      tf.GraphKeys.MODEL_VARIABLES,
                                                      tf.GraphKeys.WEIGHTS,
                                                      tools.RelationKey],
                                         )
        drnn = opts.decoder_relation_norm_normalize
        C = self.normalize_tensor(C, [-2, -1], tools.norm_dict[drnn], )
        self.C = C

    def _create_weighted_relation_mats(self, r_probs=None):
        super()._create_weighted_relation_mats(r_probs)
        if not hasattr(self, 'weighted_C'):
            self.weighted_C = tf.einsum('br,rjk->bjk', r_probs, self.C, name='weighted_C')

    def _cal_decoder_score(self, head_embs, tail_embs, relation_probs=None):
        """

        :param head_embs: (b,m,k)
        :param tail_embs: (b,n,k)
        :return: (b, r, n)
        """
        sup_score = super()._cal_decoder_score(head_embs, tail_embs, relation_probs)
        # (r, k, k)
        if relation_probs is not None:
            self._create_weighted_relation_mats(relation_probs)
            C = self.weighted_C
            rd1s = 'b'
            rd2s = ''
        else:
            C = self.C
            rd1s = rd2s = 'r'

        score = tf.einsum('bmj,{}jk,bnk->b{}mn'.format(rd1s, rd2s), head_embs, C, tail_embs, )

        if score.shape[-2] == tf.Dimension(1):
            score = tf.squeeze(score, axis=-2)
        else:
            score = tf.squeeze(score, axis=-1)
        score += sup_score
        return score

    def _cal_triplet_score(self, head_embs, tail_embs, r_ids):
        """

        :param head_embs: (a, m, k) b, 1 | b, 1
        :param tail_embs: (c, n, k) 1, n | b, n
        :param r_ids: (b)
        :return: (b, m)
        """
        C = self.C
        # b, k, k
        rl_embs = tf.nn.embedding_lookup(C, r_ids)
        sup_score = super()._cal_triplet_score(head_embs, tail_embs, r_ids)

        # This function behaves like numpy.einsum, but does not support:
        #     Ellipses (subscripts like ij...,jk...->ik...)
        # https://www.tensorflow.org/api_docs/python/tf/einsum
        if len(head_embs.shape) == 2:
            # import pdb; pdb.set_trace()
            # predict head
            score = tf.einsum('mj,bjk,bnk->bm', head_embs, rl_embs, tail_embs)
            # score = tf.squeeze(score, axis=2)
        elif len(tail_embs.shape) == 2:
            # predict tail
            score = tf.einsum('bmj,bjk,nk->bn', head_embs, rl_embs, tail_embs)
            # score = tf.squeeze(score, axis=1)
        else:
            # training
            score = tf.einsum('bmj,bjk,bnk->bmn', head_embs, rl_embs, tail_embs, )
            # # https://www.tensorflow.org/api_docs/python/tf/Dimension
            if score.shape[1] == tf.Dimension(1):
                score = tf.squeeze(score, axis=[1])
            else:
                score = tf.squeeze(score, axis=[2])
        return score + sup_score

    def regularize_by_id(self, oie_decoder, lookup_ids, relation_probabilities):
        oie_C = oie_decoder.C
        score = super().regularize_by_id(oie_decoder, lookup_ids, relation_probabilities)

        lookupC = tf.nn.embedding_lookup(self.C, lookup_ids, name='extkb_C')
        weighted_C = tf.einsum('br,rkj->bkj', relation_probabilities, oie_C)
        score += self._regularize_by_id_score(weighted_C - lookupC, 'bilinear_decoder_res')
        return score


class SPDecoder(Decoder):
    def __init__(self, opts, entity_number, relations_number=-1, emb_size=0, data=None):
        super().__init__(opts, entity_number, relations_number, emb_size, data)

    def _create_relation_mats(self):
        opts = self.options
        super()._create_relation_mats()
        rel_vec = [tools.get_or_create_variable("{}_{}".format(SP_RELATION_EMBEDDING_NAME, i),
                                                initializer=tf.random_normal_initializer(stddev=np.sqrt(0.1)),
                                                shape=(self.relations_number, self.emb_size),
                                                collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                             tf.GraphKeys.MODEL_VARIABLES,
                                                             tf.GraphKeys.WEIGHTS,
                                                             tools.RelationKey]
                                                ) for i in (0, 1)
                   ]
        drnn = opts.decoder_relation_norm_normalize
        for i, t in enumerate(rel_vec):
            t = self.normalize_tensor(t, axis=-1, norm_ord=tools.norm_dict[drnn])
            if 'A' in drnn:
                rel_vec[i] = t
        self.rel_vec = rel_vec

    def _create_weighted_relation_mats(self, r_probs):
        super()._create_weighted_relation_mats(r_probs)
        if not hasattr(self, 'weighted_rel_vec'):
            self.weighted_rel_vec = [tf.matmul(r_probs, self.rel_vec[i],
                                               name="weighted_relation_embedding_{}".format(i)
                                               ) for i in (0, 1)]

    def _cal_decoder_score(self, head_embs, tail_embs, relation_probs=None):
        # [(r, k),] *2
        score = super()._cal_decoder_score(head_embs, tail_embs, relation_probs)
        if relation_probs is None:
            rd1s = rd2s = 'r'
            rel_vec = self.rel_vec
        else:
            rd1s, rd2s = 'b', ''
            self._create_weighted_relation_mats(relation_probs)
            rel_vec = self.weighted_rel_vec
        en_vec = [head_embs, tail_embs]
        sp_score = [tf.einsum("bmk,{}k->b{}m".format(rd1s, rd2s), en_vec[i], rel_vec[i]) for i in (0, 1)]
        # (b, r, m)
        sp_score = tf.add(*sp_score, name="sp_score_batch")

        return sp_score + score

    def _cal_triplet_score(self, head_embs, tail_embs, r_ids):
        """

        :param head_embs: (a, m, k)
        :param tail_embs:(c, n, k)
        :param rl_embs: [(b, k), (b, k)] b
        :return:(b, m)
        """
        sup_score = super()._cal_triplet_score(head_embs, tail_embs, r_ids)
        en_embs = [head_embs, tail_embs]
        rel_vec = self.rel_vec
        rl_embs = [tf.nn.embedding_lookup(rel_vec[i], r_ids) for i in range(2)]

        def cal(e, r):
            if len(e.shape) == 2:
                score = tf.einsum('mk,bk->bm', e, r)
            else:
                score = tf.einsum('bmk,bk->bm', e, r)
            return score

        sp_score = [cal(en_embs[i], rl_embs[i]) for i in (0, 1)]
        # abm, cbn  | bb1, 1be
        sp_score = tf.add(*sp_score, name="sp_triplet_score_batch")
        return sp_score + sup_score

    def regularize_by_id(self, oie_decoder, lookup_ids, relation_probabilities):
        score = super().regularize_by_id(oie_decoder, lookup_ids, relation_probabilities)
        oie_vec = oie_decoder.rel_vec

        for i in (0, 1):
            weighted_rel_emb = tf.einsum('br,rk->bk', relation_probabilities, oie_vec[i])
            lookup = tf.nn.embedding_lookup(self.rel_vec[i], lookup_ids)
            score += self._regularize_by_id_score(weighted_rel_emb - lookup, 'sp_decoder_res_{}'.format(i))
        return score


class TransE(Decoder):
    """
    pTransE in http://emnlp2014.org/papers/pdf/EMNLP2014167.pdf
    """

    def __init__(self, opts: arguments.LearningArguments, entity_number, relations_number=-1, emb_size=0, data=None):
        super().__init__(opts, entity_number, relations_number, emb_size)

    def _create_relation_mats(self):
        opts = self.options
        super()._create_relation_mats()
        rel_vec = tools.get_or_create_variable(TransE_RELATION_EMBEDDING_NAME,
                                               initializer=tf.random_normal_initializer(stddev=np.sqrt(0.1)),
                                               shape=(self.relations_number, self.emb_size),
                                               collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                            tf.GraphKeys.MODEL_VARIABLES,
                                                            tf.GraphKeys.WEIGHTS,
                                                            tools.RelationKey])
        drnn = opts.decoder_relation_norm_normalize
        t = self.normalize_tensor(rel_vec, axis=-1, norm_ord=tools.norm_dict[drnn])
        if 'A' in drnn:
            rel_vec = t
        self.rel_transe_vec = rel_vec

    def _create_weighted_relation_mats(self, r_probs):
        super()._create_weighted_relation_mats(r_probs)
        if not hasattr(self, 'weighted_rel_transe_vec'):
            self.weighted_rel_transe_vec = tf.matmul(r_probs, self.rel_transe_vec)

    def _cal_decoder_score(self, head_embs, tail_embs, relation_probs=None):
        score = super()._cal_decoder_score(head_embs, tail_embs, relation_probs)

        en_vec = [head_embs, tail_embs]
        en_embs = []
        if relation_probs is None:
            rl_embs = self.rel_transe_vec
            # 1, r, 1, k
            rl_embs = tf.reshape(rl_embs, (1, self.relations_number, 1, self.emb_size))
            # b, m, k -> b,1, m, k
            for v in en_vec:
                en_embs.append(
                    tf.expand_dims(v, axis=1)
                )
        else:
            self._create_weighted_relation_mats(relation_probs)
            rl_embs = self.weighted_rel_transe_vec
            # b, 1, k
            rl_embs = tf.reshape(rl_embs, (-1, 1, self.emb_size))
            # b, m, k
            en_embs = en_vec
        # (1, r, 1, k) + (b, 1, m, k) - (b,1, n, k) -> (b,r , m/n, k)
        # or (b, 1, k) + (b, m, k) -(b, n, k)
        transe_score = en_embs[0] - en_embs[1] + rl_embs
        transe_score = 7 - tf.reduce_sum(transe_score ** 2, axis=-1, name='TransE_score')

        # b, r, m or b, m
        return transe_score + score

    def _cal_triplet_score(self, head_embs, tail_embs, r_ids):
        """

        :param head_embs: (a, m, k)
        :param tail_embs:(c, n, k)
        :param rl_embs: [(b, k), (b, k)] b
        :return:(b, m)
        """
        sup_score = super()._cal_triplet_score(head_embs, tail_embs, r_ids)
        en_embs = []
        for v in [head_embs, tail_embs]:
            if len(v.shape) == 2:
                v = tf.expand_dims(v, 0)
            en_embs.append(v)
        rel_vec = self.rel_transe_vec
        rl_embs = tf.expand_dims(tf.nn.embedding_lookup(rel_vec, r_ids), 1)

        transe_score = en_embs[0] - en_embs[1] + rl_embs
        transe_score = 7 - tf.reduce_sum(transe_score ** 2, axis=-1, name='TransE_triplet_score')
        return transe_score + sup_score

    def regularize_by_id(self, oie_decoder, lookup_ids, relation_probabilities):
        score = super().regularize_by_id(oie_decoder, lookup_ids, relation_probabilities)
        oie_vec = oie_decoder.rel_transe_vec

        weighted_rel_emb = tf.einsum('br,rk->bk', relation_probabilities, oie_vec)
        lookup = tf.nn.embedding_lookup(self.rel_transe_vec, lookup_ids)
        score += self._regularize_by_id_score(weighted_rel_emb - lookup, 'transe_decoder_res')
        return score


class BilinearPlusSP(SPDecoder, Bilinear):
    def __init__(self, opts, entity_number, relations_number=-1, emb_size=0, data=None):
        super().__init__(opts, entity_number, relations_number, emb_size, data)

    def _create_relation_mats(self):
        super()._create_relation_mats()

    def _cal_decoder_score(self, head_embs, tail_embs, relation_probs=None):
        return super()._cal_decoder_score(head_embs, tail_embs, relation_probs)

    def _cal_triplet_score(self, head_embs, tail_embs, rl_ids):
        sup_score = super()._cal_triplet_score(head_embs, tail_embs, rl_ids)

        return sup_score

    def regularize_by_id(self, oie_decoder, lookup_ids, relation_probabilities):
        return super().regularize_by_id(oie_decoder, lookup_ids, relation_probabilities)


CorrectDecoderDict = {"A": Bilinear,
                      "C": SPDecoder,
                      "AC": BilinearPlusSP,
                      "T": TransE,
                      }
