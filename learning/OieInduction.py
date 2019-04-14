import logging
import os
import sys

import numpy as np
import tensorflow as tf

import learning
from definitions import tools, log_dict, arguments
from definitions.arguments import LearningArguments
from definitions.log_dict import ArgLog
from learning import Encoders, DataFeeder
from learning.Decoders import CorrectDecoderDict
from learning.Encoders import InputPlaceholders
from learning.Evaluation import ClusterEvaluator

eval_batch_size = int(7 * 2 ** 30 / (122051 * 32) // 100) * 100
EvalPHKey = "EvalPlaceholders"
FeatInputKey = "FeatureInputPHs"
SKIP_STEP = 1000


def get_optimizer(learning_rate, optimizer_type, correct_version=True):
    # opts = self.options
    learning_rate = tf.Variable(learning_rate, trainable=False, name="lr")
    if optimizer_type == 0:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    if optimizer_type == 1:
        init_acc_val = 0.1 if correct_version else 1e-6
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=init_acc_val)
    else:
        optimizer = tools.optimizer_list[optimizer_type](learning_rate)
    # self.optimizer = optimizer
    return optimizer


class OieDatasetManager(DataFeeder.DataSetManager):
    def __init__(self, opts):
        self.kb_entity_embedding = None
        super().__init__(opts)

    def _load_data(self, ):
        opts = self.options
        super()._load_data()
        input_dir = opts.input_dir
        text_dir = os.path.join(input_dir, 'text')

        self.text_triplets = DataFeeder.TripletsFeeder.load_data(os.path.join(text_dir, 'dataset.npz'))
        if hasattr(opts, 'reg_model') and opts.reg_model != 'None':
            translator = self.get_extkb_parameters('translator')
            self.triplets_feeder = DataFeeder.ArgsTranslateFeeder(self.text_triplets, translator)
        else:
            self.triplets_feeder = DataFeeder.TripletsFeeder(self.text_triplets)
        feeders = [self.triplets_feeder,
                   DataFeeder.NegSampleFeeder(self.text_entity_lex,
                                              opts.negative_samples_number,
                                              opts.negative_samples_power), ]
        feeders += self._encoder_data_loader(opts.encoder, text_dir, )
        self.data_feeders += feeders

    def get_kb_entity_embeddings(self):
        opts = self.options
        if not hasattr(opts, 'load_ext_kb_dir'):
            return None
        if self.kb_entity_embedding is None:
            extkb_dir = os.path.join(opts.input_dir, 'kb', 'model', opts.load_ext_kb_dir)
            extkb_dir = tools.complete_dir(extkb_dir)
            if os.path.exists(os.path.join(extkb_dir, 'entity2vec.npy')):
                self.kb_entity_embedding = np.load(os.path.join(extkb_dir, 'entity2vec.npy'))
            else:
                with open(os.path.join(extkb_dir, 'entity2vec.vec'), 'r') as f:
                    self.transE_entity_vec = f.readlines()
                self.kb_entity_embedding = np.array(list(
                    map(lambda v: np.array(v.split(), dtype=np.float32),
                        self.transE_entity_vec)))
        return self.kb_entity_embedding


class ReconstructInducer:
    @tools.print_func_name
    def __init__(self, options: LearningArguments, data: OieDatasetManager
                 ):
        """
            :param options:            LearningArguments
            :param data:               DataSetManager - indexedData
            :param rand:               RandomState - random number generator
        """
        self.options = options
        self.data = data
        self.text_stats = data.text_stats
        self.logger = logging.getLogger(__name__)
        self.current_epoch = 0
        self.encoder = None

        self.input_placeholders = InputPlaceholders()
        self.graph = tf.get_default_graph()  # tensorflow.python.framework.ops.Graph
        self.build_graph()
        self.eval_epoch_interval = 1

    @tools.print_func_name
    def build_graph(self):
        self.forward()
        loss, regularization, loss_without_decay = self._get_loss()
        self.loss = loss
        self._model_summary()
        self.placeholders = list(set(self.placeholders))
        self._get_train_op(loss)

    def forward(self):
        opts = self.options
        entity_num = self.text_stats['entity_num']

        # ----------- Data Input----------------------------
        #  [[args1, args2], [neg1, neg2]]
        self.args_input = self._get_data_input()
        # --------- Relation Classifier------------------------------
        with tf.variable_scope("encoder", reuse=False,
                               regularizer=tools.regularizer_l1_l2(opts.l1_regularization,
                                                                   opts.l2_regularization),
                               initializer=tf.contrib.layers.xavier_initializer()
                               ):
            self.encoder = self._get_encoder()
            self.encoder_input = self.encoder.get_input_placeholders()
            self.relation_probabilities = relation_probabilities = self._get_encoder_score()
        self.placeholders += self.encoder_input

        # --------- Decoder Score ------------------------------
        with tf.variable_scope("decoder", reuse=False,
                               regularizer=tools.regularizer_l1_l2(opts.l1_regularization,
                                                                   opts.l2_regularization) if opts.extended_reg else None,
                               ):
            decoder, decoder_score = self._get_decoder_score(opts.model, relation_probabilities, self.args_input[0],
                                                             self.args_input[1], entity_num, opts.relations_number, )
        tf.add_to_collection(tools.LossKey, decoder_score)

        # --------- Entropy Score ------------------------------
        with tf.variable_scope("entropy"):
            alpha = tf.get_variable('alpha', initializer=opts.alpha_init,
                                    dtype=tf.float32,
                                    collections=[tools.AddToScalarSummaryKey,
                                                 tf.GraphKeys.GLOBAL_VARIABLES],
                                    trainable=False)
            entropy_sum = tools.cal_entropy(relation_probabilities)
        self.entropy = entropy_sum
        tf.add_to_collection(tools.LossKey, tf.multiply(2 * alpha, entropy_sum, name="weighted_entropy"))

    def _get_train_op(self, loss):
        opts = self.options
        # --------- Loss ------------------------------

        global_step = tf.train.get_or_create_global_step(self.graph)
        # --------- Summaries ------------------------------
        # ------- Optimization ---------------------------------
        optimizer = get_optimizer(opts.learning_rate, opts.optimization)
        # --------- Train OPs ------------------------------
        self.train_op = self.train_oie_op = optimizer.minimize(loss, global_step=global_step,
                                                               name="train_oie_op")

    def _get_data_input(self):
        # [batch, feat_dim]
        # encoder_input = self.encoder.get_input_placeholders()
        # [batchsize, 1]
        placeholder = self.input_placeholders.get
        args1 = self.input_placeholders.get(name='args1', dtype=tf.int32, shape=(None, 1))
        args2 = self.input_placeholders.get(dtype=tf.int32, shape=(None, 1), name="args2")

        neg1 = placeholder(dtype=tf.int32, shape=(None, None), name="neg1")
        neg2 = placeholder(dtype=tf.int32, shape=(None, None), name="neg2")
        args_input = [[args1, args2], [neg1, neg2]]

        self.placeholders = [args1, args2, neg1, neg2]
        return args_input

    def _get_encoder_score(self, ):
        """
        output relation probabilities
        """

        encoder = self._get_encoder()
        relation_probabilities = encoder.build_relation_probabilities()
        relation_predictions = tf.argmax(relation_probabilities, axis=-1, name='relation_predictions')
        self.relations_probabilities = relation_probabilities
        return relation_probabilities

    def _get_encoder(self):
        if self.encoder is None:
            self.encoder = Encoders.AllEncoder(self.options, self.data, self.input_placeholders)
        return self.encoder

    def _get_decoder(self, model_type, entity_num, relation_num, emb_size=0, opts=None):
        opts = self.options if not opts else opts
        model_type = model_type.split('-')[0]
        decoder = CorrectDecoderDict[model_type](self.options, entity_num,
                                                 relation_num, opts.embed_size if emb_size == 0 else emb_size,
                                                 self.data)
        #  [[args1, args2], [neg1, neg2]]
        return decoder

    def _get_decoder_score(self, model_type, relation_probabilities, pos_input, neg_input, entity_num, relation_num):
        opts = self.options
        original_decoder_score = opts.original_decoder_score if hasattr(opts, 'original_decoder_score') else False
        decoder = self._get_decoder(model_type, entity_num, relation_num, )
        #  [[args1, args2], [neg1, neg2]]
        decoder_score = decoder.get_score(pos_input, neg_input, relation_probabilities, original_decoder_score)

        self.decoder = decoder
        self.decoder_score = decoder_score
        tf.add_to_collection(tools.AddToScalarSummaryKey, decoder_score)
        return decoder, decoder_score

    def _get_loss(self):
        opts = self.options
        # --------- loss ------------------------------
        mean = opts.batch_size

        loss_without_decay = tf.multiply(-1 / mean, tf.add_n(tf.get_collection(tools.LossKey)),
                                         name="loss_without_decay")
        # --------- Regularization ------------------------------
        # adjust
        # regularization = self.graph.get_tensor_by_name("weight_decay:0")
        reg_list = [x for x in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) if 'rnn' not in x.name]
        if reg_list:
            regularization = tf.add_n(reg_list,
                                      name="weight_decay")
            tf.add_to_collection(tools.AddToScalarSummaryKey, regularization)
        else:
            regularization = tf.constant(0., name='zero_weight_decay')

        loss = tf.add(loss_without_decay, regularization, name="oie_loss")
        tf.add_to_collection(tools.AddToScalarSummaryKey, loss)
        return loss, regularization, loss_without_decay

    def _model_summary(self, loss_list=[]):
        # --------- Summaries ------------------------------
        g = self.graph
        loss_list = tf.get_collection(tools.LossKey) + loss_list + tf.get_collection(tools.AddToScalarSummaryKey)
        with tf.name_scope("loss_summary"):
            tools.scalar_summary(loss_list)
        tools.variable_summary()

        cluster_size_ph = tf.placeholder(dtype=tf.int32, name="cluster_size_ph")
        for key in ["train", "valid", "test", 'valid-decoder', 'test-decoder']:
            collections = [tf.GraphKeys.SUMMARIES]
            if key is not "train":
                collections += [tools.summary_key[key]]
            tf.summary.histogram("Evaluation/{}/cluster_size".format(key), cluster_size_ph,
                                 collections=collections)

    def load_model(self, sess, saver, step=None):
        # ----------- Saver and Restore  ----------------------------
        opts = self.options
        g = self.graph
        checkpoint_dir = os.path.join(opts.output_dir, "checkpoints/checkpoint")
        tools.make_sure_path_exists(os.path.dirname(checkpoint_dir))
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_dir))

        if ckpt and ckpt.model_checkpoint_path:
            if step is None:
                restore_path = ckpt.model_checkpoint_path
            else:
                restore_path = os.path.exists(checkpoint_dir + "-{:d}".format(step))
                assert os.path.exists(restore_path), "ckpt {:} not exists!".format(restore_path)

            saver.restore(sess, restore_path)
            self.logger.info('Load model from [{}] successfully'.format(restore_path))

    @tools.profile
    def train_model(self, sess: tf.Session, saver, writer):
        opts = self.options
        g = self.graph  # tensorflow.python.framework.ops.Graph
        checkpoint_dir = os.path.join(opts.output_dir, "checkpoints/checkpoint")
        # ----------- Restore Graph Ops ----------------------------
        train_op = self.train_op
        loss_tensor = self.loss
        global_step = tf.train.get_global_step(g)
        # Merge training summaries
        all_train_summary = tf.summary.merge_all(tools.TrainSummaries)
        total_loss_summary = g.get_tensor_by_name("loss_summary/oie_loss:0")

        total_loss = 0.
        step_value = 0
        data_iter = self.data.train_iter
        placeholders = self.placeholders

        skip_step = max(int(len(data_iter) / opts.batch_size / 10), 1)

        eval_step_interval = int(len(data_iter) * self.eval_epoch_interval) // opts.batch_size
        if not hasattr(opts, 'alpha_change') or opts.alpha_change == 'None':
            alpha_change = False
        else:
            alpha_change = True
            with tf.variable_scope('entropy', reuse=True):
                alpha_tensor = tf.get_variable('alpha')
            alpha_change_func = self.get_weight_change_func(opts.alpha_init, opts.alpha_final,
                                                            opts.alpha_change,
                                                            opts.epochs * len(data_iter))

        for data_batch in data_iter:
            feed_dict = update_feed_dict(placeholders, data_batch)
            if alpha_change:
                alpha = alpha_change_func(data_iter.current_idx)
                feed_dict[alpha_tensor] = max(alpha, 0)
            summary_op = all_train_summary if (step_value + 1) % skip_step == 0 else total_loss_summary

            _, loss_value, step_value, summary = sess.run([train_op, loss_tensor, global_step, summary_op, ],
                                                          feed_dict=feed_dict)
            # add summary
            writer.add_summary(summary, global_step=step_value)
            total_loss += loss_value

            # if (step_value+1) % SKIP_STEP == 0:
            if step_value >= eval_step_interval and step_value % eval_step_interval == 0:
                # self.evaluate_model(sess, self.data.valid_iter)
                ef = 1.
                ef = min(ef, self.summary_evaluation(sess, "valid", writer))
                ef = min(ef, self.summary_evaluation(sess, "test", writer), )
                if ef < 1e-9:
                    self.logger.warning('Early stop')
                    break

            if self.current_epoch != data_iter.epoch:
                self.logger.info(log_dict.EvalLog(step_value, self.current_epoch, 'train', 'loss', total_loss))
                # saver.save(sess, checkpoint_dir, step_value)
                self.current_epoch = data_iter.epoch
                total_loss = 0.
        self.logger.debug("Training Done")

    def eval_epoch(self, sess, data_iter, place_holders, eval_scores):
        score_batches = []
        assert isinstance(eval_scores, list), 'eval_scores should be iterable'
        for data_batch in data_iter:
            feed_dict = update_feed_dict(place_holders, data_batch)
            score_batch = sess.run(eval_scores,
                                   feed_dict=feed_dict)
            # feed score batch to kb evaluator
            score_batches.append(score_batch)
        score_list = []
        for i, s in enumerate(eval_scores):
            if len(s.shape) == 0:
                score_list.append(np.array([x[i] for x in score_batches]))
            else:
                score_list.append(np.concatenate([x[i] for x in score_batches], axis=0))
        return score_list

    def summary_evaluation(self, sess, data_key, writer):
        opts = self.options
        g = self.graph  # type:tf.Graph
        step_value = sess.run(tf.train.get_global_step(g))
        ev = None
        for key in [data_key, data_key + '-decoder']:
            rel_preds, total_entropy = self.cal_rel_probs(sess, key)
            # relation_labels = data_iter.dataset['relation'] if label_less else data_iter.dataset['relation'].astype(int)
            evaluator = ClusterEvaluator(rel_preds, total_entropy, self.data.get_relation_labels(key),
                                         step_value, self.current_epoch, key, opts.relations_number)
            evaluator.cal_all()
            summary_op = g.get_tensor_by_name("Evaluation/{}/cluster_size:0".format(key))
            cluster_size_ph = g.get_tensor_by_name("cluster_size_ph:0")
            histo_summary = sess.run(summary_op,
                                     feed_dict={cluster_size_ph: evaluator._cal_pred_cluster_size()})
            writer.add_summary(histo_summary, step_value)

            summary = tf.Summary(value=evaluator.tf_summary_value)
            writer.add_summary(summary, step_value)
            if key == data_key:
                ev = evaluator
        return ev.efficiency

    def cal_rel_probs(self, sess: tf.Session, data_key, ):
        opts = self.options
        g = self.graph
        data_iter = self.data[data_key]
        self.logger.debug('calculating {} probabilities'.format(data_key))

        if 'decoder' not in data_key.split('-'):
            with tf.variable_scope("encoder", reuse=True):
                rel_probs_tensor = g.get_tensor_by_name("encoder/relation_predictions:0")
                entropy_tensor = self.entropy
            placeholders = self.args_input[0] + list(self.encoder_input)
        else:
            rel_probs_tensor = self.decoder.predictions
            entropy_tensor = self.decoder.entropy
            placeholders = self.args_input[0]

        score_batches = self.eval_epoch(sess, data_iter, placeholders, [rel_probs_tensor, entropy_tensor])
        rel_preds = score_batches[0]
        total_entropy = np.sum(score_batches[1])

        # save predictions
        output_dir = os.path.join(opts.output_dir, 'predictions')
        tools.make_sure_path_exists(output_dir)
        step = tf.train.get_global_step(g).eval(session=sess)
        rel_labels = self.data.get_relation_labels(data_key)

        np.savez(os.path.join(output_dir, "%s_%d.npz" % (data_key, step)),
                 label = rel_labels,
                 pred = rel_preds
                 )
        return rel_preds, total_entropy

    def get_current_epoch_by_step(self, step=None, sess=None):
        if step is None:
            g = self.graph  # tensorflow.python.framework.ops.Graph
            step = tf.train.get_global_step(g).eval(session=sess)
        data_len = len(self.data['train'])
        spe = data_len // self.options.batch_size + 1

        return step // spe

    @staticmethod
    def get_weight_change_func(alpha_init, alpha_final, alpha_change, step_num):
        # todo summary alpha]
        epsilon = 1e-10
        logger = logging.getLogger(__name__)
        if alpha_change == 'None':
            logger.info(ArgLog('AlphaDecay', 'None'))
            return lambda x: alpha_init
        elif alpha_change.startswith('exp'):
            if alpha_final < 0:
                e = -np.abs(float(alpha_change[3:]))
            else:
                alpha_final = max(epsilon, alpha_final)
                e = np.log(alpha_final / max(alpha_init, epsilon)) / step_num
            logger.info(ArgLog('AlphaDecay', 'exp({:.5e})'.format(e)))
            return lambda x: alpha_init * np.exp(e * x)
        elif alpha_change.startswith('power'):
            if alpha_final < 0:
                e = -np.abs(float(alpha_change[5:]))
            else:
                alpha_final = max(epsilon, alpha_final)
                e = np.log(alpha_final / alpha_init) / np.log(step_num)
            logger.info(ArgLog('AlphaDecay', 'power({:.5e})'.format(e)))
            return lambda x: alpha_init * x ** e
        elif alpha_change.startswith('linear'):
            if alpha_final < 0:
                e = -np.abs(float(alpha_change[6:]))
            else:
                e = (alpha_final - alpha_init) / step_num
            logger.info(ArgLog('AlphaDecay', 'linear({:.5e})'.format(e)))
            return lambda x: alpha_init + e * x


def update_feed_dict(place_holders, data, feed_dict=None):
    if feed_dict is None:
        feed_dict = {}

    for ph in place_holders:
        name = ph.name.rsplit('/', 1)[-1].split(':')[0]
        if name in data:
            feed_dict[ph] = data[name]
    return feed_dict


def run(opts: LearningArguments):
    # logger
    logger = logging.getLogger(__name__)
    logger.debug("Open IE Model")
    # --------- data -----------
    Inducer = ReconstructInducer
    indexed_data = OieDatasetManager(opts)

    learning.run(Inducer, opts, indexed_data)


if __name__ == "__main__":
    opts = arguments.oie_parser.parse_args(sys.argv[1:])
    learning.learner(opts)
