import os
import logging
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import tensorflow as tf

import learning
from definitions import tools, settings
from definitions.arguments import LearningArguments
from definitions.log_dict import ArgLog
from learning import DataFeeder
from learning import Evaluation, OieInduction
from learning.Decoders import CorrectDecoderDict
from processing.OiePreprocessor import FeatureLexiconCompact


class KBDatasetManager(DataFeeder.DataSetManager):
    def __init__(self, opts):
        super().__init__(opts)

    def _load_data(self):
        opts = self.options
        super()._load_data()
        self.eval_batch_size = 1
        kb_dir = os.path.join(opts.input_dir, 'kb')
        self.logger.info(ArgLog('load_kb_dir', kb_dir))
        self.text_stats = None
        opts = self.options
        input_dir = opts.input_dir
        text_dir = os.path.join(input_dir, 'text')
        kb_dir = os.path.join(input_dir, 'kb')

        self.kb_entity_lex = FeatureLexiconCompact.load_txt(os.path.join(kb_dir, 'entity_lexicon.txt'))
        self.kb_relation_lex = FeatureLexiconCompact.load_txt(os.path.join(kb_dir, 'relation_lexicon.txt'))
        self.kb_triplets = DataFeeder.TripletsFeeder.load_data(os.path.join(kb_dir, 'dataset.npz'))
        if hasattr(opts, 'inversion') and opts.inversion:
            train_triplets = self.kb_triplets['train']
            rel_num = self.kb_stats[settings.r_num_statkey]
            inv_kb = train_triplets.copy()
            inv_kb[:, 1] = train_triplets[:, 2]
            inv_kb[:, 2] = train_triplets[:, 1]
            inv_kb[:, 3] += rel_num
            train_triplets = np.concatenate((train_triplets, inv_kb), axis=0)
            self.kb_triplets['train'] = train_triplets
            self.kb_stats[settings.r_num_statkey] *= 2

        self.triplets_feeder = DataFeeder.TripletsFeeder(self.kb_triplets)
        feeders = [self.triplets_feeder,
                   DataFeeder.NegSampleFeeder(self.kb_entity_lex, opts.negative_samples_number,
                                              opts.negative_samples_power)]
        self.data_feeders += feeders


class DecoderTrainer(OieInduction.ReconstructInducer):
    def __init__(self, options: LearningArguments, data: KBDatasetManager):
        # get stats before init
        self.data = data
        self.kb_stats = self.data.kb_stats
        super().__init__(options, data)
        self.evaluator = Evaluation.LinkPredictionEvaluator(data)
        self.eval_epoch_interval = 5

    def forward(self):
        opts = self.options
        g = self.graph
        global_step = tf.train.get_or_create_global_step(g)

        self.args_input, self.relation_inputs = self._get_data_input()

        entity_num = self.kb_stats[settings.e_num_statkey]
        relation_num = self.kb_stats[settings.r_num_statkey]

        with tf.variable_scope("ext_kb"):
            self.decoder, decoder_score = self._get_decoder_score(opts.model,
                                                                  self.relation_inputs,
                                                                  self.args_input[0],
                                                                  self.args_input[1],
                                                                  entity_num,
                                                                  relation_num)
            self.decoder._link_prediction_graph(self.args_input[0], self.relation_inputs)
            decoder_score = tf.identity(decoder_score, name="decoder_score")
        tf.add_to_collection(tools.LossKey, decoder_score)

    def _get_data_input(self):

        args1 = self.input_placeholders.get(dtype=tf.int32, shape=(None, 1), name="args1")
        args2 = self.input_placeholders.get(dtype=tf.int32, shape=(None, 1), name="args2")

        neg1 = self.input_placeholders.get(dtype=tf.int32, shape=(None, None), name="neg1")
        neg2 = self.input_placeholders.get(dtype=tf.int32, shape=(None, None), name="neg2")
        args_input = [[args1, args2], [neg1, neg2]]
        ri = self.input_placeholders.get(dtype=tf.int32, shape=(None,), name="relation_input")
        self.placeholders = [args1, args2, neg1, neg2, ri]

        return args_input, ri

    def _get_decoder(self, model_type, entity_num, relation_num, emb_size=0, opts=None):
        opts = self.options
        decoder = CorrectDecoderDict[model_type](self.options, entity_num, relation_num, opts.kbemb_dim)
        #  [[args1, args2], [neg1, neg2]]
        return decoder

    def train_model(self, sess: tf.Session, saver, writer):
        super().train_model(sess, saver, writer)
        self.output_model(sess)

    def summary_evaluation(self, sess: tf.Session, data_key: str, writer):
        opts = self.options
        g = self.graph  # tensorflow.python.framework.ops.Graph

        data_iter = self.data[data_key]

        step = tf.train.get_global_step(g).eval(session=sess)
        current_epoch = self.get_current_epoch_by_step(step)
        # load if exists
        save_to_path = os.path.join(opts.output_dir, 'LP', 'rank-{}.pkl'.format(step))
        eval_scores = g.get_collection(tools.LinkPredictionScoreKey)
        if os.path.exists(save_to_path):
            with open(save_to_path, 'rb') as pklfile:
                ranks = pickle.load(pklfile)
            tf_summary_value, _ = self.evaluator.cal_result_by_rank(ranks, step, current_epoch, data_key)
        else:
            tools.make_sure_path_exists(os.path.dirname(save_to_path))
            place_holders = self.args_input[0] + [self.relation_inputs]
            score_batches = self.eval_epoch(sess, data_iter, place_holders, eval_scores)
            tf_summary_value, ranks = self.evaluator.log_results(score_batches, step, current_epoch, data_key)
        writer.add_summary(tf.Summary(value=tf_summary_value), step)
        # if data_key is 'test':
        with open(save_to_path, 'wb') as pklfile:
            pickle.dump(ranks, pklfile)
        return ranks

    def relation_annotation(self, sess: tf.Session, data_iter):
        opts = self.options
        g = self.graph

        with tf.variable_scope("ext_kb", reuse=True):
            best_fit_r = g.get_tensor_by_name("ext_kb/best_fit_r:0")
        eval_batch_size = int(7 * 2 ** 30 / (122051 * 32) // 100) * 100
        data_iter.batch_size = eval_batch_size
        rel_probs_list = []
        for data_batch in data_iter:
            # TODO: here should feed a batch of all relations
            arg1, arg2 = np.split(data_batch, 2, -1)
            feed_dict = {
                self.args_input[0][0]: arg1, self.args_input[0][1]: arg2,
            }

            rel_id = sess.run(best_fit_r, feed_dict=feed_dict)
            rel_probs_list.append(np.squeeze(rel_id))

        # ----------- np ----------------------------
        rel_predictions = np.concatenate(rel_id)

        return rel_predictions

    def output_model(self, sess):
        opts = self.options
        if opts.not_save_parameters:
            return

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ext_kb')
        output_dir = os.path.join(opts.output_dir, settings.save_parameters_file_name)
        logging.info('Saving kb parameters to {}'.format(output_dir))
        # tools.make_sure_path_exists(output_dir)
        vals = {}
        for tensor in var_list:
            name = tensor.name
            name = name.rsplit(':', 1)[0]
            value = sess.run(tensor)
            vals[name] = value
            self.logger.info('Tensor name: {}'.format(name))
        np.savez(output_dir, **vals)
        self.logger.info('Saved keys: {}'.format(list(vals.keys())))


def run(opts):
    Inducer = DecoderTrainer
    indexed_data = KBDatasetManager(opts)
    learning.run(Inducer, opts, indexed_data)

    dst_dir = os.path.join(opts.input_dir, 'kb', 'model')
    tools.soft_link_model_dir(opts.output_dir, dst_dir, True)


def output_model(out_opts):
    kb_dir = tools.complete_dir(out_opts.dir)
    output_model_parameters(kb_dir)


def output_model_parameters(kb_dir):
    logger = logging.getLogger()
    logger.info('output_dir'.format(kb_dir))
    opts = LearningArguments()  # kb opts

    opts.load(os.path.join(kb_dir, 'options_000_.json'))

    Inducer = DecoderTrainer
    indexed_data = KBDatasetManager(opts)

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    device_placement = '/cpu:0'

    g = tf.Graph()

    config = tf.ConfigProto(log_device_placement=False)
    config.allow_soft_placement = True  # [](https://github.com/tensorflow/tensorflow/issues/2292)
    config.gpu_options.allow_growth = True

    output_dir = os.path.join(opts.output_dir, settings.save_parameters_file_name)
    if os.path.exists(output_dir):
        logger.warning("{} Exists!".format(output_dir))
        return
    with g.as_default():
        with tf.device(device_placement):
            inducer = Inducer(opts, indexed_data)
            with tf.Session(config=config, ) as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()  # saving all variables
                inducer.load_model(sess, saver)

                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ext_kb')
                # tools.make_sure_path_exists(output_dir)
                vals = {}
                for tensor in var_list:
                    name = tensor.name
                    name = name.rsplit(':', 1)[0]
                    value = sess.run(tensor)
                    vals[name] = value
                    logger.info('Tensor name: {}'.format(name))
                np.savez(output_dir, **vals)
                logger.info('save to {}'.format(output_dir))

    logger.info("$End")
