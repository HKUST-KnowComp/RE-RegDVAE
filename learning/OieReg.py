import logging
import os
import numpy as np
import tensorflow as tf

import learning
from definitions import tools, log_dict, arguments
from definitions.arguments import LearningArguments
from learning.OieInduction import OieDatasetManager, ReconstructInducer, get_optimizer, update_feed_dict
from learning.Regularizer import RegularizerDict
from definitions import tools


class OieRegularizedReconstructInducer(ReconstructInducer):
    def __init__(self, options: LearningArguments, data: OieDatasetManager):
        super().__init__(options, data)

    def _get_train_op(self, loss):
        opts = self.options
        # --------- Loss ------------------------------

        global_step = tf.train.get_or_create_global_step(self.graph)
        # --------- Summaries ------------------------------
        # ------- Optimization ---------------------------------
        optimizer = get_optimizer(opts.learning_rate, opts.optimization)
        # --------- Train OPs ------------------------------
        self.train_op = self.train_oie_reg_op = optimizer.minimize(loss, global_step=global_step,
                                                                   name="train_oie_reg_op")

    def forward(self):
        super().forward()

        opts = self.options
        # --------- Regularizer Score ------------------------------
        if opts.reg_model != 'None':
            with tf.variable_scope("regularizer"):
                beta = tf.get_variable('beta', initializer=opts.beta_init,
                                       dtype=tf.float32,
                                       collections=[tools.AddToScalarSummaryKey,
                                                    tf.GraphKeys.GLOBAL_VARIABLES],
                                       trainable=False)
                p_r = self.decoder.relation_probs if opts.reg_model == 'TD' else self.relation_probabilities
                regularizer, regularizer_score = self._get_regularizer_score(opts.reg_model, p_r,
                                                                             self.args_input[0])
            self.regularizer_score = regularizer_score
            self.placeholders += self.regularizer.get_input_placeholders()
            tf.add_to_collection(tools.LossKey, tf.multiply(-2 * beta, regularizer_score, name="weighted_regulazation"))

    def _get_regularizer(self, model_type, opts=None):
        opts = self.options if not opts else opts
        regularizer = RegularizerDict[model_type](self.options, self.data, self.input_placeholders)
        return regularizer

    def _get_regularizer_score(self, model_type, relation_probabilities, pos_input):
        regularizer = self._get_regularizer(model_type, )
        regularizer_score = regularizer.get_score(relation_probabilities)

        self.regularizer = regularizer
        self.regularizer_score = regularizer_score
        tf.add_to_collection(tools.AddToScalarSummaryKey, regularizer_score)
        return regularizer, regularizer_score

    def _get_loss(self):
        opts = self.options
        # --------- loss ------------------------------
        mean = opts.batch_size

        loss_without_decay = tf.multiply(-1 / mean, tf.add_n(tf.get_collection(tools.LossKey)),
                                         name="loss_without_decay")
        # --------- Regularization ------------------------------
        # adjust
        reg_list = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if reg_list:
            regularization = tf.add_n(reg_list,
                                      name="weight_decay")
            tf.add_to_collection(tools.AddToScalarSummaryKey, regularization)
        else:
            regularization = tf.constant(0., name='zero_weight_decay')

        loss = tf.add(loss_without_decay, regularization, name="oie_reg_loss")
        tf.add_to_collection(tools.AddToScalarSummaryKey, loss)
        return loss, regularization, loss_without_decay

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
        total_loss_summary = g.get_tensor_by_name("loss_summary/oie_reg_loss:0")

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
        if not hasattr(opts, 'beta_change') or opts.beta_change == 'None':
            beta_change = False
        else:
            beta_change = True
            with tf.variable_scope('regularizer', reuse=True):
                beta_tensor = tf.get_variable('beta')
            beta_change_func = self.get_weight_change_func(opts.beta_init, opts.beta_final,
                                                           opts.beta_change,
                                                           opts.epochs * len(data_iter))
        for data_batch in data_iter:
            feed_dict = update_feed_dict(placeholders, data_batch)
            if alpha_change:
                alpha = alpha_change_func(data_iter.current_idx)
                feed_dict[alpha_tensor] = max(alpha, 0)
            if beta_change:
                beta = beta_change_func(data_iter.current_idx)
                feed_dict[beta_tensor] = max(beta, 0)
            summary_op = all_train_summary if (step_value + 1) % skip_step == 0 else total_loss_summary

            _, loss_value, step_value, summary = sess.run([train_op, loss_tensor, global_step, summary_op, ],
                                                          feed_dict=feed_dict)
            # add summary
            writer.add_summary(summary, global_step=step_value)
            total_loss += loss_value

            # if (step_value+1) % SKIP_STEP == 0:
            if step_value >= eval_step_interval and step_value % eval_step_interval == 0:
                self.logger.debug("Epoch:{} Evaluation on Validation Set:".format(self.current_epoch))
                # self.evaluate_model(sess, self.data.valid_iter)
                try:
                    self.summary_evaluation(sess, "valid", writer)
                    self.summary_evaluation(sess, "test", writer)
                except ValueError:
                    self.logger.warning('Early stop')
                    break
            if self.current_epoch != data_iter.epoch:
                self.logger.info(log_dict.EvalLog(step_value, self.current_epoch, 'train', 'loss', total_loss))
                # saver.save(sess, checkpoint_dir, step_value)
                self.current_epoch = data_iter.epoch
                total_loss = 0.
        self.logger.debug("Training Done")


def run(opts: LearningArguments):
    # logger
    logger = logging.getLogger(__name__)
    logger.debug("Open IE Model with regularization")
    # --------- data -----------
    Inducer = OieRegularizedReconstructInducer
    indexed_data = OieDatasetManager(opts)
    learning.run(Inducer, opts, indexed_data)



