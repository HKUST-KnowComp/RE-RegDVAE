import cProfile
import logging
import logging.config
import os
import pstats

import tensorflow as tf
from tensorflow.python import debug as tf_debug

from learning import OieInduction


def run(Inducer, opts, indexed_data):
    logger = logging.getLogger(__name__)
    logger.info('$LearningArguments ' + str(opts))
    save_profile = False

    if opts.gpus == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device_placement = '/cpu:0'
    else:
        device_placement = '/gpu:{}'.format(opts.gpus)
    g = tf.Graph()
    checkpoint_dir = os.path.join(opts.output_dir, "checkpoints/checkpoint")

    logger.info("device_placement: %s", device_placement)

    config = tf.ConfigProto(log_device_placement=False)
    config.allow_soft_placement = True  # [](https://github.com/tensorflow/tensorflow/issues/2292)
    config.gpu_options.allow_growth = True

    with g.as_default():
        with tf.device(device_placement):
            inducer = Inducer(opts, indexed_data)
            with tf.Session(config=config, ) as sess:
                feed_dict = inducer.input_placeholders.init_feed_dict
                sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)
                saver = tf.train.Saver()  # saving all variables
                inducer.load_model(sess, saver)

                summary_writer = tf.summary.FileWriter(opts.output_dir, g)
                opts_summary_placeholder = tf.placeholder(dtype=tf.string, name="opts_summary_placeholder")
                opts_summary = tf.summary.text("arguments", opts_summary_placeholder,
                                               collections=[tf.GraphKeys.SUMMARIES],
                                               )
                summary = sess.run(opts_summary, feed_dict={opts_summary_placeholder: str(opts)})
                summary_writer.add_summary(summary, tf.train.get_global_step(g).eval())

                if not opts.test_mode:
                    opts.save()
                    inducer.train_model(sess, saver, summary_writer)
                    saver.save(sess, checkpoint_dir, tf.train.get_global_step(g))
                logger.debug("Testing")
                inducer.summary_evaluation(sess, 'valid', summary_writer)
                inducer.summary_evaluation(sess, 'test', summary_writer)
                summary_writer.close()
    logger.info("$End")

