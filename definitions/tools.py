import collections
from collections import defaultdict
import errno
import fcntl
import logging
import os
import re
import time
from glob import glob

import typing
from functools import wraps

import numpy as np
import tensorflow as tf

from definitions import settings, log_dict, tools
import pickle

# ----------- Definitions ----------------------------
# -------- GraphKey -------------------------------

PlaceholdersKey = "PlaceholdersKey"
FeatureInput = "FeatureInput"
TrainSummaries = "TrainSummaries"
ValidationSummaries = "ValidationSummaries"
ValidationDecoderSummaries = "ValidationDecoderSummaries"
TestSummaries = "TestSummaries"
TestDecoderSummaries = "TestDecoderSummaries"
LossSummaries = "LossSummaries"
SummaryNorm = "SummaryNorm"
HistogramSummaryTensors = "HistogramSummaryTensors"
AddToScalarSummaryKey = "AddToScalarSummaryKey"
LossKey = "Loss"
RelationKey = "Relation"
LinkPredictionScoreKey = "LinkPredictionScoreKey"

summary_key = {'train': TrainSummaries, 'valid': ValidationSummaries, 'test': TestSummaries,
               'valid-decoder': ValidationDecoderSummaries,
               'test-decoder': TestDecoderSummaries}

PROF_DATA = {}
logger = logging.getLogger(__name__)

norm_dict = {"F": 'euclidean',
             "E": 'euclidean',
             "0": 0,
             "1": 1,
             "2": 2,
             "N": "N",
             "1A": 1,
             "0A": 0,
             "2A": 2,
             "FA": 'euclidean',
             "EA": 'euclidean'
             }
optimizer_list = [tf.train.GradientDescentOptimizer,  # type: typing.List[tf.train.Optimizer]
                  tf.train.AdagradOptimizer,
                  tf.train.AdadeltaOptimizer,
                  tf.train.AdagradOptimizer,
                  tf.train.AdamOptimizer,
                  ]


# ------- file lock -----
class FileLock:
    def __init__(self, file_name, mode='a+'):
        self.fn = file_name
        self.mode = mode

    def __enter__(self):
        self.fp = acquire_file_lock(self.fn, self.mode)
        return self.fp

    def __exit__(self, type, value, traceback):
        release_file_lock(self.fp)


def acquire_file_lock(file_name, mode='a+'):
    fp = open(file_name, mode)
    while True:
        try:
            fcntl.lockf(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except:
            continue
    return fp


def release_file_lock(fp):
    fcntl.lockf(fp, fcntl.LOCK_UN)
    fp.close()


# ----------- directory ---------------
def assign_model_save_path(model_dir, experiment_mode):
    output_dir = os.path.join(model_dir, experiment_mode, "m")
    output_dir = find_available_filename(output_dir)
    make_sure_path_exists(output_dir)
    return output_dir


def find_available_filename(name, num=1):
    '''
    find (but not create) a new file/dir, append {04d} to name(befor extention)
    '''
    assert num >= 1, "num wrong:{}".format(num)
    """Get file or dir name that not exists yet"""
    file_name, file_extension = os.path.splitext(name)
    dirname = os.path.dirname(name)
    make_sure_path_exists(dirname)

    lock_file = os.path.join(dirname, settings.lock_file_name)

    with FileLock(lock_file, 'a+') as fw, open(lock_file, 'r') as fr:
        lines = fr.readlines()
        i = 0
        if len(lines) > 0:
            i = int(lines[-1]) + 1
        else:
            while True:
                tmp = "{}_{:04d}_{}".format(file_name, i, file_extension)
                if not os.path.exists(tmp):
                    break
                i += 1

        return_filename_list = []
        for j in range(num):
            tmp = "{}_{:03d}_{}".format(file_name, i, file_extension)
            assert not os.path.exists(tmp), "dir exists: {}".format(tmp)

            if file_extension:
                p = os.path.dirname(tmp)
            else:
                p = tmp

            # make_sure_path_exists(p)
            fw.write("{}\n".format(i))
            i += 1
            return_filename_list.append(tmp)

    if num == 1:
        return return_filename_list[0]
    else:
        return return_filename_list


def make_sure_path_exists(p):
    #    p = os.path.dirname(path)
    try:
        os.makedirs(p)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def change_file_extension(file_path, target_ext):
    if not target_ext.startswith('.'):
        target_ext = '.' + target_ext
    fn, fe = os.path.splitext(file_path)
    return fn + target_ext


def complete_dir(d):
    d = os.path.abspath(os.path.expanduser(d))
    if not os.path.exists(d):
        if '*' not in d:
            d = d + '*'
        candidates = glob(d)
        if not candidates:
            raise FileNotFoundError('path {} not found'.format(d))
        elif len(candidates) > 1:
            raise ValueError('path {} matches multiple dir'.format(d))
        else:
            d = candidates[0]
    return d


def soft_link_model_dir(source, target, newid=False, abs_path=True):
    """
    create softlink from source to target, with same name. and (optional) different id
    :param source: source dir. typically start with 'm_'
    :param target: target dir
    :param newid: if ignore original model id.
    :return:
    """
    logger = logging.getLogger(__name__)
    if abs_path:
        source = os.path.abspath(os.path.expanduser(source))
    base = os.path.basename(os.path.normpath(source))
    target = os.path.abspath(os.path.expanduser(target))
    make_sure_path_exists(target)

    if newid:
        base = base.split('_', 2)[2]
        target = os.path.join(target, 'm')
        target = tools.find_available_filename(target)
        target += base
    else:
        target = os.path.join(target, base)
    os.symlink(source, target)
    logger.debug('(link) {} to {}'.format(source, target))


def tos(x):
    if isinstance(x, bool):
        return 'T' if x else 'F'
    elif isinstance(x, float):
        try:
            if np.abs(int(x) - x) < 1e-9:
                return str(int(x))
            else:
                return '{:.1e}'.format(x)
        except:
            return 'nan'
    else:
        return str(x)


# --------- OS -----------
def memory():
    """
    Get node total memory and memory usage
    """
    with open('/proc/meminfo', 'r') as mem:
        ret = {}
        tmp = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) == 'MemTotal:':
                ret['total'] = int(sline[1])
            elif str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                tmp += int(sline[1])
        ret['free'] = tmp
        ret['used'] = int(ret['total']) - int(ret['free'])
    return ret


# --------- Logging ---------
def log_b3_eval(logger, step, epoch, data, eval_results):
    names = ['B3-F1', 'B3-F0.5', 'B3-Recall', 'B3-Precision']
    for i in range(4):
        logger.info(log_dict.EvalLog(step, epoch, data, names[i], eval_results[i]))


# ---------- profile wrapper ----------
def print_func_name(fn):
    @wraps(fn)
    def within_print(*args, **kwargs):
        cutoff = 4
        logger.debug("Executing {}".format(fn.__qualname__))
        ret = fn(*args, **kwargs)
        logger.debug("Done      {}".format(fn.__qualname__))
        return ret

    return within_print


def profile(fn):
    @wraps(fn)
    def with_profiling(*args, **kwargs):
        start_time = time.time()

        ret = fn(*args, **kwargs)

        elapsed_time = time.time() - start_time
        logger.debug("Done [{:s}] ({:2f}s)".format(fn.__qualname__, elapsed_time))
        # if fn.__name__ not in PROF_DATA:
        #     PROF_DATA[fn.__name__] = [0, []]
        # PROF_DATA[fn.__name__][0] += 1
        # PROF_DATA[fn.__name__][1].append(elapsed_time)

        return ret

    return with_profiling


# ---------- named_tuple_wrapper ----------
def namedtuple_add_defaults(T, default_values=()):
    T.__new__.__defaults__ = (None,) * len(T._fields)
    if isinstance(default_values, collections.Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T


def namedtuple_to_function_arguments(nt):
    fields = nt._fields
    s = ""
    for i in range(len(fields)):
        if nt[i] is None:
            continue
        if isinstance(nt[i], str):
            value = "'{:s}'".format(nt[i])
        elif isinstance(nt[i], type):
            value = re.search("'(.*?)'", str(nt[i])).group(1)
        else:
            value = str(nt[i])
        s += fields[i] + "=" + value + ","
    return s[:-1]


# -------- Math tools -------------------------------
def index_elements(elements: list) -> typing.Tuple[list, typing.Dict[object, int]]:
    if not isinstance(elements, list):
        raise TypeError("_index_elements takes list but not {:s}".format(str(type(elements))))

    id2str = elements
    str2id = dict(zip(elements,
                      np.arange(len(elements)
                                )
                      )
                  )
    return id2str, str2id


def normalize(array, ord=2, dim=0):
    return array / np.linalg.norm(array, ord=ord, axis=dim, keepdims=True)


def list_subtraction(a, b):
    return [x for x in a if x not in b]


def list_intersection(a, b):
    return [x for x in a if x in b]


# --------- save pickle --------------------
def dump_pickle(o, p):
    with open(p, 'wb') as f:
        pickle.dump(o, f)


def load_pickle(p):
    with open(p, 'rb') as f:
        pickle.load(f)


# - dict -
def invert_dict(d):
    nd = {}
    for k, v in d.items():
        if v not in nd:
            nd[v] = k
        else:
            print('({}:{}) conflict ({})'.format(v, k, nd[v]))

    return nd


def invert_dict_with_duplicate(d):
    nd = defaultdict(list)
    for k, v in d.items():
        nd[v].append(k)
    return nd


# --------- Tensorflow Helpers ------------------------------
def variable_on_cpu(name, shape, initializer):
    """
    Helper to create a Variable stored on CPU memory.
    :param name: name of the variable
    :param shape: list of ints
    :param initializer: initializer for Variable
    :return: Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var


def variable_with_weight_decay(name, shape, initializer, weight_decay):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    var = variable_on_cpu(
        name,
        shape,
        initializer)
    if weight_decay is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def get_or_create_variable(*args, **kwargs):
    try:
        v = tf.get_variable(*args, **kwargs)
    except ValueError:
        scope = tf.get_variable_scope()
        scope.reuse_variables()
        v = tf.get_variable(*args, **kwargs)
        scope._reuse = False
    return v


def l1_regularizer(tensor):
    return tf.reduce_sum(tf.abs(tensor))


def regularizer_l1_l2(l1, l2: float):
    def reg(x):
        regularization = 0
        if abs(l1) > 1e-15:
            regularization += l1 * l1_regularizer(x)
        if abs(l2) > 1e-15:
            # TODO: should not *2 here, but diego doesn't do that
            regularization += (2 * l2) * tf.nn.l2_loss(x)
        return regularization

    return reg


def loss_summary(loss_list):
    with tf.name_scope("loss_summary"):
        scalar_summary(loss_list)


def scalar_summary(lst):
    for l in lst:
        s = tf.summary.scalar(l.op.name, l,
                              [tf.GraphKeys.SUMMARIES, TrainSummaries])


def variable_summary():
    varlist = set(tf.trainable_variables() + tf.get_collection(tools.HistogramSummaryTensors))
    for var in varlist:
        s = tf.summary.histogram(var.op.name, var,
                                 collections=[tf.GraphKeys.SUMMARIES, TrainSummaries])


def norm_summary(var, axis, norm_ord='euclidean', collections=None):
    norm = tf.norm(var, axis=axis, ord=norm_ord, name=var.op.name + "_Norm")
    tf.add_to_collection(HistogramSummaryTensors, norm)
    return norm


def cal_entropy(probs, name='entropy'):
    probs = tf.clip_by_value(probs, 1e-10, 1)
    entropy_sum = tf.negative(
        tf.reduce_sum(tf.log(probs) * probs, ), name=name)
    return entropy_sum


# --------- Numpy ------------------------------
def onehot(x, row_size):
    mat = np.zeros((len(x), row_size))
    mat[np.arange(len(x)), x] = 1
    return mat


# --------- DataFrame  ------------------------------
def df_append_row(df_input, row, save_all=True):
    import logtools
    if isinstance(df_input, str):
        filelock = acquire_file_lock(df_input + '.lc')
        df = logtools.read_df(df_input)
    else:
        df = df_input

    cols = df.columns
    df = df.append(row, ignore_index=True)[cols]
    if isinstance(df_input, str):
        logtools.write_df(df, df_input, save_all)
        release_file_lock(filelock)
    return df


# --------- Summary ------------------------------
def activation_summary(x, family_name=None):
    """
    Helper to create summaries for activations:
    :param x:
    :param family_name: tf r1.3 feature
    :return:
    """
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x,
                         collections=[tf.GraphKeys.SUMMARIES, TrainSummaries], )
    tf.summary.scalar(tensor_name + '/mean', tf.reduce_mean(x),
                      collections=[tf.GraphKeys.SUMMARIES, TrainSummaries])


def tf_cal_kl(q, p, name=None):
    """
    calculate KL(q||p) = \sum_r q \log(q/p)
    :param q: (...,r)
    :param p: (..., r) same dim with q
    :param name: op name
    :return: kl, (...,)
    """
    p = tf.maximum(p, 1e-6)
    kl = tf.reduce_sum(q * tf.log(q / p), axis=-1, name=name)
    return kl
