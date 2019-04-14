import argparse
import json
import logging
import logging.config
import os
import subprocess
from copy import deepcopy
from datetime import datetime

import pandas as pd

import definitions
from definitions import settings, tools, log_dict

time_format = "%Y-%m-%d %H:%M:%S.%f"


def strip_args(args):
    if args is None:
        return None
    name_set = set()
    # remove redundant args
    for i in range(len(args) - 1, -1, -1):
        name = args[i]
        if name.startswith("--"):
            if name in name_set:  # a redundant argument
                args.pop(i)
                if not args[i].startswith("--"):
                    args.pop(i)
            else:
                name_set.add(name)
    return args


class _HoldoutRelationDirAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):

        if isinstance(values, str):
            if values.startswith(('freq', 'random')):
                values = os.path.join(settings.data_root, 'holdout',
                                      values)
            if not values.endswith('.pkl'):
                values += '.pkl'
        setattr(namespace, self.dest, values)


class _WeightDecayAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        adjustment = 5e-5
        values *= adjustment
        setattr(namespace, self.dest, values)


class _InputDirAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if not os.path.exists(values):
            values = os.path.join(settings.data_root, values)
            if not os.path.exists(values):
                raise FileNotFoundError('InputDir %s' % values)
        setattr(namespace, self.dest, values)


# class _LoadKBModelDirAction(argparse.Action):
#     pass


class _EncoderChoicesContainer:
    def __init__(self, encoders):
        raise NotImplementedError()
        self.encoders = list(encoders)

    def __contains__(self, item):
        item = item.split('+')
        for k in item:
            if k not in self.encoders:
                return False
        return True

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration


class OieParser(argparse.ArgumentParser):
    kbreg_modes = ['kb_lookup_id', 'en_pair_kl']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def parse_known_args(self, args=None, namespace=None, generate_dir=False):
        if namespace is None:
            namespace = LearningArguments()
        assert isinstance(namespace, LearningArguments), "namespace wrong type:" + str(type(namespace))
        generate_dir_str = '~~GENERATE~DIR~~'
        if generate_dir:
            args.append(generate_dir_str)

        namespace, return_args = super().parse_known_args(args, namespace)
        # super parse know will call this function again
        if generate_dir_str in return_args:
            output_dir = self.generate_output_dir(namespace)
            if output_dir is not None:
                namespace.output_dir = output_dir
            return_args.remove(generate_dir_str)
        if generate_dir_str in args:
            args.remove(generate_dir_str)
        return namespace, return_args

    def parse_args(self, args=None, namespace=None, generate_dir=True):
        args, argv = self.parse_known_args(args, namespace, generate_dir)
        if argv:
            msg = argparse._('unrecognized arguments: %s')
            self.error(msg % ' '.join(argv))
        return args

    def try_parse_known_args(self, args=None, namespace=None):
        if namespace is None:
            namespace = LearningArguments()
        args = strip_args(args)
        namespace, args = self.parse_known_args(args, namespace, False)

        return namespace, args

    def try_parse_args(self, args=None, namespace=None):
        args, argv = self.try_parse_known_args(args, namespace, )
        if argv:
            msg = argparse._('unrecognized arguments: %s')
            self.error(msg % ' '.join(argv))
        return args

    def generate_output_dir(self, namespace):
        if isinstance(namespace, list):
            namespace, _ = self.try_parse_args(namespace)
        if not hasattr(namespace, 'output_dir') or not hasattr(namespace, 'experiment_mode'):
            return None
        output_dir = namespace.output_dir
        if output_dir is None:
            model_dir = settings.model_dir
            experiment_mode = namespace.experiment_mode

            output_dir = os.path.join(model_dir, experiment_mode, "m")
            output_dir = tools.find_available_filename(output_dir)
            # model_id = namespace._get_file_name(self)
            # output_dir += model_id

        tools.make_sure_path_exists(output_dir)
        return output_dir

    def get_type_dict(self):
        d = {}
        for item in self._actions:
            if item.__class__ == argparse._HelpAction:
                continue
            elif item.__class__ == argparse._StoreTrueAction:
                t = bool
            elif item.__class__ == argparse._CountAction:
                t = int
            elif item.type is None:
                t = str
            else:
                t = item.type
            d[item.dest] = t
        return d

    def get_option_strings(self):
        d = {}
        for action in self._actions:
            if isinstance(action, argparse._StoreFalseAction):
                continue
            else:
                option_strings = action.option_strings
                d[action.dest] = option_strings
        return d

    def get_args_fields(self):
        # remove duplicate
        return list(set([x.dest for x in self._actions if x.dest not in {'v', 'gpus', 'test_mode', }]))

    def get_id_fields(self):
        return [x.dest for x in self._actions if x.help is not None and '$ID$' in x.help]

    @staticmethod
    def config_log(output_dir, log_level=0):
        logfilename = os.path.join(output_dir, '.log')
        logging.config.dictConfig(log_dict.logging_config_dict(log_dict.logging_level[log_level], logfilename))
        logger = logging.getLogger(__name__)
        logger.info("$Start $Version {:s}".format(definitions.__version__))

    @staticmethod
    def hrd_to_id(holdout_relation_dir):
        hrd = holdout_relation_dir.split('/')
        hrd = hrd[-2:]
        hrd[-1] = hrd[-1].split('.')[0]
        hrd = '/'.join(hrd)
        return hrd


class LearningArguments(argparse.Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __str__(self):
        d = self.get_dict()

        return json.dumps(d)

    def get_dict(self):
        d = self.__dict__
        if hasattr(self, 'run'):
            d = deepcopy(self.__dict__)
            d.pop('run')

        return d

    def save(self, output_dir=None):
        if not output_dir:
            output_dir = self.output_dir
        tools.make_sure_path_exists(output_dir)
        self.output_time = str(datetime.now().strftime(time_format))
        output_filename = os.path.join(output_dir, "options.json")
        output_filename = tools.find_available_filename(output_filename)

        json.dump(self.get_dict(), open(output_filename, 'w'), indent=13)

        logger = logging.getLogger(__name__)
        logger.debug("Save learning arguments to {}".format(output_filename))
        logger.info("[Learning Arguments] {}".format(str(self)))

    def load(self, input_filename=None):
        if not input_filename:
            input_filename = os.path.join(self.output_dir, "options_000_.json")

        d = json.load(open(input_filename, 'r'))
        self.__dict__.update(d)
        return self, d

    def loads(self, input_str):
        d = json.loads(input_str)
        self.__dict__.update(d)
        return self, d

    def _get_file_name(self, parser=None):
        def tos(x):
            if isinstance(x, bool):
                return 'T' if x else 'F'
            elif isinstance(x, float):
                return '{:.1e}'.format(x)
            else:
                return str(x)

        name = ""
        # holdout relation dir
        for action in parser._actions:
            if isinstance(action, argparse._StoreFalseAction):
                continue
            elif not hasattr(self, action.dest):
                continue
            else:
                option_strings = action.option_strings
                for s in option_strings:
                    if s.startswith('-') and not s.startswith('--'):
                        value = getattr(self, action.dest)
                        if 'dir' in action.dest:
                            if action.dest == 'input_dir':
                                name += s[1:] + os.path.basename(value) + '_'
                            if action.dest == 'load_ext_kb_dir':
                                name += s[1:] + str(value) + '_'
                        elif action.dest not in ['v']:  # skip_actions
                            name += s[1:] + tos(value) + '_'
                        break
        return name[:-1]


# def preset_arguments(model_name):
#     args = ["--optimization", "1",
#             "--epochs", "10",
#             "--batch_size", "100",
#             "--relations_number", "100",
#             "--negative_samples_number", "20",
#             "--l2_regularization", "0.1",
#             "--learning_rate", "0.1",
#             "--embed_size", "30",
#             ]
#     if model_name == "A":
#         args += [
#             "--model", "A",
#             "--alpha_init", "0.25",
#         ]
#     elif model_name == "AC":
#         args += [
#             "--model", "AC",
#             "--alpha_init", "0.1",
#                  ]
#     elif model_name == "C":
#         args += [
#             "--model", "C",
#             "--alpha_init", "0.01",
#                  ]
#     return args


def get_parser_name_short(parser):
    d = {}

    for action in parser._actions:
        option_strings = action.option_strings
        if len(option_strings) != 2:
            continue
        else:
            a, b = option_strings[0][1:], option_strings[1][1:]
            if not a.startswith('-'):
                a, b = b, a
            d[a[1:]] = b
    return d


# ------ ext kb ------
# def get_ext_kb_id(kbreg_args, id_fields=None):
#     if isinstance(kbreg_args, LearningArguments):
#         opts = kbreg_args
#     else:
#         opts = oie_parser.try_parse_args(kbreg_args)
#     if id_fields is None:
#         id_fields = ['model', 'holdout_relation_dir', 'embed_size',
#                      'decoder_relation_norm_normalize',
#                      'emb_normalize',
#                      ]

#     extkb_id = {k: getattr(opts, k) for k in id_fields}
#     extkb_id['experiment_mode'] = 'ext_kb'
#     return extkb_id


# def get_extkb_dir(kbreg_args, id_fields=None, df_path=None, train_if_nonexists=True):
#     raise NotImplementedError()
#     import logtools
#     if isinstance(kbreg_args, LearningArguments):
#         opts = kbreg_args
#     else:
#         opts = oie_parser.try_parse_args(kbreg_args)

#     if opts.experiment_mode not in OieParser.kbreg_modes:
#         return ""

#     if opts.load_ext_kb_dir is not None and opts.load_ext_kb_dir != "":
#         return opts.load_ext_kb_dir

#     extkb_id = get_ext_kb_id(opts, id_fields)
#     # https://stackoverflow.com/questions/34157811/filter-a-pandas-dataframe-using-values-from-a-dict
#     if df_path is None:
#         df_path = os.path.join(settings.model_dir, "ext_kb", 'summary_Epoch_max.csv')
#     df = logtools.read_df(df_path)
#     try:
#         return df.loc[(df[list(extkb_id)] == pd.Series(extkb_id)).all(
#             axis=1)].reset_index().loc[0, 'output_dir']
#     except KeyError:
#         # the model do not exist
#         logger = logging.getLogger(__name__)
#         logger.warning("Ext KB do not exists! ")

#         if train_if_nonexists:
#             logger.debug("Training new.")
#             args = expand_ext_kb_arguments(extkb_id, opts)
#             subprocess.run(['python3', '-m', 'main', ] + args,
#                            check=True)
#             subprocess.run(
#                 ['python3', '-m', 'logtools', '-p', os.path.dirname(df_path), '-s', 'Epoch', '-m', 'max'],
#                 check=True)
#             return get_extkb_dir(opts, id_fields, df_path, False)
#         else:
#             raise Exception("ExtKB not found {:s}".format(str(extkb_id)))


# def expand_ext_kb_arguments(extkb_id, kbreg_args=None):
#     args = []
#     for k, v in extkb_id.items():
#         if isinstance(v, bool):
#             if v:
#                 args += ['--' + k]
#         elif v is None:
#             continue
#         else:
#             args += ['--' + k, str(v)]
#     args += ['--epochs', str(100),
#              '--shuffle_flag',
#              '--learning_rate', '0.5',
#              ]
#     if kbreg_args is not None:
#         if isinstance(kbreg_args, LearningArguments):
#             opts = kbreg_args
#         else:
#             opts = oie_parser.try_parse_args(kbreg_args)
#         args += ['--gpus', str(opts.gpus),
#                  ]
#         if opts.v > 0:
#             args.append('-' + 'v' * opts.v)

#     return args


if __name__ == "__main__":
    # options = LearningArguments()
    # options.parse("--model A --pickled_dataset ./data/data.pkl".split())
    settings.model_dir = '/tmp/tmp'
    tools.make_sure_path_exists(settings.model_dir)

