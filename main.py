import logging
import os
import sys
from argparse import ArgumentParser

from definitions import settings, tools, arguments
from definitions.arguments import OieParser
from learning import OieInduction, OieReg, DecoderTrainer

# def parser
# ------- ----- parent parser ------- -----
basic_arguments_parser = OieParser(add_help=False)
# training control parameters
basic_arguments_parser.add_argument('--input_dir', '-in', dest='input_dir', default=settings.data_default_settings,
                                    action=arguments._InputDirAction, type=str,
                                    help='model input directory')
basic_arguments_parser.add_argument("--output_dir", '-out', dest="output_dir", default=None, type=str,
                                    help="model output directory")
basic_arguments_parser.add_argument("-v", dest="v", default=0, action="count",
                                    help="Increase verbosity. -v at most. ")
basic_arguments_parser.add_argument("--gpus", dest="gpus", default=0, type=int, nargs="?",
                                    help="GPU usage. -1 for None. 0 for gpu0.")
basic_arguments_parser.add_argument("--test_mode", action='store_true', help='test mode')
# model parameters
basic_arguments_parser.add_argument("--model", "-m", dest="model", nargs='?', type=str, default="AC",
                                    help='Model Type choose among A:Bilinear, C:Selectional, AC, T(ransE) $ID$',
                                    choices=["A", "C", "AC", "T", 'AC-Fix'
                                            ])
basic_arguments_parser.add_argument("--optimization", "-optm", dest="optimization", nargs='?', type=int, default=1,
                                    help='optimization algorithm 0 SGD, 1 ADAGrad, 2 ADADelta, 3 AdaGrad, 4 Adam. Default SDG. $ID$',
                                    choices={0, 1, 2, 3, 4})
# ------- ----- Hyper parameters------- -----
basic_hyperparameters_parser = OieParser(add_help=False, parents=[basic_arguments_parser])
basic_hyperparameters_parser.add_argument("--epochs", '-ep', dest="epochs", type=int, default=120,
                                          help='maximum number of epochs')
basic_hyperparameters_parser.add_argument("--learning_rate", "-lr", dest="learning_rate", type=float, default=0.5,
                                          help='initial learning rate $ID$')
basic_hyperparameters_parser.add_argument("--batch_size", '-bz', dest="batch_size", type=int, default=100,
                                          help='size of the mini batches $ID$')
basic_hyperparameters_parser.add_argument("--embed_size", dest="embed_size", type=int, default=50,
                                          help='initial learning rate $ID$')  # k
basic_hyperparameters_parser.add_argument("--emb_normalize", '-en', dest="emb_normalize", action="store_true",
                                          default=True,
                                          help="normalize entity embeddings $ID$")
basic_hyperparameters_parser.add_argument("--no-emb_normalize", dest="emb_normalize", action="store_false",
                                          default=False,
                                          help="NOT normalize entity embeddings")
basic_hyperparameters_parser.add_argument("--no_emb_normalize", dest="emb_normalize", action="store_false",
                                          default=True,
                                          help="normalize entity embeddings ")
basic_hyperparameters_parser.add_argument("--negative_samples_number", dest="negative_samples_number",
                                          type=int, default=10, help='number of negative samples $ID$')
basic_hyperparameters_parser.add_argument("--negative_samples_power", dest="negative_samples_power",
                                          type=float, default=0,
                                          help='power over frequencies of entities, to define there negative sampling '
                                               'probability distribution $ID$')
# Regularization
basic_hyperparameters_parser.add_argument("--l1_regularization", dest="l1_regularization", nargs='?', type=float,
                                          default=0.0, action=arguments._WeightDecayAction,
                                          help='lambda value of L1 regularization $ID$')
basic_hyperparameters_parser.add_argument("--l2_regularization", '-l2', dest="l2_regularization", nargs='?', type=float,
                                          default=5e-8, action=arguments._WeightDecayAction,
                                          help='lambda value of L2 regularization $ID$')
basic_hyperparameters_parser.add_argument("--extended_reg", '-dr', dest="extended_reg", default=False,
                                          action='store_true',
                                          help='extended regularization on reconstruction parameters, default false $ID$')

basic_hyperparameters_parser.add_argument("--decoder_relation_norm_normalize", "-drnn",
                                          dest="decoder_relation_norm_normalize",
                                          default="N", choices=["N", "1", "F", "1A", "FA"],
                                          help="normailize decoder relation matrix with norm 1 $ID$")
# --------oie part ------- -----
basic_oie_parser = OieParser(add_help=False, parents=[basic_hyperparameters_parser])
basic_oie_parser.add_argument('--encoder', '-ecd', dest='encoder', type=str, default='F',
                              help='F(eature): features processed by diego, '
                                   'SE: sentence embedding. '
                                   'EV: entity vectors from ext_kb '
                                   ' All must be separated by "+"'
                                   '$ID$')
basic_oie_parser.add_argument("--relations_number", dest="relations_number", type=int,
                              default=40,
                              help='number of relations to induce $ID$')
basic_oie_parser.add_argument("--alpha_init", "-ai", type=float, default=4.0,
                              help='initial alpha coefficient for scaling the entropy term $ID$')
basic_oie_parser.add_argument('--alpha_change', type=str, default='exp',
                              help='form for the growth or decay of alpha: None, linear, exp, power $ID$')
basic_oie_parser.add_argument('--alpha_final', '-af', type=float, default=1e-5,
                              help='final alpha if alpha_change is applied $ID$')
basic_oie_parser.add_argument("--load_ext_kb_dir", '-lekd', dest="load_ext_kb_dir", default=settings.ext_kb_default,
                              help='ext kb model dir $ID$')
basic_oie_parser.add_argument('--original_decoder_score', action='store_true',
                              help='apply relation probability to decoder relation matrix $ID')
# --------oie_reg part ------- -----
basic_oie_reg_parser = OieParser(add_help=False, parents=[basic_oie_parser])
basic_oie_reg_parser.add_argument("--reg_model", "-rm", dest="reg_model", nargs='?', type=str, default="TransE",
                                  choices=["None", "TransE", "TD"],
                                  help='Regularizer Model Type choose among: None, TransE:TransE, TD: TransE over Decocer', )
basic_oie_reg_parser.add_argument("--beta_init", "-bi", type=float, default=4.0,
                                  help='initial beta coefficient for scaling the entropy term $ID$')
basic_oie_reg_parser.add_argument('--beta_change', '-bc', type=str, default='exp',
                                  help='form for the growth or decay of beta: None, linear, exp, power $ID$')
basic_oie_reg_parser.add_argument('--beta_final', '-bf', type=float, default=1e-5,
                                  help='final beta if beta_change is applied $ID$')
basic_oie_reg_parser.add_argument('--adj_cos', '-adjcos', dest='adj_cos', action='store_true',
                                  help='use adjust cos to compute similarity $ID$')
basic_oie_reg_parser.add_argument('--no-adj_cos', '-nadjcos', dest='adj_cos', action='store_false',
                                  help='do not use adjust cos to compute similarity $ID$')
basic_oie_reg_parser.set_defaults(adj_cos=True)
basic_oie_reg_parser.add_argument('--abs_cos', '-abscos', dest='abs_cos', action='store_true',
                                  help='use abs(cos) to compute similarity $ID$')
basic_oie_reg_parser.add_argument('--no-abs_cos', '-nabscos', dest='abs_cos', action='store_false',
                                  help='do not use abs(cos) to compute similarity $ID$')
basic_oie_reg_parser.set_defaults(abs_cos=True)
basic_oie_reg_parser.add_argument('--cos_threshold', '-ct', type=float, default=0.0,
                                  help='threshold for cos similarity $ID$')
basic_oie_reg_parser.add_argument('--distance', '-rd', type=str, default='euclidean',
                                  choices=['euclidean', 'KL', 'JS'],
                                  help='form for distance computing: euclidean $ID$')
# ------- -----  kb_reg part ------- -----
basic_kb_reg_parser = OieParser(add_help=False, parents=[basic_oie_parser])
basic_kb_reg_parser.add_argument("--kb_reg_weight", '-krw', dest="kb_reg_weight", default=1.0, type=float,
                                 help="kb regularization weight $ID$")
basic_kb_reg_parser.add_argument("--kb_reg_l2", type=float, default=5e-8,
                                 help='l2 for kb reg variables $ID$'
                                 )
basic_kb_reg_parser.add_argument('--normalize_enemb', '-ne', dest='normalize_enemb',
                                 action='store_true', default=False,
                                 help='normalize entity embeddings $ID')
# ------- -----  ext_kb part ------- -----
basic_ext_kb_parser = OieParser(add_help=False, parents=[basic_hyperparameters_parser])
basic_ext_kb_parser.add_argument('--kbemb_dim', '-kd', default=50, type=int,
                                 help='ext kb dimension $ID$')
basic_ext_kb_parser.add_argument('--inversion', '-inv', default=False, action='store_true',
                                 help='training kb with additional inversion relations $ID')
basic_ext_kb_parser.add_argument('--not_save_parameters', default=True, action='store_false',
                                 help='save kb model parameters to npz')

parser_dict = {}
# program parser(oie_parser)
program_parser = OieParser(add_help=True)
# subparsersk
subparsers = program_parser.add_subparsers()
# oie_parser
parser_dict['oie'] = parser_oie = subparsers.add_parser('oie', add_help=True, help='Open IE',
                                                        parents=[basic_oie_parser])
parser_oie.set_defaults(run=OieInduction.run)
parser_oie.set_defaults(experiment_mode='oie')
# oie_parser
parser_dict['oie_reg'] = parser_oie = subparsers.add_parser('oie_reg', add_help=True,
                                                            help='Open IE with regularization',
                                                            parents=[basic_oie_reg_parser])
parser_oie.set_defaults(run=OieReg.run)
# parser_oie.set_defaults(run=OieReg.kmeans)
parser_oie.set_defaults(experiment_mode='oie_reg')
# ext_kb_parser
parser_dict['ext_kb'] = parser_ext_kb = subparsers.add_parser('ext_kb', add_help=True, help='train KB model',
                                                              parents=[basic_ext_kb_parser])
parser_ext_kb.set_defaults(experiment_mode='ext_kb')
parser_ext_kb.set_defaults(run=DecoderTrainer.run)
# kb_reg parser
parser_dict['en_pair_kl'] = parser_en_pair_kl = subparsers.add_parser('en_pair_kl', add_help=True,
                                                                      help='train kb regularization model, use kl',
                                                                      parents=[basic_kb_reg_parser])
# other parsers
parser_outmodel = subparsers.add_parser('outmodel', help='output model parameters', parents=[basic_arguments_parser])
parser_outmodel.set_defaults(run=DecoderTrainer.output_model, )
parser_outmodel.add_argument('--dir', '-d', help='load/save dir')


def run_model(main_opts, args):
    # parse, get default arguments related to model
    parser = program_parser
    # parse
    opts = parser.parse_args(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opts.gpus)

    # run
    if opts.output_dir is not None:
        tools.make_sure_path_exists(opts.output_dir)
        OieParser.config_log(opts.output_dir, opts.v)
    else:
        logging.basicConfig(level='DEBUG')
        logging.warning('No output log file')
    logger = logging.getLogger(__name__)
    logger.debug(opts)
    opts.run(opts)
    return opts


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--use_data_example", dest="use_data_example", )
    opts, args = parser.parse_known_args(sys.argv[1:])
    run_model(opts, args)
