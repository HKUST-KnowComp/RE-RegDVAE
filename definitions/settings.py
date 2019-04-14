import os
import typing
from collections import namedtuple
from definitions.OieExample import OieExample

pkl_protocol = 4

# ---------- Definitions ----------
DataDir = namedtuple('DataDir', ['root', 'train', 'valid', 'test'])
DataSplitType = typing.List[OieExample]
DatasetType = typing.Dict[str, DataSplitType]
OieEvaluationResults = namedtuple("OieEvaluationResults", ['F1', 'F0_5', 'Recall', 'Precision'])
# ---------- Directories ----------
use_data_example = False
if use_data_example:
    raw_data_root_dir = "./data"
    data_example_dir = os.path.join(raw_data_root_dir, "data-sample.txt")
    raw_data_dir = DataDir(raw_data_root_dir,
                           data_example_dir,
                           data_example_dir,
                           data_example_dir,
                           )
    data_root = "./data"
    data_example_root = data_root
    # data_root = '/tmp'
    # data_example_model_dir = model_dir = "/data/yan/rl-pr-vae/oie_sample/"
    data_example_model_dir = model_dir = '/tmp'
else:
    raw_data_root_dir = "./data/diego/raw"
    raw_data_dir = DataDir(raw_data_root_dir,
                           os.path.join(raw_data_root_dir, 'candidate-2000s.context.filtered.triples.pathfiltered.pos'
                                                           '.single-relation.sortedondate.txt'),
                           os.path.join(raw_data_root_dir, 'candidate-2000s.context.filtered.triples.pathfiltered.pos'
                                                           '.single-relation.sortedondate.validation.20%.txt'),
                           os.path.join(raw_data_root_dir, 'candidate-2000s.context.filtered.triples.pathfiltered.pos'
                                                           '.single-relation.sortedondate.test.80%.txt'),
                           )

    data_root = "./data"
    # data_root = './data/zhiyuan/processed'

    # model_dir = "/home/data/yan/rl-pr-vae/oie_nyt/m"

    model_dir = "./model"

data_default = 'NYT122'
ext_kb_default = '*mFT_d50_invF*'
if data_default == 'NYT122':
    data_default_settings = os.path.join(data_root, "NYT122")
elif data_default == 'NYT71':
    data_default_settings = os.path.join(data_root, "NYT71")
elif data_default == 'NYT27':
    data_default_settings = os.path.join(data_root, "NYT27")

# file lock name
lock_file_name = "lock_file.lc"
save_parameters_file_name = 'parameters.npz'

# names

ENTITY_EMBEDDING_NAME = 'entitiy_embeddings'
ENTITY_BIAS_NAME = 'entitiy_biases'
BILINEAR_RELATION_MATRIX_NAME = 'relation_matrix'
SP_RELATION_EMBEDDING_NAME = 'relation_embeddings'
TransE_ENTITY_EMBEDDING_NAME = 'TransE_entitiy_embeddings'
TransE_RELATION_EMBEDDING_NAME = 'TransE_relation_embeddings'

# stats keys
e_num_statkey = 'entity_num'
r_num_statkey = 'relation_num'

# --------- Sync Experiment Result ------------------------------
out_exp_result_local_dir = os.path.join(data_root, "exp_results")
# out_exp_result_sync_dir = os.path.expanduser("~/Dropbox/Experiments/rl-pr-vae/")

# --------- Parameters ------------------------------
low = -1e-3
high = 1e-3

word_vec_dim = 30
sentence_vec_dim = 600
