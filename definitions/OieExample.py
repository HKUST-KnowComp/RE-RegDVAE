from recordclass import recordclass

__author__ = 'yan'


class OieExample(object):
    """
    A container for a sentence: entities,k list of feature ids, trigger, (optional) relation label
    """

    def __init__(self, arg1, arg2, features, trigger, relation=''):
        self.features = features
        self.arg1 = arg1  # str
        self.arg2 = arg2  # str
        self.relation = relation
        self.trigger = trigger

    def setFeatures(self, features):
        self.features = features

RawDataExampleDType = [
    ('id', object),
    ('dependency_path', object),
    ('arg1', object),
    ('arg2', object),
    ('entity_types', object),
    ('trigger', object),
    ('sid', object),
    ('sentence', object),
    ('pos', object),
    ('relation', object),
]

DataExample = recordclass("DataExample", ["id", "arg1", "arg2", "feats", "relation", "neg", "trigger"])
DataExampleDType = [
    ('id', int),
    ('arg1', int, (1,)),
    ('arg2', int, (1,)),
    ('feats', list),
    ('relation', object),
    ('neg', list),
    ('trigger', str)
]

DataSet = recordclass("DataSet", ["train", "valid", "test"])  # list[DataExample]

AllData = recordclass("AllData", ['feature_extrs',
                                  'dataset',
                                  'feature_lexicon',
                                  'entity_lexicon',
                                  'relation_lexicon',
                                  ])
"""
Tr1: train ext KB
Tr2: train oie
Te1: dev oie
Te2: test oie
"""
HoldoutDataset = recordclass("HoldoutDataset",
                             ["holdout_relations", "trans_rllex", "trans_enlex", "Tr1", "Tr2", "Te1", "Te2"])


