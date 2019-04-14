import spacy
import os
from stanfordcorenlp import StanfordCoreNLP
from nltk.stem import WordNetLemmatizer
import argparse
from abc import ABC, abstractmethod
from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from tqdm import tqdm


OIEPARSER = None
SLICE_NUM = 100

class OieParser(object):
    def __init__(self):
        pass
    
    @abstractmethod
    def parse(self, sentence_entity):
        pass
    
    def parse2str(self, sentence_entity, sep='\t'):
        return sep.join(self.parse(sentence_entity))    
    
    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()

class SpacyParser(OieParser):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def _get_path(self, subject):
        path = [subject]
        while path[-1] != path[-1].head:
            path.append(path[-1].head)
        return path

    def parse(self, sentence_entity):
        _id, sentence, entities = sentence_entity
        sentence = sentence.strip()

        entities_replace = []
        sentence_replace = sentence
        for i in range(len(entities)):
            e = entities[i].replace('_', ' ').title()
            entities_replace.append((len(e), e))
        for i, e in enumerate(sorted(entities_replace, reverse=True)):
            sentence_replace = sentence_replace.replace(entities[i], e[1])

        if not isinstance(_id, str):
            _id = 'id_' + str(_id)

        doc = self.nlp(sentence_replace)

        # entity
        entities_ = list(filter(lambda e: e.text in entities, doc.noun_chunks))
        if len(entities_) != 2:
            return []

        if entities_[0].text != entities_replace[0][1]:
            entities_ = entities_[::-1]
        entitiy_types = '-'.join(map(lambda e: (e.label_), entities_))

        # tag
        tag = ' '.join(map(lambda x: x.tag_, doc))

        # lexicalized dependency path
        left_path = self._get_path(entities_[0].root)
        right_path = self._get_path(entities_[1].root)
        min_len = min(len(left_path), len(right_path))
        root = None
        while min_len > 1 and left_path[-1] == right_path[-1]:
            # if left_path[-1].pos_ == ADP
            if left_path[-1].pos == 84: 
                break
            root = left_path[-1]
            min_len -= 1
            left_path.pop()
            right_path.pop()

        if root is None:
            return []

        # trigger
        trigger = 'TRIGGER:%s' %(root.lemma_)

        dependency_path = ''
        for x in left_path:
            dependency_path += '<-' + x.dep_
        dependency_path += '<-' + root.lemma_ + '->'
        for x in right_path[::-1]:
            dependency_path += x.dep_ + '->'
        
        return dependency_path, entities[0], entities[1], entitiy_types, trigger, _id, sentence, tag

class StanfordParser(OieParser):
    def __init__(self):
        self.nlp = StanfordCoreNLP('/home/xliucr/stanford-corenlp/stanford-corenlp-full-2018-02-27', memory='4g')
        # self.nlp = StanfordCoreNLP(r'/Users/Sean/Workspace/stanford-corenlp/stanford-corenlp-full-2018-02-27', memory='2g')
        self.wnl = WordNetLemmatizer()

    def _get_path(self, index, dependency_parse):
        path = []
        root_index = dependency_parse[0][2]
        # begin from 1
        index += 1
        if index == root_index:
            return path

        while index != root_index: 
            if index > root_index:
                index -= 1
            path.append(dependency_parse[index][0])
            index = dependency_parse[index][1]
        return path
    
    def parse(self, sentence_entity):
        try:
            _id, sentence, entities = sentence_entity
            sentence = sentence.strip()

            entities_replace = []
            sentence_replace = sentence
            for i in range(len(entities)):
                e = entities[i].replace('_', ' ').title()
                entities_replace.append((len(e), e))
            for i, e in enumerate(sorted(entities_replace, reverse=True)):
                sentence_replace = sentence_replace.replace(entities[i], e[1])
            
            if not isinstance(_id, str):
                _id = 'id_' + str(_id)

            # entity
            ner = self.nlp.ner(sentence_replace)
            ner_indice = [-1] * len(entities_replace)

            for j, x in enumerate(ner):
                for i, e in enumerate(entities_replace):           
                    if ner_indice[i] == -1 and e[1].startswith(x[0]):
                        ner_indice[i] = j

            for index in ner_indice:
                if index == -1:
                    return []
            
            entitiy_types = '-'.join(map(lambda x: (ner[x][1]), ner_indice))

            # tag
            tag = ' '.join(map(lambda x: x[1], self.nlp.pos_tag(sentence_replace)))
            # lexicalized dependency path
            dependency_parse = self.nlp.dependency_parse(sentence_replace)

            for i in range(1, len(dependency_parse)):
                # if dependency_parse[i][0] == 'ROOT':
                if dependency_parse[i][1] == 0:
                    return []

            left_path = self._get_path(ner_indice[0], dependency_parse)
            right_path = self._get_path(ner_indice[1], dependency_parse)
            

            # trigger
            root_index = dependency_parse[0][2] - 1
            root = self.wnl.lemmatize(self.wnl.lemmatize(ner[root_index][0]), 'v')
            trigger = 'TRIGGER:%s' %(root)

            dependency_path = ''
            for x in left_path:
                dependency_path += '<-' + x
            dependency_path += '<-' + root + '->'
            for x in right_path[::-1]:
                dependency_path += x + '->'
            
            return dependency_path, entities[0], entities[1], entitiy_types, trigger, _id, sentence, tag
        except:
            return []
    
    def shutdown(self):
        self.nlp.close()

def init():
    global OIEPARSER
    OIEPARSER = StanfordParser()
    Finalize(OIEPARSER, OIEPARSER.shutdown, exitpriority=100)

def parse(sentence_entity):
    global OIEPARSER
    return OIEPARSER.parse(sentence_entity)

def parse2str(sentence_entity):
    global OIEPARSER
    return OIEPARSER.parse2str(sentence_entity)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, 
                        help='Path to file to parse')
    parser.add_argument('--output', type=str, 
                        help='Path to file to save')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Num of threads to parse')
    args = parser.parse_args()

    sentence_entity = []
    """
    with open(args.input, 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.split('\t')
            sentence_entity.append((i+1, line[5], (line[2], line[3])))
    """
    for filename in ['train_entity-sentence.txt', 'valid_entity-sentence.txt', 'test_entity-sentence.txt']:
        with open(os.path.join(args.input, filename), 'r') as f:
            for i, line in enumerate(f.readlines()):
                line = line.split('\t')
                sentence_entity.append((i+1, line[2], (line[0], line[1])))

    if args.num_workers == 1:
        stanford_parser = StanfordParser()
        total = len(sentence_entity)
        for begin_index in tqdm(range(0, total, SLICE_NUM)):
            end_index = min(begin_index+SLICE_NUM, total)
            results = map(lambda s: stanford_parser.parse2str(s), sentence_entity[begin_index:end_index])
            with open(args.output, 'a+') as f:
                for line in results:
                    f.write(line + '\n')
            # print('%d/%d...' % (end_index, total))
    else:
        workers = ProcessPool(
            args.num_workers,
            initializer=init
        )

        slice_num = args.num_workers * SLICE_NUM
        total = len(sentence_entity)
        for begin_index in tqdm(range(0, total, slice_num)):
            end_index = min(begin_index+slice_num, total)
            results = workers.map_async(parse2str, sentence_entity[begin_index: end_index])
            results = list(results.get())
            with open(args.output, 'a+') as f:
                for line in results:
                    f.write(line + '\n')
            # print('%d/%d...' % (end_index, total))
