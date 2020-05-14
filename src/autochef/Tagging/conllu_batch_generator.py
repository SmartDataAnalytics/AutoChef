#!/usr/bin/env python3
# coding: utf-8

# # Conllu Batch Generator
# 
# read conllu documents in batches

import sys
sys.path.append('../')

from conllu import parse
from Tagging.tagging_tools import print_visualized_tags

from sklearn import preprocessing
import numpy as np


import settings  # noqa

import gzip


class ConlluSentenceIterator(object):
    def __init__(self, conllu_reader):
        self.conllu_reader = conllu_reader
        self._fileobj = None
        self._open()
    
    def _open(self):
        if self.conllu_reader._path.endswith(".gz"):
            self._fileobj = gzip.open(self.conllu_reader._path, 'r')
            self._nextline = self.read_byte_line
        else:
            self._fileobj = open(self.conllu_reader._path, 'r')
            self._nextline = self.read_str_line

    def __next__(self):
        next_sent = self.next_sentence()
        if next_sent is None:
            raise StopIteration
        return next_sent
    
    def read_str_line(self):
        return self._fileobj.readline()
    
    def read_byte_line(self):
        return self._fileobj.readline().decode("utf-8")

    def next_sentence(self):
        data = ""
        while True:
            line = self._nextline()
            if line == "":
                break
            if line == "\n" and len(data) > 0:
                break
            data += line

        if data == "":
            return None

        if data[-1] != "\n":
            data += "\n"

        conllu_obj = parse(data + "\n")
        return conllu_obj


class ConlluDocumentIterator(object):
    def __init__(self, conllu_reader, return_recipe_ids = False):
        self.conllu_reader = conllu_reader
        self._fileobj = None
        self._open()
        self._return_recipe_ids = return_recipe_ids
    
    def _open(self):
        if self.conllu_reader._path.endswith(".gz"):
            self._fileobj = gzip.open(self.conllu_reader._path, 'r')
            self._nextline = self.read_byte_line
        else:
            self._fileobj = open(self.conllu_reader._path, 'r')
            self._nextline = self.read_str_line
        
    def read_str_line(self):
        return self._fileobj.readline()
    
    def read_byte_line(self):
        return self._fileobj.readline().decode("utf-8")

    def next_document(self):
        doc_id = None
        data = ""
        last_line_empty = False
        while True:
            line = self._nextline()
            if line.startswith('#'):
                # looking for an recipe id:
                comment = line.replace('#', '')
                splitted = comment.split(':')
                if len(splitted) == 2:
                    if splitted[0].strip() == "id":
                        doc_id = splitted[1].strip()
                continue
                
            if line == "":
                break
            if line == "\n" and len(data) > 0:
                if last_line_empty:
                    break
                last_line_empty = True
            else:
                last_line_empty = False
            data += line

        if data == "":
            return None

        if data[-1] != "\n":
            data += "\n"

        conllu_obj = parse(data + "\n")
        
        if self._return_recipe_ids:
            return conllu_obj, doc_id
        return conllu_obj

    def __next__(self):
        next_sent = self.next_document()
        if next_sent is None:
            raise StopIteration
        return next_sent


class ConlluReader(object):
    def __init__(self, path, iter_documents=False, return_recipe_ids = False):
        self._path = path
        self.iter_documents = iter_documents
        self.return_recipe_ids = return_recipe_ids

    def __iter__(self):
        return ConlluDocumentIterator(self, self.return_recipe_ids) if self.iter_documents else ConlluSentenceIterator(self)


class SlidingWindowListIterator(object):
    def __init__(self, parent):
        self.parent = parent
        self.i = 0

    def __next__(self):
        if len(self.parent) == self.i:
            raise StopIteration

        self.i += 1
        return self.parent[self.i - 1]


class SlidingWindowList(list):
    def __init__(self, sliding_window_size, input=None, border_value=None):

        self.sliding_window_size = sliding_window_size
        self.border_value = border_value

        if border_value is None and input is not None:
            self.border_value = type(input[0])()

        if input is not None:
            super(SlidingWindowList, self).__init__(input)

    def __getitem__(self, index):

        if type(index) == slice:
            start = 0 if index.start is None else index.start
            stop = len(self) if index.stop is None else index.stop
            step = 1 if index.step is None else index.step
            return [self[i] for i in range(start, stop, step)]

        else:
            n = self.sliding_window_size * 2 + 1
            res = n * [self.border_value]

            j_start = index - self.sliding_window_size

            for i in range(n):
                ind = j_start + i
                if ind >= 0 and ind < len(self):
                    res[i] = super(SlidingWindowList, self).__getitem__(ind)

            return res

    def __iter__(self):
        return SlidingWindowListIterator(self)


'''
class ConlluDataProviderIterator(object):
    def __init__(self, parent):
        self.parent = parent
        self.conllu_reader = ConlluReader(
            parent.filepath, parent.iter_documents)

    def __next__(self):
        result = self.parent.getNextDataBatch(conllu_reader=self.conllu_reader)
        if result is None:
            raise StopIteration
        return result
'''

'''
class ConlluDataProvider(object):
    def __init__(self,
                 filepath,
                 word2vec_model,
                 batchsize=100,
                 window_size=3,
                 iter_documents=False,
                 food_type=None):
        self.batchsize = batchsize
        self.word2vec_model = word2vec_model
        self.filepath = filepath
        self.conllu_reader = ConlluReader(filepath, iter_documents)
        self.window_size = window_size
        self.food_type = food_type
        self.iter_documents = iter_documents

        # create a label binarizer for upos tags:
        self.lb = preprocessing.LabelBinarizer()
        self.lb.fit(['.', 'ADJ', 'ADP', 'ADV', 'CONJ', 'DET',
                     'NOUN', 'NUM', 'PRON', 'PRT', 'VERB', 'X'])

    def _get_next_conllu_objects(self, n: int, conllu_reader):
        i = 0
        conllu_list = []

        while i < n:
            try:
                conllu_list.append(conllu_reader.__iter__().__next__())
                i += 1

            except StopIteration:
                break

        return conllu_list

    def _get_upos_X(self, conllu_list):
        n_tokens = 0
        l_global = []
        for document in conllu_list:
            l = []
            for sentence in document:
                for token in sentence:
                    upos = token['upostag']
                    l.append(upos)
                    n_tokens += 1
            if len(l) > 0:
                l_global.append(self.lb.transform(l))

        return l_global, n_tokens

    def _get_y(self, conllu_list, misk_key="food_type", misc_val="ingredient"):
        n_tokens = 0
        y_global = []
        for document in conllu_list:
            y = []
            for sentence in document:
                for token in sentence:
                    m = token['misc']
                    t_y = m is not None and misk_key in m and m[misk_key] == misc_val
                    y.append(t_y)
                    n_tokens += 1
            if len(y) > 0:
                y_global.append(y)

        return y_global, n_tokens

    def getNextDataBatch(self, y_food_type_label=None, conllu_reader=None):

        if y_food_type_label is None:
            y_food_type_label = self.food_type

        if conllu_reader is None:
            conllu_reader = self.conllu_reader
        conllu_list = self._get_next_conllu_objects(
            self.batchsize, conllu_reader)

        if len(conllu_list) == 0:
            return None

        # generate features for each document/sentence
        n = len(conllu_list)

        d = self.window_size * 2 + 1

        buf_X, x_tokens = self._get_upos_X(conllu_list)
        buf_ingr_y, y_tokens = self._get_y(conllu_list)

        assert len(buf_X) == len(buf_ingr_y) and x_tokens == y_tokens

        X_upos = np.zeros(shape=(x_tokens, d * len(self.lb.classes_)))
        y = None

        if y_food_type_label is not None:
            y = np.zeros(shape=(x_tokens))

        i = 0
        for xupos in buf_X:
            tmp = SlidingWindowList(self.window_size,
                                    xupos,
                                    border_value=[0] * len(self.lb.classes_))
            for upos_window in tmp:
                X_upos[i, :] = np.array(upos_window).flatten()
                i += 1

        i = 0
        if y_food_type_label is not None:
            for sentence in buf_ingr_y:
                for yl in sentence:
                    y[i] = yl
                    i += 1

        return X_upos, y
    
    def __iter__(self):
        return ConlluDataProviderIterator(self)

'''

