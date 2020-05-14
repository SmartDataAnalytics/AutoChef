#!/usr/bin/env python3
# coding: utf-8

# # crf data Generator

import sys
sys.path.append("../")


import Tagging.conllu_batch_generator as cbg


def word2features(sent, i):
    word = sent[i]['form']
    postag = sent[i]['upostag']
    features = [
        'bias',
        #'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        word1 = sent[i-1]['form']
        postag1 = sent[i-1]['upostag']
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
        if i > 1:
            word1 = sent[i-2]['form']
            postag1 = sent[i-2]['upostag']
            features.extend([
                '-2:word.lower=' + word1.lower(),
                '-2:word.istitle=%s' % word1.istitle(),
                '-2:word.isupper=%s' % word1.isupper(),
                '-2:postag=' + postag1,
                '-2:postag[:2]=' + postag1[:2],
            ])
    else:
        features.append('BOS')

    if i < len(sent)-1:
        word1 = sent[i+1]['form']
        postag1 = sent[i+1]['upostag']
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
        if i < len(sent)-2:
            word1 = sent[i+1]['form']
            postag1 = sent[i+1]['upostag']
            features.extend([
                '+2:word.lower=' + word1.lower(),
                '+2:word.istitle=%s' % word1.istitle(),
                '+2:word.isupper=%s' % word1.isupper(),
                '+2:postag=' + postag1,
                '+2:postag[:2]=' + postag1[:2],
            ])
    else:
        features.append('EOS')

    return features


def sent2labels(sent):
    labels = []
    for token in sent:
        if token['misc'] is not None and 'food_type' in token['misc']:
            labels.append(token['misc']['food_type'])
        else:
            labels.append("0")
    return labels


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2tokens(sent):
    return [token['form'] for token in sent]


def feature2tokens(sent):
    return [t[1].split("=")[1] for t in sent]


class ConlluCRFReaderIterator(object):
    def __init__(self, parent):
        self._parent = parent
        self._iter = self._parent._conllu_reader.__iter__()

    def __next__(self):
        features = None
        labels = None
        tokens = None

        if not self._parent._iter_documents:
            next_sent = self._iter.__next__()[0]
            features = sent2features(next_sent)
            labels = sent2labels(next_sent)
            tokens = sent2tokens(next_sent)
        else:
            next_doc = self._iter.__next__()
            features = [sent2features(sentence) for sentence in next_doc]
            labels = [sent2labels(sentence) for sentence in next_doc]
            tokens = [sent2tokens(sentence) for sentence in next_doc]

        return features, labels, tokens


class ConlluCRFReader(object):
    def __init__(self, path, iter_documents=False):
        self._path = path
        self._iter_documents = iter_documents

        self._conllu_reader = cbg.ConlluReader(path, iter_documents)

    def __iter__(self):
        return ConlluCRFReaderIterator(self)




