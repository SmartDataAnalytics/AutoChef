#!/usr/bin/env python3

import numpy as np
import json

import nltk
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords as nltk_stopwords

from pprint import pprint

from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec, KeyedVectors

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from json_buffered_reader import JSON_buffered_reader as JSON_br

import pandas as pd

import settings

from ipypb import track
from IPython.display import HTML, Markdown


# loading learned wordvectors
wv = KeyedVectors.load("data/wordvectors.kv")
porter = PorterStemmer()


def word_similarity(word_a: str, word_b: str, model=wv, stemmer=porter):
    return model.similarity(stemmer.stem(word_a), stemmer.stem(word_b))


def word_exists(word: str, model=wv, stemmer=porter):
    return stemmer.stem(word) in model

from cooking_vocab import cooking_verbs
from cooking_ingredients import ingredients

model_actions = []
model_ingredients = []

for action in cooking_verbs:
    if word_exists(action):
        model_actions.append(action)

for ingredient in ingredients:
    if word_exists(ingredient):
        model_ingredients.append(ingredient)

def tsne_plot(tokens, model=wv, dist_token=None):
    vecs = []
    labels = []
    for token in tokens:
        vecs.append(model[token])
        labels.append(token)

    tsne_model = TSNE(perplexity=40, n_components=2,
                      init='pca', n_iter=2500, random_state=23)
    plot_values = tsne_model.fit_transform(vecs)

    distances = []

    min_size = 10
    max_size = 500

    if dist_token is not None:
        distances = np.array([model.similarity(t, dist_token) for t in tokens])
        # scale:
        min_s = np.min(distances)
        max_s = np.max(distances)
        distances = min_size + (distances - min_s) * ((max_size - min_size) / (max_s - min_s))


    x = []
    y = []
    for value in plot_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        if dist_token is None:
            plt.scatter(x[i], y[i])
        else:
            plt.scatter(x[i], y[i], s=distances[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


stemmed_ingredients = [porter.stem(ing) for ing in model_ingredients]
stemmed_actions = [porter.stem(act) for act in model_actions]