#!/usr/bin/env python3
# coding: utf-8

# # Conllu Generator
# 
# tools for creating:
# * conllu tokens
# * conllu sentences
# * conllu documents

# ## imports and settings

import sys
sys.path.append("../")

import nltk
from nltk.tag import pos_tag, map_tag
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords as nltk_stopwords
from Tagging.stemmed_mwe_tokenizer import StemmedMWETokenizer
from nltk.stem import WordNetLemmatizer


CONLLU_ATTRIBUTES = [
    "id",
    "form",
    "lemma",
    "upos",
    "xpos",
    "feats",
    "head",
    "deprel",
    "deps",
    "misc"
]


# * default stemming and lemmatization functions

porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()

def stem(token, stemmer = porter_stemmer):
    return stemmer.stem(token)

def lemmatize(token, lemmatizer = wordnet_lemmatizer, pos = 'n'):
    return lemmatizer.lemmatize(token, pos)


# took from: https://stackoverflow.com/a/16053211


def replace_tab(s, tabstop=4):
    result = str()
    s = s.replace("\t", " \t")
    for c in s:
        if c == '\t':
            while (len(result) % (tabstop) != 0):
                result += ' '
        else:
            result += c
    return result


# ## Conllu Dict Class

class ConlluDict(dict):

    def from_str(self, s: str):
        entries = s.split("|")
        for entry in entries:
            key, val = entry.split("=")
            self[key.strip()] = val.strip()

    def __repr__(self):
        if len(self) == 0:
            return "_"

        result = ""
        for key, value in self.items():
            result += key + "=" + value + "|"

        return result[:-1]

    def __str__(self):
        return self.__repr__()


# ## Conllu Element Class

class ConlluElement(object):
        # class uses format described here: https://universaldependencies.org/format.html
    def __init__(
            self,
            id: int,
            form: str,
            lemma: str = "_",
            upos: str = "_",
            xpos: str = "_",
            feats: str = "_",
            head: str = "_",
            deprel: str = "_",
            deps: str = "_",
            misc: str = "_"):
        self.id = id
        self.form = form
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos

        self.feats = ConlluDict()
        if feats != "_":
            self.feats.from_str(feats)

        self.head = head
        self.deprel = deprel
        self.deps = deps

        self.misc = ConlluDict()
        if misc != "_":
            self.misc.from_str(misc)

    def add_feature(self, key: str, value: str):
        self.feats[key] = value

    def add_misc(self, key: str, value: str):
        self.misc[key] = value

    def __repr__(self):
        result = ""
        for attr in CONLLU_ATTRIBUTES:
            result += str(self.__getattribute__(attr)) + " \t"
        return replace_tab(result, 16)
    
    def __getitem__(self, key):
        
        # conllu module compability:
        if key == "upostag":
            key = "upos"
        if key == "xpostag":
            key = "xpos"
        
        if key not in CONLLU_ATTRIBUTES:
            return None
        attr = self.__getattribute__(key)
        if str(attr) == "_":
            return None
        return attr


# ## Conllu Sentence Class

class ConlluSentence(object):
    def __init__(self):
        self.conllu_elements = []

    def add(self, conllu_element: ConlluElement):
        self.conllu_elements.append(conllu_element)
    
    def get_conllu_elements(self):
        return self.conllu_elements

    def __repr__(self):
        result = ""
        for elem in self.conllu_elements:
            result += elem.__repr__() + "\n"

        return result

    def __str__(self):
        return self.__repr__()


# ## Conllu Document Class

class ConlluDocument(object):
    def __init__(self, id=None):
        self.conllu_sentences = []
        self.id = id
    
    def add(self, conllu_sentence: ConlluSentence):
        self.conllu_sentences.append(conllu_sentence)
    
    def get_conllu_elements(self):
        return [c_sent.get_conllu_elements() for c_sent in self.conllu_sentences]
    
    def __repr__(self):
        result = "# newdoc\n"
        if self.id is not None:
            result += "# id: " + self.id + "\n"
        for elem in self.conllu_sentences:
            result += elem.__repr__() + "\n"

        return result

    def __str__(self):
        return self.__repr__()


# ## Conllu Generator Class

class ConlluGenerator(object):
    def __init__(self, documents: list, stemmed_multi_word_tokens=None, stemmer=PorterStemmer(), ids=None):
        self.documents = documents
        self.stemmed_multi_word_tokens = stemmed_multi_word_tokens
        
        if self.stemmed_multi_word_tokens is not None:
            self.mwe_tokenizer = StemmedMWETokenizer(
                [w.split() for w in stemmed_multi_word_tokens])
        else:
            self.mwe_tokenizer = None
        
        self.stemmer = stemmer

        self.conllu_documents = []

        self.ids = ids
    
    def tokenize(self):
        tokenized_documents = []

        i = 0
        for doc in self.documents:
            tokenized_sentences = []
            sentences = doc.split("\n")
            for sent in sentences: 
                if (len(sent) > 0):
                    simple_tokenized = nltk.tokenize.word_tokenize(sent)
                    if self.mwe_tokenizer is None:
                        tokenized_sentences.append(simple_tokenized)
                    else:
                        tokenized_sentences.append(
                            self.mwe_tokenizer.tokenize(simple_tokenized))
            tokenized_documents.append(tokenized_sentences)
        
        # now create initial colln-u elemnts
        for doc in tokenized_documents:
            if self.ids:
                conllu_doc = ConlluDocument(self.ids[i])
            else:
                conllu_doc = ConlluDocument()
            for sent in doc:
                token_id = 0
                conllu_sent = ConlluSentence()
                for token in sent:
                    token_id += 1
                    conllu_sent.add(ConlluElement(
                        id=token_id,
                        form=token,
                    ))
                conllu_doc.add(conllu_sent)
            self.conllu_documents.append(conllu_doc)
            i += 1


    def pos_tagging_and_lemmatization(self, stem_function = lemmatize):
        pos_dict = {'ADJ': 'a', 'ADJ_SAT': 's', 'ADV': 'r', 'NOUN': 'n', 'VERB': 'v'}
        for conllu_document in self.conllu_documents:
            for conllu_sent in conllu_document.conllu_sentences:
                tokens = [x.form for x in conllu_sent.conllu_elements]
                pos_tags = pos_tag(tokens)
                simplified_tags = [map_tag('en-ptb', 'universal', tag)
                                for word, tag in pos_tags]

                for i in range(len(tokens)):
                    conllu_elem = conllu_sent.conllu_elements[i]
                    conllu_elem.upos = simplified_tags[i]
                    conllu_elem.xpos = pos_tags[i][1]
                    p = 'n'
                    if conllu_elem.upos in pos_dict:
                        p = pos_dict[conllu_elem.upos]
                    conllu_elem.lemma = stem_function(conllu_elem.form, pos=p).lower()

    def add_misc_value_by_list(self, key, value, stemmed_keyword_list):
        for conllu_document in self.conllu_documents:
            for conllu_sent in conllu_document.conllu_sentences:
                for elem in conllu_sent.conllu_elements:
                    if elem.lemma in stemmed_keyword_list:
                        elem.add_misc(key, value)
    
    def get_conllu_elements(self):
        return [doc.get_conllu_elements() for doc in self.conllu_documents]

    def __repr__(self):
        result = ""
        for document in self.conllu_documents:
            result += document.__repr__() + "\n"
        return result

    def __str__(self):
        return self.__repr__()

