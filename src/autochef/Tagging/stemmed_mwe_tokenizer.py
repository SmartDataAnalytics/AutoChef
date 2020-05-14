#!/usr/bin/env python3

import nltk
from nltk import PorterStemmer

from nltk.util import Trie

# modified MWE Tokenizer which stems multi word expressions before the merge check


class StemmedMWETokenizer(nltk.tokenize.api.TokenizerI):
    def __init__(self, stemmed_tokens, stemmer=PorterStemmer(), separator="_"):
        self.stemmer = stemmer
        self.stemmed_tokens = stemmed_tokens
        self.mwes = Trie(stemmed_tokens)
        self.separator = separator

    def tokenize(self, text):
        """

        :param text: A list containing tokenized text
        :type text: list(str)
        :return: A list of the tokenized text with multi-words merged together
        :rtype: list(str)

        :Example:

        >>> tokenizer = MWETokenizer([('hors', "d'oeuvre")], separator='+')
        >>> tokenizer.tokenize("An hors d'oeuvre tonight, sir?".split())
        ['An', "hors+d'oeuvre", 'tonight,', 'sir?']

        """
        i = 0
        n = len(text)
        result = []

        while i < n:
            if self.stemmer.stem(text[i]) in self.mwes:
                # possible MWE match
                j = i
                trie = self.mwes
                while j < n and self.stemmer.stem(text[j]) in trie:
                    trie = trie[self.stemmer.stem(text[j])]
                    j = j + 1
                else:
                    if Trie.LEAF in trie:
                        # success!
                        result.append(self.separator.join(text[i:j]))
                        i = j
                    else:
                        # no match, so backtrack
                        result.append(text[i])
                        i += 1
            else:
                result.append(text[i])
                i += 1

        return result
