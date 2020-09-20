# -*- coding: utf-8 -*-

"""
Language-related data loading helper functions and class wrappers.
"""

import re
import torch
import codecs
import spacy
import pickle
import numpy as np

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
END_TOKEN = '<eos>'
START_TOKEN = '<go>'
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def __getitem__(self, a):
        if isinstance(a, int):
            return self.idx2word[a]
        elif isinstance(a, list):
            return [self.idx2word[x] for x in a]
        elif isinstance(a, str):
            return self.word2idx[a]
        else:
            raise TypeError("Query word/index argument must be int or str")

    def __contains__(self, word):
        return word in self.word2idx


class Corpus(object):
    def __init__(self, glove):
        self.dictionary = Dictionary()
        self.nlp = spacy.load("en_core_web_sm")
        self.glove = glove
    def set_max_len(self, value):
        self.max_len = value

    def load_file(self, filename):
        self.dictionary.add_word(PAD_TOKEN)
        self.dictionary.add_word(START_TOKEN)
        self.dictionary.add_word(END_TOKEN)
        self.dictionary.add_word(UNK_TOKEN)
        with codecs.open(filename, 'r', 'utf-8') as f:
            for line in f:
                line = line.strip()
                self.add_to_corpus(line)

    def get_glove_embed(self):
        self.glove_embed = np.zeros((len(self.dictionary), 300))
        words_found = 0
        for w, i in self.dictionary.word2idx.items():
            try :
                self.glove_embed[i] = self.glove[w]
                words_found += 1
            except KeyError:
                self.glove_embed[i] = np.random.normal(scale= .5, size= (300))
        print(f"{words_found} words found in GloVe Embeddings out of {len(self.dictionary)} words in vocabulary.")
        del self.glove
        return self.glove_embed

    def add_to_corpus(self, line):
        """Tokenizes a text line."""
        # Add words to the dictionary
        # words = line.split()
        words = [x.text for x in self.nlp(line)]

        # tokens = len(words)
        for word in words:
            word = word.lower()
            self.dictionary.add_word(word)

    def tokenize(self, line, max_len=40):
        # Tokenize line contents
        words = [x.text for x in self.nlp(line.strip())]
        # words = [w.lower() for w in words if len(w) > 0]
        words = [w.lower() for w in words if (len(w) > 0 and w!=' ')]   ## do not include space as a token

        if words[-1] == '.':
            words = words[:-1]

        if max_len > 0:
            if len(words) > max_len:
                words = words[:max_len]
            elif len(words) < max_len:
                # words = [PAD_TOKEN] * (max_len - len(words)) + words
                words = words + [END_TOKEN] + [PAD_TOKEN] * (max_len - len(words) - 1)

        tokens = len(words) ## for end token
        ids = torch.LongTensor(tokens)
        token = 0
        for word in words:
            if word not in self.dictionary:
                word = UNK_TOKEN
            # print(word, type(word), word.encode('ascii','ignore').decode('ascii'), type(word.encode('ascii','ignore').decode('ascii')))
            if type(word)!=type('a'):
                print(word, type(word), word.encode('ascii','ignore').decode('ascii'), type(word.encode('ascii','ignore').decode('ascii')))
                word = word.encode('ascii','ignore').decode('ascii')
            ids[token] = self.dictionary[word]
            token += 1
        # ids[token] = self.dictionary[END_TOKEN]
        return ids

    def __len__(self):
        return len(self.dictionary)
