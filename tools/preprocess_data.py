#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 16:43:22 2018

@author: leah
"""

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 50

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    
    s = unicodeToAscii(s.lower().strip())
#    s = re.sub(r"([.|!|?])", r" \1", s)
    s = re.sub(r"([.|!|?])", r" ", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
#    s = re.sub(r'"', '', s) # remove quotes
#    s = re.sub(r"'", '', s) # remove quotes
    s = s.replace("apos", "").replace("quot","")
    
    return s

# read datasets
def read_data(path):
    data = []
    f = open(path,'r')
    try:
#        for line in f:
        for line in f.readlines():
            data.append(line)   
    finally:
        f.close()  
    return data

# create Lang instances for source and target language
# create pairs
def readLangs(t, lang1, lang2, reverse=False):
    
    print("Reading lines...")
   
    path_lang1 = "/Users/leah/Desktop/MT/data/iwslt-%s-%s/%s.tok.%s" % (lang2, lang1, t, lang1)
    path_lang2 = "/Users/leah/Desktop/MT/data/iwslt-%s-%s/%s.tok.%s" % (lang2, lang1, t, lang2)
    
    zipped = zip(read_data(path_lang1), read_data(path_lang2))
    
    print("Creating pairs...")
    
    pairs = []
    for target, source in zipped:
        source = re.sub(r"([,|.|!|?])", r" ", source)
        source = source.replace("apos", "").replace("quot","")
        pairs.append([normalizeString(target).strip('\n'), source.strip('\n')])
        
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2) # lang2 = 'zh' or 'vi'
        output_lang = Lang(lang1) # lang1 = 'en'
    else:
        input_lang = Lang(lang1) 
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


# filter out sentences with too long length
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(dataset, lang1, lang2, reverse=False):
    
    input_lang, output_lang, pairs = readLangs(dataset, lang1, lang2, reverse)
    
    print("Read %s sentence pairs" % len(pairs))
    
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('train', 'en', 'vi', True)
print(random.choice(pairs))

#input_lang, output_lang, pairs = prepareData('dev', 'en', 'zh', True)
#print(random.choice(pairs))


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


n_iters = 75000
training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]


# batchify / DataLoader





