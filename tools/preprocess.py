from __future__ import unicode_literals, print_function, division
from collections import Counter
from io import open
import numpy as np
import unicodedata
import os
import string
import re
import random
import torch
from tools.Constants import *
from tqdm import tqdm
import pickle as pkl

class Lang:
    def __init__(self, name):
        self.name = name
#         self.word2index = pre_train_w2i
#         self.index2word = pre_train_i2w
#         self.n_words = len(self.word2index)

        self.word2index = {"<PAD>": PAD, "<SOS>": SOS, "<EOS>": EOS, "<UNK>": UNK} 
        self.word2count = Counter()
        self.index2word = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        self.n_words = 4
        
    def addSentence(self, sentence):
        self.word2count.update(sentence.split(' '))

    def build_vocab(self, voc_ratio=0.9):
        max_vocab_size = len(self.word2count)
        
        vocab, count = zip(*self.word2count.most_common(int(max_vocab_size*voc_ratio)))
        self.index2word.extend(list(vocab))
        self.word2index.update(dict(zip(vocab, range(4,4+len(vocab)))))
        assert len(self.index2word) == len(self.word2index)
        self.n_words = len(self.word2index)
        
        print("There are {} unique words. Least common word count is {}. ".format(self.n_words, count[-1]))

def unicodeToAscii(s):
    """
    Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' # Nonspacing_Mark
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s, noPunc=False):
    
    s = unicodeToAscii(s.lower().strip())
    s = s.replace("&apos", "").replace("&quot","")
    if noPunc:
        s = re.sub("([.|!|?])", " ", s)
    s = re.sub("[^a-zA-Z,.!?]+", " ", s)
    
    
    return s

# read datasets
def read_data(path):
    data = []
    f = open(path,'r', encoding='utf-8')
    try:
        # for line in f:
        for line in f.readlines():
            data.append(line)   
    finally:
        f.close()  
    return data

# create Lang instances for source and target language
# create pairs
def readLangs(t, lang1, lang2, path, reverse=False):
 
    path_lang1 = "%s/iwslt-%s-%s/%s.tok.%s" % (path, lang1, lang2, t, lang1)
    path_lang2 = "%s/iwslt-%s-%s/%s.tok.%s" % (path, lang1, lang2, t, lang2)
    
    zipped = zip(read_data(path_lang1), read_data(path_lang2))
    
    pairs = []
    for source, target in zipped:
        source = source.replace("&apos", "").replace("&quot","")
        source = re.sub( '\s+', ' ', source).strip()
        # source = re.sub("([,|.|!|?])", "", source)
        
        pairs.append([source.strip(), normalizeString(target, noPunc=True).strip()])
        
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1) 
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def filterPair(p, max_length):
    return len(p[0].split(' ')) < max_length[0] and \
        len(p[1].split(' ')) < max_length[1]

def filterPairs(pairs, max_length):
    return [pair for pair in pairs if filterPair(pair, max_length)]

def prepareData(t, lang1, lang2, path="", reverse=False, max_len_ratio=0.95, voc_ratio=0.9):
    input_lang, output_lang, pairs = readLangs(t, lang1, lang2, path, reverse)
    max_length = [0, 0]
    max_length[0] = sorted([len(p[0].split(" ")) for p in pairs])[int(len(pairs) * max_len_ratio)]
    max_length[1] = sorted([len(p[1].split(" ")) for p in pairs])[int(len(pairs) * max_len_ratio)]
    print("max length of source and target", max_length)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs, max_length)
    print("Trimmed to %s sentence pairs" % len(pairs))
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
        
    input_lang.build_vocab(voc_ratio)
    output_lang.build_vocab(voc_ratio)
    if input_lang is not None:
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs, [max_length[0]+5, max_length[1]+5]


def load_fasttext_embd(fname, lang, input_lang, words_to_load=100000, emb_size=300, reload=False):
    label = lang.name+"-from-"+input_lang.name
    print(label)
    if os.path.exists(label+"pkl") and (not reload):
        embeddings, notPretrained = pkl.load(open(label+"pkl", "rb"))
        print("found existing embeddings pickles.."+fname[:-4])
    else:
        print("loading embeddings.."+fname[:-4])
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        fin.readline()
        ft_weights = np.zeros((words_to_load + 4, emb_size))
        ft_word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3} 

        for i, line in enumerate(fin):
            if i >= words_to_load:
                break
            tokens = line.rstrip().split(' ')
            if tokens[0] in ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]:
                continue
            else:
                ft_weights[i+4, :] = np.asarray(tokens[1:])
                ft_word2idx[tokens[0]] = i+4

        fin.close()    
        notPretrained = []
        embeddings = [get_pretrain_emb(ft_weights, ft_word2idx, token, notPretrained) for token in lang.index2word]
        pkl.dump([embeddings, notPretrained], open(label+"pkl", "wb"))
    print("There are {} not pretrained {} words out of {} total words.".format(sum(notPretrained), lang.name, len(notPretrained)))
    return embeddings, notPretrained

def get_pretrain_emb(pretrained, ft_word2idx, token, notPretrained):
    if token == '<pad>':
        notPretrained.append(0)
        return [0] * 300
    if token in ft_word2idx:
        notPretrained.append(0)
        return pretrained[ft_word2idx[token]]
    else:
        notPretrained.append(1)
        return [0] * 300


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] if word in lang.word2index else UNK for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):    
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS)
    return np.array(indexes)


def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)
