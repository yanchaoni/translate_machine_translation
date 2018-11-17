from __future__ import unicode_literals, print_function, division
from io import open
import numpy as np
import unicodedata
import string
import re
import random
import torch
from tools.Constants import *
from tqdm import tqdm

class Lang:
    def __init__(self, name, pre_train_w2i, pre_train_i2w):
        self.name = name
        self.word2index = pre_train_w2i
        self.index2word = pre_train_i2w
        self.n_words = len(self.word2index)
        assert len(self.word2index) == len(self.index2word)
#        self.word2index = {}
#        self.word2count = {}
#        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}
#        self.n_words = 4  # Count PAD, UNK, SOS and EOS

#    def addSentence(self, sentence):
#        for word in sentence.split(' '):
#            self.addWord(word)
#
#    def addWord(self, word):
#        if word not in self.word2index:
#            self.word2index[word] = self.n_words
#            self.word2count[word] = 1
#            self.index2word[self.n_words] = word
#            self.n_words += 1
#        else:
#            self.word2count[word] += 1


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
#    s = re.sub(r'"', '', s) # remove quotes
#    s = re.sub("'", '', s) # remove quotes
    
    
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
def readLangs(t, lang1, lang2, path, pre_trained_lang1=None, pre_trained_lang2=None, reverse=False):

    print("Reading lines...")
   
    path_lang1 = "%s/iwslt-%s-%s/%s.tok.%s" % (path, lang1, lang2, t, lang1)
    path_lang2 = "%s/iwslt-%s-%s/%s.tok.%s" % (path, lang1, lang2, t, lang2)
    
    zipped = zip(read_data(path_lang1), read_data(path_lang2))
    
    print("Creating pairs...")
    
    pairs = []
    for source, target in zipped:
        source = source.replace("&apos", "").replace("&quot","")
        source = re.sub( '\s+', ' ', source).strip()
        # source = re.sub("([,|.|!|?])", "", source)
        
        pairs.append([source.strip(), normalizeString(target).strip()])
        
    # Reverse pairs, make Lang instances
    if pre_trained_lang1 is None:
        input_lang = output_lang = None
    else:
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang(lang2, *pre_trained_lang2)
            output_lang = Lang(lang1, *pre_trained_lang1)
        else:
            input_lang = Lang(lang1, *pre_trained_lang1) 
            output_lang = Lang(lang2, *pre_trained_lang2)

    return input_lang, output_lang, pairs

def filterPair(p, max_length):
    return len(p[0].split(' ')) < max_length[0] and \
        len(p[1].split(' ')) < max_length[1]

def filterPairs(pairs, max_length):
    return [pair for pair in pairs if filterPair(pair, max_length)]

def prepareData(t, lang1, lang2, path="", pre_trained_lang1=None, pre_trained_lang2=None, reverse=False, max_len_ratio=0.95):
    input_lang, output_lang, pairs = readLangs(t, lang1, lang2, path, pre_trained_lang1, pre_trained_lang2, reverse)
    max_length = [0, 0]
    max_length[0] = sorted([len(p[0].split(" ")) for p in pairs])[int(len(pairs) * max_len_ratio)]
    max_length[1] = sorted([len(p[1].split(" ")) for p in pairs])[int(len(pairs) * max_len_ratio)]
    print("max length of source and target", max_length)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs, MAX_WORD_LENGTH)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
#    for pair in pairs:
#        input_lang.addSentence(pair[0])
#        output_lang.addSentence(pair[1])
    print("Counted words:")
    if input_lang is not None:
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs, max_length


def load_fasttext_embd(fname, words_to_load=100000, emb_size=300):
    print("loading embeddings.."+fname[:-4])
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    fin.readline()
    ft_weights = np.zeros((words_to_load + 4, emb_size))
    ft_word2idx = {"PAD": 0, "SOS": 1, "EOS": 2, "UNK": 3} 
    ft_idx2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"} 
    
    for i, line in enumerate(fin):
        if i >= words_to_load:
            break
        tokens = line.rstrip().split(' ')
        if tokens[0] in ["PAD", "SOS", "EOS", "UNK"]:
            continue
        else:
            ft_weights[i+4, :] = np.asarray(tokens[1:])
            ft_word2idx[tokens[0]] = i+4
            ft_idx2word[i+4] = tokens[0]
    
    fin.close()    
    return ft_weights, ft_word2idx, ft_idx2word


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] if word in lang.word2index else UNK for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS)
    return np.array(indexes)
#     indexes = indexesFromSentence(lang, sentence)
#     indexes.append(EOS)
#     return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)
