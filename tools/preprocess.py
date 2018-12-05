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

    def build_vocab(self, voc_ratio=0.8):
#         max_vocab_size = len(self.word2count)
        
#         vocab, count = zip(*self.word2count.most_common(int(max_vocab_size*voc_ratio)))
        vocab = list(filter(lambda x: self.word2count[x] > 1, self.word2count))
        self.index2word.extend(vocab)
        self.word2index.update(dict(zip(vocab, range(4,4+len(vocab)))))
        assert len(self.index2word) == len(self.word2index)
        self.n_words = len(self.word2index)
        
        print("There are {} unique words. Least common word count is {}. ".format(self.n_words, 2))

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
    s = re.sub('\s+', ' ', s)
    return s

# read datasets
def read_data(path):
    data = []
    f = open(path,'r', encoding='utf-8')
    for line in f:
        data.append(line)   
    f.close()  
    return data

def char_tokenizer(sent):
    char_sent = ' '.join(list(sent))
    return char_sent

def readLangs(t, lang1, lang2, path, reverse=False, char=True):
    
    if char:
    # tokenize in chinese character level
        path_lang1 = "%s/iwslt-%s-%s/%s.%s" % (path, lang1, lang2, t, lang1) # get source sentence
        path_lang2 = "%s/iwslt-%s-%s/%s.tok.%s" % (path, lang1, lang2, t, lang2) # get target sentence
        char_lang1 = [char_tokenizer(sent) for sent in read_data(path_lang1)]
        zipped = zip(char_lang1, read_data(path_lang2))
    else:
        path_lang1 = "%s/iwslt-%s-%s/%s.tok.%s" % (path, lang1, lang2, t, lang1)
        path_lang2 = "%s/iwslt-%s-%s/%s.tok.%s" % (path, lang1, lang2, t, lang2)
        zipped = zip(read_data(path_lang1), read_data(path_lang2))
    
    pairs = []
    for source, target in zipped:
        # remove quotation marks and also remove underscore in vietnamese word
        source = source.replace("&apos", "").replace("&quot","").replace("_","") 
        source = re.sub( '\s+', ' ', source).strip()
#         source = re.sub("([,|.|!|?])", "", source)
        
        pairs.append([source.strip(), normalizeString(target, noPunc=True).strip()])
    
    # undo the first-word capitalization for Vietamese
    if lang1 == 'vi': 
        for i in range(len(pairs)):
            if pairs[i][0] is not "":
                pairs[i][0] = pairs[i][0].replace(pairs[i][0][0], pairs[i][0][0].lower())
    
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

def prepareData(t, lang1, lang2, path="", reverse=False, max_len_ratio=0.95, voc_ratio=0.9, char=True):
    input_lang, output_lang, pairs = readLangs(t, lang1, lang2, path, reverse, char)
    max_length = [0, 0]
    max_length[0] = sorted([len(p[0].split(" ")) for p in pairs])[int(len(pairs) * max_len_ratio)-1]
    max_length[1] = sorted([len(p[1].split(" ")) for p in pairs])[int(len(pairs) * max_len_ratio)-1]
    print("max length of source and target", max_length)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs, max_length)
    print("Trimmed to %s sentence pairs" % len(pairs))
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
        
    input_lang.build_vocab()
    output_lang.build_vocab()
    if input_lang is not None:
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs, [max_length[0]+5, max_length[1]+5]

def load_fasttext_embd(fname, lang, input_lang, words_to_load=100000, reload=False):
    label = lang.name+"-from-"+input_lang.name
    print(label)
    if os.path.exists(label+".pkl") and (not reload):
        data = pkl.load(open(label+".pkl", "rb"))
        print("found existing embeddings pickles.."+fname[:-4])
    else:
        print("loading embeddings.."+fname[:-4])
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        fin.readline()
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            if tokens[0] in lang.index2word:
                data[tokens[0]] = list(map(float, tokens[1:]))

        fin.close()
        pkl.dump(data, open(label+".pkl", "wb"))
    notPretrained = []
    embeddings = [get_pretrain_emb(data, token, notPretrained) for token in lang.index2word]
        
    print("There are {} not pretrained {} words out of {} total words.".format(sum(notPretrained), lang.name, len(notPretrained)))

            
    return embeddings, np.array(notPretrained) #, notin_token_lst

# load in Chinese character level embedding
def read_vectors(path):  # read top n word vectors, i.e. top is 10000
    vectors = {}

    with open(path, encoding='utf-8', errors='ignore') as f:
        f.readline()
        for line in f:
            tokens = line.rstrip().split(' ')
            if (len(tokens[0]) > 1):
                continue
            vectors[tokens[0]] = list(map(float, tokens[1:]))

    return vectors

def load_char_embd(fname, lang, reload=False):
    label = "zh_char"
    if os.path.exists(label+".pkl") and (not reload):
        data = pkl.load(open(label+".pkl", "rb"))
        print("found existing embeddings pickles.."+label+".pkl")
    else:
        print("loading embeddings..."+fname)
        data = read_vectors(fname)
        pkl.dump(data, open(label+".pkl", "wb"))
        
    notPretrained = []
    embeddings = [get_pretrain_emb(data, token, notPretrained) for token in lang.index2word]
        
    print("There are {} not pretrained {} words out of {} total words.".format(sum(notPretrained), lang.name, len(notPretrained)))
    
    return embeddings, np.array(notPretrained) 


def get_pretrain_emb(pretrained, token, notPretrained):
    if token == '<pad>':
        notPretrained.append(0)
        return [0] * 300
    if token in pretrained:
        notPretrained.append(0)
        return pretrained[token]
    else:
        notPretrained.append(1)
        return [0] * 300

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] if word in lang.word2index else UNK for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):    
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS)
    return indexes


def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)
