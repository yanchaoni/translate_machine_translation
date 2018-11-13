from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import torch
import numpy as np
from tools.Constants import EOS, DEVICE

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "UNK"}
        self.n_words = 4  # Count PAD, UNK, SOS and EOS

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
def readLangs(t, lang1, lang2, path, reverse=False):
    
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

def prepareData(t, lang1, lang2, path="", reverse=False, max_len_ratio=0.95):
    input_lang, output_lang, pairs = readLangs(t, lang1, lang2, path, reverse)
    max_length = [0, 0]
    max_length[0] = sorted([len(p[0].split(" ")) for p in pairs])[int(len(pairs) * max_len_ratio)]
    max_length[1] = sorted([len(p[1].split(" ")) for p in pairs])[int(len(pairs) * max_len_ratio)]

    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs, max_length)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs, max_length

def load_fasttext_embd(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in tqdm(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

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
