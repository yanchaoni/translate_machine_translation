from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import torch

SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def unicodeToAscii(s):
    """
    Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' # Nonspacing_Mark
    )

def normalizeString(s):
    """
     Lowercase, trim, and remove non-letter characters
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, file_name="", reverse=False):
    """
    Read the file and split into lines, 
    Split every line into pairs and normalize
    Reverse pairs, make Lang instances
    """
    print("Reading lines...")

    if file_name == "":
        file_name = 'data/%s-%s.txt' % (lang1, lang2)
    lines = open(file_name, encoding='utf-8').\
        read().strip().split('\n')

    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

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

def prepareData(lang1, lang2, file_name="", reverse=False, max_len_ratio=0.95):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, file_name, reverse)
    max_length = [0, 0]
    max_length[0] = sorted([len(p[0]) for p in pairs])[int(len(pairs) * 0.95)]
    max_length[1] = sorted([len(p[1]) for p in pairs])[int(len(pairs) * 0.95)]

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
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)