import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from subprocess import call
import os.path
import os


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def file_check(file_path): 
    path_dict = {}
    path_dict["chinese_ft_300.txt"] = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.zh.300.vec.gz'
    path_dict["vietnamese_ft_300.txt"] = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/word-vectors-v2/cc.vi.300.vec.gz'
    path_dict["english_ft_300.txt"] = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/cc.en.300.vec.gz'
    if not os.path.exists(file_path):
        print("downloading fasttext embeddings...")
        os.system('wget -cO - ' + path_dict[file_path.split("/")[-1]] + '> ' + file_path[:-3] + 'gz')
        os.system('gunzip < '+ file_path[:-3] + 'gz' + ' > ' + file_path)
    else:
        print("found existing embeddings!")
