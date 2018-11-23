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

#def showPlot(points):
#    plt.figure()
#    fig, ax = plt.subplots()
#    # this locator puts ticks at regular intervals
#    loc = ticker.MultipleLocator(base=0.2)
#    ax.yaxis.set_major_locator(loc)
#    plt.plot(points)

def showPlot(points, title, save_pth):
    plt.figure()
    fig, ax = plt.subplots(figsize=(15,10))
    # this locator puts ticks at regular intervals
    mo = len(train_loader) // print_every # how many prints in each epoch 
    loc = ticker.MultipleLocator(base=mo)
    ax.xaxis.set_major_locator(loc)
    ax.set_xbound(ax.get_xbound())
    ax.set_xticklabels(range(n_iters))
    plt.plot(points)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    fig.savefig(save_pth)

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


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power
    """
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    lr = init_lr*(1 - iter/max_iter)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
