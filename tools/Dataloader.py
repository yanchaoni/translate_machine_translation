from torch.utils import data
from tools.preprocess import *
import torch.nn.utils.rnn as rnn
from tools.Constants import MAX_WORD_LENGTH, PAD
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, pairs, input_lang, output_lang):
        self.ind_dec_order = self.desc_pairs(pairs)
        self.pairs = pairs
        self.input_lang = input_lang
        self.output_lang = output_lang
    
    def __len__(self):
        return len(self.pairs)   

    def __getitem__(self, index):
        # Select sample
        pair = self.pairs[self.ind_dec_order[index]]
        tensors = tensorsFromPair(pair, self.input_lang, self.output_lang)
        return (tensors[0], tensors[1], len(tensors[0]), len(tensors[1]))
    
    def desc_pairs(self, pairs):
        lengths = [len(pair[1].split(" ")) for pair in pairs]
        ind_dec_order = np.argsort(lengths)[::-1]
        return ind_dec_order


def vocab_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    data_list1 = []
    data_list2 = []
    length_list1 = []
    length_list2 = []
    for datum in batch:
        length_list1.append(datum[2])
        length_list2.append(datum[3])
    # padding
    max_word_length = [max(length_list1), max(length_list2)]
    for datum in batch:
        padded_vec1 = np.pad(np.array(datum[0]),
                                pad_width=((0, max_word_length[0]-datum[2])),
                                mode="constant", constant_values=PAD)
        data_list1.append(padded_vec1)
        padded_vec2 = np.pad(np.array(datum[1]),
                                pad_width=((0, max_word_length[1]-datum[3])),
                                mode="constant", constant_values=PAD)
        data_list2.append(padded_vec2)
    ind_dec_order = np.argsort(length_list1)[::-1]
    data_list1 = np.array(data_list1)[ind_dec_order]
    data_list2 = np.array(data_list2)[ind_dec_order]
    length_list1 = np.array(length_list1)[ind_dec_order]
    length_list2 = np.array(length_list2)[ind_dec_order]
    return [torch.from_numpy(np.array(data_list1)), \
            torch.from_numpy(np.array(data_list2)), \
            torch.LongTensor(length_list1), \
            torch.LongTensor(length_list2)]
