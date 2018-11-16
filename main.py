from models.encoder_decoder import EncoderRNN, DecoderRNN
import os.path
import os
import torch
import tools.Constants as Constants
from tools.Dataloader import *
from tools.helper import *
from tools.preprocess import *
from train import trainIters


"""
Issues: 
need mask when doing attention
"""
data_path = "/scratch/yn811/MT_data"
fname = "" # emb_fname
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
teacher_forcing_ratio = 0.5
words_to_load = 1000
hidden_size = 300
n_iters = 2


# pre-trained embedding 
file_check('/scratch/yn811/chinese_ft_300.txt')
# file_check('/scratch/yn811/vietnamese_ft_300.txt')
file_check('/scratch/yn811/english_ft_300.txt')

source_embedding, ft_word2idx, ft_idx2word = load_fasttext_embd('/scratch/yn811/chinese_ft_300.txt', words_to_load=words_to_load, emb_size=300)
pre_trained_lang1 = [ft_word2idx, ft_idx2word]
target_embedding, ft_word2idx, ft_idx2word = load_fasttext_embd('/scratch/yn811/english_ft_300.txt', words_to_load=words_to_load, emb_size=300)
pre_trained_lang2 = [ft_word2idx, ft_idx2word]

input_lang, output_lang, train_pairs, train_max_length = prepareData("train", "zh", "en", data_path, pre_trained_lang1, pre_trained_lang2)
# dev_input_lang, dev_output_lang, dev_pairs, dev_max_length = prepareData("dev", "zh", "en", path=data_path)
_, _, dev_pairs, _ = prepareData('dev', 'zh', 'en', path=data_path)
# _, _, test_pairs, _ = prepareData('test', 'zh', 'en', path=data_path)

params = {'batch_size': 16,'shuffle': False,'collate_fn': vocab_collate_func,'num_workers':20}
params2 = {'batch_size': 1,'shuffle': False,'collate_fn': vocab_collate_func,'num_workers':20}

train_set, dev_set = Dataset(train_pairs, input_lang, output_lang,tensorsFromPair), Dataset(dev_pairs,input_lang, output_lang,tensorsFromPair)
train_loader = torch.utils.data.DataLoader(train_set, **params)
dev_loader = torch.utils.data.DataLoader(dev_set, **params2)


encoder = EncoderRNN(input_lang.n_words, Constants.EMB_DIM, hidden_size, source_embedding).to(device)
decoder = DecoderRNN(output_lang.n_words, Constants.EMB_DIM, hidden_size, target_embedding, n_layers=2, dropout_p=0.1).to(device)

trainIters(encoder, decoder, train_loader, dev_loader, \
            input_lang, output_lang, \
            n_iters, print_every=1000, plot_every=100, \
            learning_rate=0.01, device=DEVICE, teacher_forcing_ratio=0.5)

#encoder.load_state_dict(torch.load("encoder.pth"))
#decoder.load_state_dict(torch.load("attn_decoder.pth"))


