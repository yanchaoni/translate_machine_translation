import torch
from tools.preprocess import prepareData, load_fasttext_embd
from tools.Dataloader import *
from models.encoder_decoder import EncoderRNN, DecoderRNN
from train import trainIters
"""
Issues: 
need mask when doing attention
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_name = "../lab8/data/eng-fra.txt"
input_lang, output_lang, pairs, max_length = prepareData('eng', 'fra', file_name, True)

# pre-trained embedding 
fasttext_chinese_embd = load_fasttext_embd('.........../chinese_ft_300.txt')
fasttext_viet_embd = load_fasttext_embd('.........../vietnamese_ft_300.txt')

train_input_lang, train_output_lang, train_pairs, train_max_length = prepareData("train", "zh", "en", path="data")
dev_input_lang, dev_output_lang, dev_pairs, dev_max_length = prepareData("dev", "zh", "en", path="data")
params = {'batch_size': 16,'shuffle': False,'collate_fn': vocab_collate_func,'num_workers':1}
params2 = {'batch_size': 1,'shuffle': False,'collate_fn': vocab_collate_func,'num_workers':1}

training_set, validation_set = Dataset(train_pairs, train_input_lang, train_output_lang), Dataset(val_pairs,val_input_lang,avl_output_lang)
train_loader = data.DataLoader(training_set, **params)
dev_loader = data.DataLoader(validation_set, **params)



## need to add match function to load embds into lookup tables

teacher_forcing_ratio = 0.5

hidden_size = 300
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = DecoderRNN(hidden_size, output_lang.n_words, n_layers=1, dropout_p=0.1).to(device)

##UNCOMMENT TO TRAIN THE MODEL
trainIters(encoder, decoder,training_generator, 350, print_every=10)

#encoder.load_state_dict(torch.load("encoder.pth"))
#decoder.load_state_dict(torch.load("attn_decoder.pth"))


