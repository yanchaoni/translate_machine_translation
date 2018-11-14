import torch
from tools.preprocess import *
from tools.Dataloader import *
from models.encoder_decoder import EncoderRNN, DecoderRNN
from train import trainIters
"""
Issues: 
need to batchify: sort, pack padded seq etc.
need mask when doing attention
"""

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # pre-trained embedding 
# fasttext_chinese_embd = load_fasttext_embd('.........../chinese_ft_300.txt')
# fasttext_viet_embd = load_fasttext_embd('.........../vietnamese_ft_300.txt')

train_input_lang, train_output_lang, train_pairs, train_max_length= prepareData("train", "zh", "en", path="data")
# dev_input_lang, dev_output_lang, dev_pairs, dev_max_length = prepareData("dev", "zh", "en", path="data")
_, _, val_pairs, _ = prepareData('dev', 'zh', 'en', path="data")
_, _, test_pairs, _ = prepareData('test', 'zh', 'en', path="data")

params = {'batch_size': 16,'shuffle': False,'collate_fn': vocab_collate_func,'num_workers':20}
params2 = {'batch_size': 1,'shuffle': False,'collate_fn': vocab_collate_func,'num_workers':20}

training_set, validation_set = Dataset(train_pairs, train_input_lang, train_output_lang,tensorsFromPair), Dataset(val_pairs,train_input_lang, train_output_lang,tensorsFromPair)
train_loader = data.DataLoader(training_set, **params)
dev_loader = data.DataLoader(validation_set, **params2)

## need to add match function to load embds into lookup tables
teacher_forcing_ratio = 0.5

hidden_size = 300
encoder = EncoderRNN(train_input_lang.n_words, hidden_size).to(device)
decoder = DecoderRNN(hidden_size, train_output_lang.n_words, n_layers=1, dropout_p=0.1).to(device)
##UNCOMMENT TO TRAIN THE MODEL
trainIters(encoder, decoder,train_loader, 10, print_every=1,device=device)

#encoder.load_state_dict(torch.load("encoder.pth"))
#decoder.load_state_dict(torch.load("attn_decoder.pth"))


