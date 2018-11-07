import torch
from utils.preprocess import *
from models.encoder_decoder import EncoderRNN, DecoderRNN
from train import trainIters
"""
Issues: 
need to batchify: sort, pack padded seq etc.
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_name = ""
input_lang, output_lang, pairs = prepareData('eng', 'fra', file_name, True)

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
# attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
attn_decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(device)

##UNCOMMENT TO TRAIN THE MODEL
trainIters(encoder1, attn_decoder1, pairs, max_length, 75000, print_every=5000)

encoder1.load_state_dict(torch.load("encoder.pth"))
attn_decoder1.load_state_dict(torch.load("attn_decoder.pth"))