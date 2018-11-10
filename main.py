import torch
from tools.preprocess import prepareData, load_fasttext_embd
from models.encoder_decoder import EncoderRNN, DecoderRNN
from train import trainIters
"""
Issues: 
need to batchify: sort, pack padded seq etc.
need mask when doing attention
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_name = "../lab8/data/eng-fra.txt"
input_lang, output_lang, pairs, max_length = prepareData('eng', 'fra', file_name, True)

# pre-trained embedding 
fasttext_chinese_embd = load_fasttext_embd('.........../chinese_ft_300.txt')
fasttext_viet_embd = load_vectors('.........../vietnamese_ft_300.txt')

## need to add match function to load embds into lookup tables

teacher_forcing_ratio = 0.5
hidden_size = 256
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
# attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
decoder = DecoderRNN(hidden_size, output_lang.n_words).to(device)

##UNCOMMENT TO TRAIN THE MODEL
trainIters(encoder, decoder, input_lang, output_lang, pairs, max_length, 75000, print_every=5000)

encoder.load_state_dict(torch.load("encoder.pth"))
decoder.load_state_dict(torch.load("attn_decoder.pth"))