from models.encoder_decoder import *
import os.path
import os
import torch
from tools.Constants import *
from tools.Dataloader import *
from tools.helper import *
from tools.preprocess import *
from train import trainIters


"""
Issues:
need mask when doing attention
"""
data_path = "/scratch/yn811/MT_data"
FT_emb_path = '/scratch/yn811/'
plot_save_path = "results/"
save_result_path = "results/"
save_model_name = "attn"
language = "zh"
device = DEVICE
print(device)
teacher_forcing_ratio = 0.5
max_len_ratio = 0.9
source_words_to_load = 1000000
target_words_to_load = 1000000
encoder_hidden_size = 150
decoder_hidden_size = 150
encoder_layers = 1
decoder_layers = 1
plot_every = 100000
teacher_forcing_ratio = 0.5
learning_rate = 0.001
n_iters = 200
beam_width=10
min_len=5
n_best=5
use_bi = True
decode_method = "beam"
decoder_type = "attn"


input_lang, output_lang, train_pairs, train_max_length = prepareData("train", language, "en", data_path, max_len_ratio=max_len_ratio)
_, _, dev_pairs, _ = prepareData('dev', language, 'en', path=data_path, max_len_ratio=0.99999)
# _, _, test_pairs, _ = prepareData('test', language, 'en', path=data_path)
if language == "zh":
    file_check(FT_emb_path+'chinese_ft_300.txt')
    source_embedding, source_notPretrained = load_fasttext_embd(FT_emb_path+'chinese_ft_300.txt', input_lang, input_lang, source_words_to_load)
else:
    file_check(FT_emb_path+'vietnamese_ft_300.txt')
    source_embedding, source_notPretrained = load_fasttext_embd(FT_emb_path+'vietnamese_ft_300.txt', input_lang, input_lang, source_words_to_load)

file_check(FT_emb_path+'english_ft_300.txt')
target_embedding, target_notPretrained = load_fasttext_embd(FT_emb_path+'english_ft_300.txt', output_lang, input_lang, target_words_to_load)

params = {'batch_size':BATCH_SIZE, 'shuffle':False, 'collate_fn':vocab_collate_func, 'num_workers':20}
params2 = {'batch_size':BATCH_SIZE, 'shuffle':False, 'collate_fn':vocab_collate_func, 'num_workers':20}

train_set, dev_set = Dataset(train_pairs, input_lang, output_lang), Dataset(dev_pairs, input_lang, output_lang)
train_loader = torch.utils.data.DataLoader(train_set, **params)
dev_loader = torch.utils.data.DataLoader(dev_set, **params2)

encoder = EncoderRNN(input_lang.n_words, EMB_DIM, encoder_hidden_size,
                     encoder_layers, decoder_hidden_size, source_embedding, use_bi, device).to(device)
if decoder_type == "basic":
    decoder = DecoderRNN(output_lang.n_words, EMB_DIM, decoder_hidden_size,
                         decoder_layers, target_embedding, dropout_p=0.1, device=device).to(device)
elif decoder_type == "attn":
    decoder = DecoderRNN_Attention(output_lang.n_words, EMB_DIM, decoder_hidden_size,
                                   decoder_layers, target_embedding, dropout_p=0.1, device=device).to(device)
else:
    raise ValueError

print(encoder, decoder)
trainIters(encoder, decoder, train_loader, dev_loader, \
            input_lang, output_lang, train_max_length, \
            n_iters, plot_every=plot_every, \
            learning_rate=learning_rate, device=device, teacher_forcing_ratio=teacher_forcing_ratio, label=save_model_name,
            use_lr_scheduler = True, gamma_en = 0.99, gamma_de = 0.99, 
            beam_width=beam_width, min_len=min_len, n_best=n_best, decode_method=decode_method, save_result_path = save_result_path)

showPlot(plot_losses, 'Train_Loss_Curve', plot_save_path)
#encoder.load_state_dict(torch.load("encoder.pth"))
#decoder.load_state_dict(torch.load("attn_decoder.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--device', type=str, action='store', help='use cuda?', default='cuda')
    parser.add_argument('--model_path', required=False, help='path to save model', default='./checkpoint')
    parser.add_argument('--epoch', type=int, action='store', help='number of epoches to train', default=50)
    parser.add_argument('--batch_size', type=int, action='store', help='batch size', default=32)
    parser.add_argument('--model', type=str, action='store', help='model to use', default='vanilla')
    parser.add_argument('--resume', type=str, action='store', help='model path to resume', default='False')
    parser.add_argument('--dataset', type=str, action='store', help='dataset to train on', default="../data/fl_processed_data_train.pickle")

    args = parser.parse_args()
    print(args)
    main(args)
