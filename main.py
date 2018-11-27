from models.encoder_decoder import EncoderRNN, DecoderRNN
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
plot_save_path = "results/"
SAVE_RESULT_PATH = "results/"
device = DEVICE
print(device)
teacher_forcing_ratio = 0.5
source_words_to_load = 1000000
target_words_to_load = 1000000
encoder_hidden_size = 150
decoder_hidden_size = 150
maxout_size = 300
encoder_layers = 1
decoder_layers = 1
plot_every = 100
teacher_forcing_ratio = 0.5
learning_rate = 0.001
n_iters = 200
beam_width=10
min_len=5
n_best=5



input_lang, output_lang, train_pairs, train_max_length = prepareData("train", "zh", "en", data_path)
_, _, dev_pairs, _ = prepareData('dev', 'zh', 'en', path=data_path)
# _, _, test_pairs, _ = prepareData('test', 'zh', 'en', path=data_path)

file_check('/scratch/yn811/chinese_ft_300.txt')
# file_check('/scratch/yn811/vietnamese_ft_300.txt')
file_check('/scratch/yn811/english_ft_300.txt')
source_embedding, notPretrained = load_fasttext_embd('/scratch/yn811/chinese_ft_300.txt', input_lang, source_words_to_load)
target_embedding, notPretrained = load_fasttext_embd('/scratch/yn811/english_ft_300.txt', output_lang, target_words_to_load)

params = {'batch_size':BATCH_SIZE, 'shuffle':False, 'collate_fn':vocab_collate_func, 'num_workers':20}
params2 = {'batch_size':BATCH_SIZE, 'shuffle':False, 'collate_fn':vocab_collate_func, 'num_workers':20}

train_set, dev_set = Dataset(train_pairs, input_lang, output_lang), Dataset(dev_pairs, input_lang, output_lang)
train_loader = torch.utils.data.DataLoader(train_set, **params)
dev_loader = torch.utils.data.DataLoader(dev_set, **params2)

encoder = EncoderRNN(input_lang.n_words, EMB_DIM, encoder_hidden_size,
                     encoder_layers, decoder_hidden_size, source_embedding, device).to(device)
decoder = DecoderRNN(output_lang.n_words, EMB_DIM, decoder_hidden_size, maxout_size,
                     decoder_layers, target_embedding, dropout_p=0.1, device=device).to(device)

print(encoder, decoder)
trainIters(encoder, decoder, train_loader, dev_loader, \
            input_lang, output_lang, \
            n_iters, plot_every=plot_every, \
            learning_rate=learning_rate, device=device, teacher_forcing_ratio=teacher_forcing_ratio, label="RNN_encoder_decoder",
            use_lr_scheduler = True, gamma_en = 0.9, gamma_de = 0.9, beam_width=beam_width, min_len=min_len, n_best=n_best, save_result_path = SAVE_RESULT_PATH)

showPlot(plot_losses, 'Train_Loss_Curve', plot_save_path)
#encoder.load_state_dict(torch.load("encoder.pth"))
#decoder.load_state_dict(torch.load("attn_decoder.pth"))


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='training')
#     parser.add_argument('--CUDA', type=str, action='store', help='use cuda?', default='True')

#     parser.add_argument('--model_path', required=False, help='path to save model', default='./checkpoint')
#     parser.add_argument('--epoch', type=int, action='store', help='number of epoches to train', default=50)
#     parser.add_argument('--batch_size', type=int, action='store', help='batch size', default=32)
#     parser.add_argument('--model', type=str, action='store', help='model to use', default='vanilla')
#     parser.add_argument('--resume', type=str, action='store', help='model path to resume', default='False')
#     parser.add_argument('--dataset', type=str, action='store', help='dataset to train on', default="../data/fl_processed_data_train.pickle")

#     args = parser.parse_args()
#     print(args)
#     main(args)