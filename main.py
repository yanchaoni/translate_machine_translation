import argparse
from models.encoder_decoder import *
import os.path
import os
import torch
from tools.Constants import *
from tools.Dataloader import *
from tools.helper import *
from tools.preprocess import *
from train import trainIters


def main(args):
    if args.decoder_type == "attn":
        args.use_bi = True
    source_words_to_load = 1000000
    target_words_to_load = 1000000
    input_lang, output_lang, train_pairs, train_max_length = prepareData("train", args.language, "en", args.data_path, max_len_ratio=args.max_len_ratio)
    _, _, dev_pairs, _ = prepareData('dev', args.language, 'en', path=args.data_path, max_len_ratio=0.99999)
    # _, _, test_pairs, _ = prepareData('test', args.language, 'en', path=args.data_path)
    if args.language == "zh":
        file_check(args.FT_emb_path+'chinese_ft_300.txt')
        source_embedding, source_notPretrained = load_fasttext_embd(args.FT_emb_path+'chinese_ft_300.txt', input_lang, input_lang, source_words_to_load)
    else:
        file_check(args.FT_emb_path+'vietnamese_ft_300.txt')
        source_embedding, source_notPretrained = load_fasttext_embd(args.FT_emb_path+'vietnamese_ft_300.txt', input_lang, input_lang, source_words_to_load)

    file_check(args.FT_emb_path+'english_ft_300.txt')
    target_embedding, target_notPretrained = load_fasttext_embd(args.FT_emb_path+'english_ft_300.txt', output_lang, input_lang, target_words_to_load)

    params = {'batch_size':args.batch_size, 'shuffle':False, 'collate_fn':vocab_collate_func, 'num_workers':20}
    params2 = {'batch_size':args.batch_size, 'shuffle':False, 'collate_fn':vocab_collate_func, 'num_workers':20}

    train_set, dev_set = Dataset(train_pairs, input_lang, output_lang), Dataset(dev_pairs, input_lang, output_lang)
    train_loader = torch.utils.data.DataLoader(train_set, **params)
    dev_loader = torch.utils.data.DataLoader(dev_set, **params2)

    encoder = EncoderRNN(input_lang.n_words, EMB_DIM, args.encoder_hidden_size,
                        args.encoder_layers, args.decoder_hidden_size, source_embedding, args.use_bi, args.device).to(args.device)
    if args.decoder_type == "basic":
        decoder = DecoderRNN(output_lang.n_words, EMB_DIM, args.decoder_hidden_size,
                            args.decoder_layers, target_embedding, dropout_p=0.1, device=args.device).to(args.device)
    elif args.decoder_type == "attn":
        decoder = DecoderRNN_Attention(output_lang.n_words, EMB_DIM, args.decoder_hidden_size,
                                    args.decoder_layers, target_embedding, dropout_p=0.1, device=args.device).to(args.device)
    else:
        raise ValueError

    print(encoder, decoder)
    trainIters(encoder, decoder, train_loader, dev_loader, \
                input_lang, output_lang, train_max_length, \
                args.epoch, plot_every=args.plot_every, \
                learning_rate=args.learning_rate, device=args.device, teacher_forcing_ratio=args.teacher_forcing_ratio, label=args.save_model_name,
                use_lr_scheduler = True, gamma_en = 0.99, gamma_de = 0.99, 
                beam_width=args.beam_width, min_len=args.min_len, n_best=args.n_best, decode_method=args.decode_method, save_result_path = args.save_result_path)

    showPlot(plot_losses, 'Train_Loss_Curve', args.save_result_path)
    #encoder.load_state_dict(torch.load("encoder.pth"))
    #decoder.load_state_dict(torch.load("attn_decoder.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--language', type=str, action='store', help='source language')
    parser.add_argument('--save_model_name', type=str, action='store', help='what name to save the model')
    parser.add_argument('--FT_emb_path', type=str, action='store', help='what path is pretrained embedding saved/to be saved')
    parser.add_argument('--data_path', type=str, action='store', help='what path is translation data saved')
    
    parser.add_argument('--device', type=str, action='store', help='what device to use', default=DEVICE)
    parser.add_argument('--batch_size', type=int, action='store', help='batch size', default=64)
    parser.add_argument('--learning_rate', type=float, action='store', help='learning rate', default=0.001)
    parser.add_argument('--teacher_forcing_ratio', type=float, action='store', help='teacher forcing ratio', default=0.5)
    parser.add_argument('--plot_every', type=int, action='store', help='save plot log every ? steps', default=1e5)
    parser.add_argument('--epoch', type=int, action='store', help='number of epoches to train', default=50)    
    parser.add_argument('--model_path', required=False, help='path to save model', default='./') # not imp
    
    parser.add_argument('--encoder_layers', type=int, action='store', help='num of encoder layers', default=1) # might have bug
    parser.add_argument('--encoder_hidden_size', type=int, action='store', help='encoder num hidden', default=150)
    parser.add_argument('--use_bi', type=bool, action='store', help='if use bid encoder', default=False)
    
    parser.add_argument('--decoder_type', type=str, action='store', help='basic/attn', default='attn')    
    parser.add_argument('--decoder_layers', type=int, action='store', help='num of decoder layers', default=1) # init not imp
    parser.add_argument('--decoder_hidden_size', type=int, action='store', help='decoder num hidden', default=150)    

    parser.add_argument('--decode_method', type=str, action='store', help='beam/greedy', default='beam')
    parser.add_argument('--beam_width', type=int, action='store', help='beam width', default=10)
    parser.add_argument('--n_best', type=int, action='store', help='find >=n best from beam', default=5)
    parser.add_argument('--min_len', type=int, action='store', help='placeholder, meaningless', default=5)   

    parser.add_argument('--max_len_ratio', type=float, action='store', help='max len ratio to filter training pairs', default=0.9)
    parser.add_argument('--save_result_path', type=str, action='store', help='what path to save results', default='results/')

    args = parser.parse_args()
    print(args)
    main(args)
