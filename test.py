import argparse
from models.encoder_decoder import *
import os.path
import os
import torch
from tools.Constants import *
from tools.Dataloader import *
from tools.helper import *
from tools.preprocess import *
from eval import *


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
        source_embedding, source_notPretrained = load_fasttext_embd(args.FT_emb_path+'chinese_ft_300.txt', input_lang, input_lang, source_words_to_load, reload=args.reload_emb)
    else:
        file_check(args.FT_emb_path+'vietnamese_ft_300.txt')
        source_embedding, source_notPretrained = load_fasttext_embd(args.FT_emb_path+'vietnamese_ft_300.txt', input_lang, input_lang, source_words_to_load, reload=args.reload_emb)

    file_check(args.FT_emb_path+'english_ft_300.txt')
    target_embedding, target_notPretrained = load_fasttext_embd(args.FT_emb_path+'english_ft_300.txt', output_lang, input_lang, target_words_to_load, reload=args.reload_emb)

    params = {'batch_size':args.batch_size, 'shuffle':False, 'collate_fn':vocab_collate_func, 'num_workers':20}
    params2 = {'batch_size':args.batch_size, 'shuffle':False, 'collate_fn':vocab_collate_func, 'num_workers':20}

    train_set, dev_set = Dataset(train_pairs, input_lang, output_lang), Dataset(dev_pairs, input_lang, output_lang)
    train_loader = torch.utils.data.DataLoader(train_set, **params)
    dev_loader = torch.utils.data.DataLoader(dev_set, **params2)
    print(len(train_loader), len(dev_loader))
    encoder = EncoderRNN(input_lang.n_words, EMB_DIM, args.encoder_hidden_size,
                        args.encoder_layers, args.decoder_hidden_size, source_embedding, source_notPretrained,
                         args.use_bi, args.device).to(args.device)
    if args.decoder_type == "basic":
        decoder = DecoderRNN(output_lang.n_words, EMB_DIM, args.decoder_hidden_size,
                            args.decoder_layers, target_embedding, target_notPretrained, 
                             dropout_p=args.decoder_emb_dropout, device=args.device).to(args.device)
    elif args.decoder_type == "attn":
        decoder = DecoderRNN_Attention(output_lang.n_words, EMB_DIM, args.decoder_hidden_size,
                                       args.decoder_layers, target_embedding, target_notPretrained, 
                                       dropout_p=args.decoder_emb_dropout,
                                       device=args.device, method=args.attn_method).to(args.device)
    else:
        raise ValueError

    print(encoder, decoder)
    encoder.load_state_dict(torch.load('encoder' + "-" + args.save_model_name + '.ckpt', map_location=lambda storage, location: storage))
    decoder.load_state_dict(torch.load('decoder' + "-" + args.save_model_name + '.ckpt', map_location=lambda storage, location: storage))
    
    bleu_score, decoded_list, target_list = test(encoder, decoder, train_loader, input_lang, output_lang, 
                                                 args.beam_width, args.min_len, args.n_best, train_max_length, 
                                                 args.decode_method, args.device)
    
    i = 0
    with open("results/test_examples.txt", "w+") as f:
        f.write("bleu: {}\n".format(bleu_score))
        for (source, target, source_len, target_len) in (dev_loader):
            source_list = [ [input_lang.index2word[k.item()] for k in source[i]][:source_len[i]-1] for i in range(len(source))]
            for s in source_list:
                f.write("S: {}\n".format(" ".join(s)))
                f.write("T: {}\n".format(decoded_list[i]))
                f.write("H: {}\n".format(target_list[i]))
                i += 1
             
# python test.py --language vi --save_model_name vi_attn --FT_emb_path /scratch/yn811/ --data_path /scratch/yn811/MT_data --encoder_hidden_size 256 --decoder_hidden_size 256 --save_model False --max_len_ratio 0.99 --decode_method greedy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--language', type=str, action='store', help='source language')
    parser.add_argument('--save_model_name', type=str, action='store', help='what name to save the model')
    parser.add_argument('--FT_emb_path', type=str, action='store', help='what path is pretrained embedding saved/to be saved')
    parser.add_argument('--data_path', type=str, action='store', help='what path is translation data saved')
    
    parser.add_argument('--device', type=str, action='store', help='what device to use', default=DEVICE)
    parser.add_argument('--batch_size', type=int, action='store', help='batch size', default=64)
    parser.add_argument('--learning_rate', type=float, action='store', help='learning rate', default=3e-4)
    parser.add_argument('--teacher_forcing_ratio', type=float, action='store', help='teacher forcing ratio', default=1)
    parser.add_argument('--print_every', type=int, action='store', help='save plot log every ? epochs', default=1)
    parser.add_argument('--plot_every', type=int, action='store', help='save plot log every ? steps', default=1e5)
    parser.add_argument('--epoch', type=int, action='store', help='number of epoches to train', default=50)    
    parser.add_argument('--model_path', required=False, help='path to save model', default='./') # not imp
    parser.add_argument('--reload_emb', type=bool, help='whether to reload embeddings', default=False)
    parser.add_argument('--save_model', type=bool, help='whether to save model on the fly', default=False)
    
    parser.add_argument('--encoder_layers', type=int, action='store', help='num of encoder layers', default=1) # might have bug
    parser.add_argument('--encoder_hidden_size', type=int, action='store', help='encoder num hidden', default=150)
    parser.add_argument('--use_bi', type=bool, action='store', help='if use bid encoder', default=False)
    
    parser.add_argument('--decoder_type', type=str, action='store', help='basic/attn', default='attn')    
    parser.add_argument('--decoder_layers', type=int, action='store', help='num of decoder layers', default=1) # init not imp
    parser.add_argument('--decoder_hidden_size', type=int, action='store', help='decoder num hidden', default=150)
    parser.add_argument('--decoder_emb_dropout', type=float, action='store', help='decoder emb dropout', default=0)
    parser.add_argument('--attn_method', type=str, action='store', help='attn method: cat/dot', default='dot')

    parser.add_argument('--decode_method', type=str, action='store', help='beam/greedy', default='beam')
    parser.add_argument('--beam_width', type=int, action='store', help='beam width', default=10)
    parser.add_argument('--n_best', type=int, action='store', help='find >=n best from beam', default=5)
    parser.add_argument('--min_len', type=int, action='store', help='placeholder, meaningless', default=5)   

    parser.add_argument('--max_len_ratio', type=float, action='store', help='max len ratio to filter training pairs', default=0.9)
    parser.add_argument('--save_result_path', type=str, action='store', help='what path to save results', default='results/')

    args = parser.parse_args()
    print(args)
    main(args)
