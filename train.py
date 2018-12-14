import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
import random
import time
from tools.helper import timeSince, showPlot
from tools.preprocess import tensorsFromPair
from tools.Constants import *
from eval import test

def train(source, target, source_len, target_len, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_WORD_LENGTH[1],device=DEVICE, teacher_forcing_ratio=0.5, use_transformer = False):
    """
    source: (batch_size, max_input_len)
    target: (batch_size, max_output_len)
    """
    encoder_hidden, encoder_c_state = encoder.initHidden(source.size(0))
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0
    if use_transformer:
        e_outputs = encoder(source, src_mask)
    else: 
        c, decoder_hidden, encoder_outputs, encoder_output_lengths, encoder_c_state = \
                                                    encoder(source, encoder_hidden, source_len, encoder_c_state)
        decoder_c_state = encoder_c_state
        decoder_input = torch.tensor([[SOS]]*source.size(0), device=device)



    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        for di in range(target_len.max().item()):
            decoder_output, decoder_hidden, attn, decoder_c_state = decoder(decoder_input, decoder_hidden, c, 
                                                     encoder_outputs, encoder_output_lengths, decoder_c_state)
            # TODO: mask out irrelevant loss
            loss += criterion(decoder_output, target[:, di])
            decoder_input = target[:, di].unsqueeze(1) # (batch_size, 1)
    else:
        for di in range(target_len.max().item()):
            decoder_output, decoder_hidden, attn, decoder_c_state = decoder(decoder_input, decoder_hidden, c, 
                                                     encoder_outputs, encoder_output_lengths, decoder_c_state)
            loss += criterion(decoder_output, target[:,di])
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach().unsqueeze(1)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), 3)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 3)

#     import pdb
#     pdb.set_trace()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_len.max().item()

def trainIters(encoder, decoder, train_loader, dev_loader, \
               input_lang, output_lang, 
               input_lang_dev, output_lang_dev,
               max_word_len, n_iters, 
               plot_every=100, print_every=1, weight_decay=0,
               learning_rate=0.01, device=DEVICE, 
               teacher_forcing_ratio=0.5, label="", 
               use_lr_scheduler = True, gamma_en = 0.9, gamma_de=0.9, 
               beam_width=3, min_len=1, n_best=1, decode_method="beam", 
               save_result_path = '', save_model=False):
    start = time.time()
    num_steps = len(train_loader)
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    cur_best = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler_encoder = ExponentialLR(encoder_optimizer, gamma_en, last_epoch=-1) 
    scheduler_decoder = ExponentialLR(decoder_optimizer, gamma_de, last_epoch=-1) 
    criterion = nn.NLLLoss()
 
    loss_file = open(save_result_path +'/%s-loss.txt'%label, 'w+')
    bleu_file = open(save_result_path +'/%s-bleu.txt'%label, 'w+')
    for epoch in range(1, n_iters + 1):
        if use_lr_scheduler:
            scheduler_encoder.step()
            scheduler_decoder.step()
        for i, (data1, data2, len1, len2) in enumerate(train_loader):
            encoder.train()
            decoder.train()
            source, target, source_len, target_len = data1.to(device), data2.to(device),len1.to(device),len2.to(device)

            loss = train(source, target, source_len, target_len, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, 
                         device=device, teacher_forcing_ratio=teacher_forcing_ratio)
            print_loss_total += loss
            plot_loss_total += loss

            if i != 0 and (i % plot_every == 0):
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
        if epoch != 0 and (epoch % print_every == 0):        
            print_loss_avg = print_loss_total / len(train_loader)
            print_loss_total = 0
            print("testing..")
            bleu_score, _ , _, _ = test(encoder, decoder, dev_loader, 
                                    input_lang, output_lang,
                                    input_lang_dev, output_lang_dev,
                                    beam_width, min_len, n_best, 
                                    max_word_len, decode_method, device)
            print('%s epoch:(%d %d%%) step[%d %d] Average_Loss %.4f, Bleu Score %.3f' % (timeSince(start, epoch / n_iters),
                                        epoch, epoch / n_iters * 100, i, num_steps, print_loss_avg, bleu_score))
            loss_file.write("%s\n" % print_loss_avg)    
            bleu_file.write("%s\n" % bleu_score)
            if (bleu_score > cur_best):
                print("found best! save model...")
                fail_cnt = 0
                if save_model:
                    torch.save(encoder.state_dict(), 'encoder' + "-" + label + '.ckpt')
                    torch.save(decoder.state_dict(), 'decoder' + "-" + label + '.ckpt')
                    print("model saved")
                cur_best = bleu_score
            else:
                fail_cnt += 1
            if fail_cnt == 15:
                print("No improvement for 15 epochs. Halt!")
                return 0
        
        torch.cuda.empty_cache()
    loss_file.close()
    bleu_file.close()
