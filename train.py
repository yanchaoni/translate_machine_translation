import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import time
from tools.helper import timeSince, showPlot
from tools.preprocess import tensorsFromPair
from tools.Constants import *
from eval import test

def train(source, target, source_len, target_len, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_WORD_LENGTH[1],device=DEVICE, teacher_forcing_ratio=0.5):
    """
    source: (batch_size, max_input_len)
    target: (batch_size, max_output_len)
    """
    encoder_hidden = encoder.initHidden(source.size(0))
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0
    encoder_outputs, encoder_hidden = encoder(source, encoder_hidden, source_len)

    decoder_input = torch.tensor([[SOS]]*source.size(0), device=device)
    decoder_hidden = encoder_hidden # (batch_size, 1, hidden_size*num_layers)
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        for di in range(len(target[0])):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # TODO: mask out irrelevant loss
            loss += criterion(decoder_output, target[:, di])
            decoder_input = target[:, di].unsqueeze(1) # (batch_size, 1)
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(len(target[0])):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target[:,di])
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach().unsqueeze(1)

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() #/ target_length

def trainIters(encoder, decoder, train_loader, dev_loader, \
            input_lang, output_lang, \
            n_iters, print_every=1000, plot_every=100, 
            learning_rate=0.01, device=DEVICE, teacher_forcing_ratio=0.5, label=""):
    start = time.time()
    num_steps = len(train_loader)
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    cur_best = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_iters + 1):
        for i, (data1, data2, len1, len2) in enumerate(train_loader):
            print(i, end='\r')
            source, target, source_len, target_len = data1.to(device), data2.to(device),len1.to(device),len2.to(device)
            loss = train(source, target, source_len, target_len, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, device=device)
            print_loss_total += loss
            plot_loss_total += loss

            if i != 0 and (i % print_every == 0):
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print("testing..")
                bleu_score = test(encoder, decoder, dev_loader, input_lang, output_lang, device)
                print('%s epoch:(%d %d%%) step[%d %d] %.4f, %.3f' % (timeSince(start, epoch / n_iters),
                                            epoch, epoch / n_iters * 100, i, num_steps, print_loss_avg, bleu_score))

                if (bleu_score > cur_best):
                    print("found best! save model...")
                    torch.save(encoder.state_dict(), 'encoder' + "-" + label + '.ckpt')
                    torch.save(decoder.state_dict(), 'decoder' + "-" + label + '.ckpt')                    
                    print("model saved")
                    cur_best = bleu_score

            # if epoch % plot_every == 0:
            #     plot_loss_avg = plot_loss_total / plot_every
            #     plot_losses.append(plot_loss_avg)
            #     plot_loss_total = 0