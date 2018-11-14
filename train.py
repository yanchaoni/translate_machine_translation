import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import time
from tools.helper import timeSince, showPlot
from tools.preprocess import tensorsFromPair
from tools.Constants import SOS, EOS, DEVICE
from tools.Constants import MAX_WORD_LENGTH

def train(fre,eng,lenfre,leneng, encoder1, decoder1, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_WORD_LENGTH[1],device=DEVICE,teacher_forcing_ratio=0.5):
    encoder_hidden = encoder1.initHidden(fre)
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0
    encoder_output, encoder_hidden = encoder1(fre, encoder_hidden,lenfre)

    decoder_input = torch.tensor([[SOS]*fre.size(0)], device=device)
    decoder_hidden = encoder_hidden
    encoder_outputs=encoder_output
    
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        for di in range(max_length):
            decoder_output, decoder_hidden= decoder1(decoder_input, decoder_hidden)#, encoder_outputs
            loss += criterion(decoder_output, eng[:,di])
            decoder_input = eng[:,di].unsqueeze(0)  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(max_length):
            decoder_output, decoder_hidden= decoder1(decoder_input, decoder_hidden)#, encoder_outputs
            loss += criterion(decoder_output, eng[:,di])
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach().unsqueeze(0)
#             if decoder_input.item() == EOS_token:
#                 break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() #/ target_length

def trainIters(encoder, decoder,training_generator, n_iters,print_every=1000, plot_every=100, learning_rate=0.01,device=DEVICE,teacher_forcing_ratio=0.5):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        for i, (data1,data2, len1,len2) in enumerate(training_generator):
            if i % 1000==0:
                print(i)
            fre,eng,lenfre,leneng=data1.to(device),data2.to(device),len1.to(device),len2.to(device)
            loss = train(fre, eng,lenfre,leneng, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion,device=DEVICE)
            print_loss_total += loss
            plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
