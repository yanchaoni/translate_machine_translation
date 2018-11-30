import torch
from tools.preprocess import tensorFromSentence
from tools.Constants import SOS, EOS, DEVICE, BATCH_SIZE, MAX_WORD_LENGTH
import numpy.random as random
from tools.beam import Beam
from tools.bleu_calculation import *

def beam_decode(decoder, decoder_hidden, c, encoder_hidden, 
                encoder_outputs, encoder_output_lengths, 
                max_length, batch_size, beam_width, min_len, n_best, device):
    """
    run beam search to decode
    """
    beams = [Beam(beam_width, min_len, n_best, device) for _ in range(batch_size)]
    if c is not None:
        c = torch.cat(
            [c[:, i:i+1, :].expand(c.size(0), beam_width, c.size(2)) for i in range(batch_size)], dim=1).to(device)
    encoder_outputs = torch.cat(
        [encoder_outputs[i:i+1, :, :, :].expand(beam_width, encoder_outputs.size(1), encoder_outputs.size(2), encoder_outputs.size(3)) \
            for i in range(batch_size)], dim=0).to(device)

    encoder_output_lengths = torch.cat([encoder_output_lengths[i:i+1].expand(beam_width) for i in range(batch_size)])
    assert encoder_outputs.size(0) == beam_width*batch_size
    assert encoder_output_lengths.size(0) == beam_width*batch_size
    
    for di in range(max_length):

        decoder_input = torch.stack(
            [b.get_current_state() for b in beams] # (B, k)
            ).view(-1, 1).to(device) # (B x k, 1)
        
        if beams[0].prev_ks:
            decoder_hidden = torch.cat(
                [decoder_hidden[:, i:i+beam_width, :].index_select(1, b.get_current_origin()) 
                    for i, b in enumerate(beams)], dim=1).to(device)
        else:
            decoder_hidden = torch.cat([decoder_hidden[:, i:i+1, :].expand(decoder_hidden.size(0), beam_width, decoder_hidden.size(2)) 
                           for i in range(batch_size)], dim=1).to(device)
        
        assert decoder_hidden.size(1) == batch_size*beam_width
        
        decoder_output, decoder_hidden, attn = decoder(
                decoder_input, decoder_hidden, c, encoder_outputs, encoder_output_lengths)
        _, topi = decoder_output.topk(1, dim=1)
        decoder_output = decoder_output.view(-1, beam_width, decoder.output_size)
        assert decoder_output.size(0) == batch_size
        active = []
        for b in range(batch_size):
            if beams[b].done():
                continue

            if not beams[b].advance(decoder_output.data[b]):
                active.append(b)
        if not active:
            break
    decoded_words = []    
    for b in beams:
        ks = b.sort_finished()[1]
        decoded_words.append(b.get_hyp(*ks[0]))
    return decoded_words

def evaluate(encoder, decoder, source, source_len, max_length, beam_width, min_len, n_best, method, device):
    """
    Function that generate translation.
    First, feed the source sentence into the encoder and obtain the hidden states from encoder.
    Secondly, feed the hidden states into the decoder and unfold the outputs from the decoder.
    Lastly, for each outputs from the decoder, collect the corresponding words in the target language's vocabulary.
    And collect the attention for each output words.
    @param encoder: the encoder network
    @param decoder: the decoder network
    @param sentence: string, a sentence in source language to be translated
    @param max_length: the max # of words that the decoder can return
    @output decoded_words: a list of words in target language
    @output decoder_attentions: a list of vector, each of which sums up to 1.0
    """
    # process input sentence
    with torch.no_grad():
        batch_size = source.size(0)
        # encode the source lanugage
        encoder_hidden = encoder.initHidden(source.size(0))

        c, decoder_hidden, encoder_outputs, encoder_output_lengths = encoder(source, encoder_hidden, source_len)
        if method == "greedy":
            decoder_input = torch.tensor([[SOS]]*source.size(0), device=source.device)  # (B, 1)

            decoded_words = []

            for di in range(max_length):
                # for each time step, the decoder network takes two inputs: previous outputs and the previous hidden states
                decoder_output, decoder_hidden, attn = decoder(
                    decoder_input, decoder_hidden, c, encoder_outputs, encoder_output_lengths)


                _, topi = decoder_output.topk(1, dim=1)
                decoded_words.append(topi.squeeze().detach())
                decoder_input = topi.squeeze().detach().unsqueeze(1)
            decoded_words = list(zip(*decoded_words))
        elif method == "beam":
            decoded_words = beam_decode(decoder, decoder_hidden, c, encoder_hidden, 
                                        encoder_outputs, encoder_output_lengths, 
                                        max_length, batch_size, beam_width, min_len, n_best, device)
        else:
            raise ValueError


        return decoded_words # , decoder_attentions[:di + 1]

def trim_decoded_words(decoded_words):
    # HAZARD!!!!
    try:
        trim_loc = decoded_words.index("<EOS>")
    except:
        trim_loc = len(decoded_words)
    return decoded_words[:trim_loc]

def test(encoder, decoder, dataloader, input_lang, output_lang, beam_width, min_len, n_best, max_word_len, method, device):
    all_scores = 0
    decoded_list =[]
    target_list = []
    bleu_cal = BLEUCalculator(smooth="exp", smooth_floor=0.00,
                 lowercase=False, use_effective_order=True,
                 tokenizer=DEFAULT_TOKENIZER)
    for (data1,data2,len1,len2) in (dataloader):
        source, target, source_len, target_len = data1.to(device),data2.to(device),len1.to(device),len2.to(device)
        decoded_words = evaluate(encoder, decoder, source, source_len, max_word_len[1],
                                beam_width, min_len, n_best, method, device)

        decoded_words = [[output_lang.index2word[k.item()] for k in decoded_words[i]] for i in range(len(decoded_words))]
        target_words = [[output_lang.index2word[k.item()] for k in target[i]] for i in range(len(decoded_words))]

        decoded_list.extend([' '.join(trim_decoded_words(j)) for j in decoded_words])
        target_list.extend([' '.join(target_words[j][:target_len[j]-1]) for j in range(len(decoded_words))])
    bleu_scores = bleu_cal.bleu(decoded_list,[target_list])[0]
    return bleu_scores, decoded_list, target_list

def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, max_length, n=10):
    """
    Randomly select a English sentence from the dataset and try to produce its French translation.
    Note that you need a correct implementation of evaluate() in order to make this function work.
    """
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], input_lang,  max_length)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def evaluate_1(encoder, decoder, sentence, max_length=MAX_WORD_LENGTH):
    # process input sentence
    with torch.no_grad():
        encoder_hidden = encoder.initHidden(sentence)
        encoder_output, encoder_hidden = encoder1(sentence, encoder_hidden,lenfre)
        decoder_input = torch.tensor([[SOS_token]*sentence.size(0)], device=device)
        decoder_hidden = encoder_hidden
        encoder_outputs=encoder_output
        decoded_words = []
#         print(decoder_input)
        # Without teacher forcing: use its own predictions as the next input
        for di in range(10):
#             print(decoder_input.size(), decoder_hidden.size())
            decoder_output, decoder_hidden= decoder(decoder_input, decoder_hidden)#, encoder_outputs
            topv, topi = decoder_output.topk(1)
#             print(topi,topi.squeeze().detach().unsqueeze(0))
            decoder_input = topi.detach()
            decoded_words.append(topi.squeeze().detach().item())
    return decoded_words

# for i, (data1,data2, len1,len2) in enumerate(testing_generator):
#     fre,eng,lenfre,leneng=data1.to(device),data2.to(device),len1.to(device),len2.to(device)
#     words=evaluate_1(encoder1, attn_decoder1, fre, max_length=MAX_LENGTH)
#     print(' '.join([input_lang.index2word[k.item()] for k in fre[0]]))
#     print(' '.join([output_lang.index2word[k.item()] for k in eng[0]]))
#     print(' '.join([output_lang.index2word[k] for k in words]))
#     if i==10:
#         break
