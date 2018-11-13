import torch
from tools.preprocess import tensorFromSentence
from tools.Constants import SOS, EOS, DEVICE
import numpy.random as random
from tools.beam import Beam


def evaluate(encoder, decoder, sentence, input_lang,  max_length):
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
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        # ++++++++++++++++++++++ #
        # ++ need to batchify ++ #
        beam = [Beam(3,3,3,DEVICE)]
        # encode the source lanugage
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS]], device=DEVICE)  # SOS
        # decode the context vector
        decoder_hidden = encoder_hidden # decoder starts from the last encoding sentence
        # output of this function
        decoded_words = []
        # decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            # for each time step, the decoder network takes two inputs: previous outputs and the previous hidden states
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            
            # hint: print out decoder_output and decoder_attention
            # TODO: add your code here to populate decoded_words and decoder_attentions
            # TODO: do this in 2 ways discussed in class: greedy & beam_search
            # --- greedy ---
            topv, topi = decoder_output.topk(1)
            decoded_words.append(topv)
            decoder_input = topi.squeeze().detach()
            # --- beam search ---
            # TODO: wrap beam width dimension on batch dim and unwrap 
            for i in range(len(beam)):
                beam[i].advance(decoder_output)
            decoder_input = topi.squeeze().detach()

            # END TO DO
            

        return decoded_words # , decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, pairs, input_lang,  max_length, n=10):
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
        
def evaluate_1(encoder, decoder, sentence, max_length=MAX_LENGTH):
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

for i, (data1,data2, len1,len2) in enumerate(testing_generator):
    fre,eng,lenfre,leneng=data1.to(device),data2.to(device),len1.to(device),len2.to(device)
    words=evaluate_1(encoder1, attn_decoder1, fre, max_length=MAX_LENGTH)
    print(' '.join([input_lang.index2word[k.item()] for k in fre[0]]))
    print(' '.join([output_lang.index2word[k.item()] for k in eng[0]]))
    print(' '.join([output_lang.index2word[k] for k in words]))
    if i==10:
        break
