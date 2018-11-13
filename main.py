import torch
from eval import *
from tools.preprocess import prepareData, load_fasttext_embd
from tools.Dataloader import *
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
fasttext_viet_embd = load_fasttext_embd('.........../vietnamese_ft_300.txt')

list_index=list(range(len(pairs)))
random.shuffle(list_index)
train_list,val_list,test_list=list_index[:round(0.7*len(pairs))],list_index[round(0.7*len(pairs)):round(0.9*len(pairs))],list_index[round(0.9*len(pairs)):]

params = {'batch_size': 16,'shuffle': False,'collate_fn': vocab_collate_func,'num_workers':1}
params2 = {'batch_size': 1,'shuffle': False,'collate_fn': vocab_collate_func,'num_workers':1}

training_set, validation_set,testing_set = Dataset(train_list,pairs,input_lang,output_lang), Dataset(val_list,pairs,input_lang,output_lang),Dataset(test_list,pairs,input_lang,output_lang)
training_generator = data.DataLoader(training_set, **params)
validation_generator = data.DataLoader(validation_set, **params)
testing_generator = data.DataLoader(testing_set, **params2)



## need to add match function to load embds into lookup tables

teacher_forcing_ratio = 0.5

hidden_size = 300
encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
decoder = DecoderRNN(hidden_size, output_lang.n_words, n_layers=1, dropout_p=0.1).to(device)

##UNCOMMENT TO TRAIN THE MODEL
trainIters(encoder, decoder,training_generator, 350, print_every=10)

#encoder.load_state_dict(torch.load("encoder.pth"))
#decoder.load_state_dict(torch.load("attn_decoder.pth"))

for i, (data1,data2, len1,len2) in enumerate(testing_generator):
    fre,eng,lenfre,leneng=data1.to(device),data2.to(device),len1.to(device),len2.to(device)
    words=evaluate_1(encoder1, attn_decoder1, fre, max_length=MAX_LENGTH)
    print(' '.join([input_lang.index2word[k.item()] for k in fre[0]]))
    print(' '.join([output_lang.index2word[k.item()] for k in eng[0]]))
    print(' '.join([output_lang.index2word[k] for k in words]))
    if i==10:
        break
