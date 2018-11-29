import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from tools.Constants import *
import numpy as np

# check all the sizes!!!

class EncoderRNN(nn.Module):
    def __init__(self, input_size, emb_dim, hidden_size, num_layers, decoder_hidden_size, pre_embedding=None, use_bi=False, device=DEVICE):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bi = use_bi
        self.embedding = nn.Embedding(input_size, emb_dim, padding_idx=PAD)
        if pre_embedding is not None:
            self.embedding.weight = nn.Parameter(torch.FloatTensor(pre_embedding))

        self.gru = nn.GRU(emb_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=use_bi)
        self.decoder2c = nn.Sequential(nn.Linear(hidden_size*(1+use_bi), hidden_size), nn.Tanh())
        self.decoder2h0 = nn.Sequential(nn.Linear(hidden_size, decoder_hidden_size), nn.Tanh())
        
        self.device = device

    def forward(self, source, hidden, lengths):
        batch_size = source.size(0)
        seq_len = source.size(1)
        embedded = self.embedding(source) # (batch_sz, seq_len, emb_dim)
        packed = rnn.pack_padded_sequence(embedded, lengths.cpu().numpy(), batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = rnn.pad_packed_sequence(outputs, batch_first=True)

        outputs = outputs.view(batch_size, seq_len, 2, self.hidden_size) # batch, seq_len, num_dir, hidden_sz        

        if self.use_bi:
            hidden = outputs[:, 0, 1, :]
            hidden = self.decoder2h0(hidden)
            hidden = hidden.unsqueeze(0)
            return None, hidden, outputs, output_lengths 
        else:
            c = self.decoder2c(hidden) # (num_layers, batch_sz, hidden_size)
            hidden = self.decoder2h0(c) # (num_layers, batch_sz, decoder_hidden_size)
            return c, hidden, outputs, output_lengths

    def initHidden(self, batch_size):
        return nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(self.num_layers*(1+self.use_bi), batch_size,
                                                       self.hidden_size).type(torch.FloatTensor).to(self.device)), requires_grad=False)
    
    
class DecoderRNN(nn.Module):
    def __init__(self, output_size, emb_dim, hidden_size, maxout_size, num_layers=1, pre_embedding=None, dropout_p=0.1, device=DEVICE):
        super(DecoderRNN, self).__init__()
        
        # Define parameters
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.device = device
        
        # Define layers
        self.embedding = nn.Embedding(output_size, emb_dim, padding_idx=PAD)
        if pre_embedding is not None:
            self.embedding.weight = nn.Parameter(torch.FloatTensor(pre_embedding))
        self.gru = nn.GRU(emb_dim+hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.maxout = Maxout(hidden_size + hidden_size + emb_dim, maxout_size, 2)
        self.linear = nn.Linear(maxout_size, output_size)
    
    def forward(self, word_input, last_hidden, c, 
                encoder_outputs, encoder_output_lengths):
        """
        @ word_input: (batch, 1)
        @ last_hidden: (num_layers, batch, hidden_size)
        """
        rnn_input = self.embedding(word_input)  # B x 1 x emb_dim     
        c = c.transpose(0, 1).contiguous().view(word_input.size(0), 1, -1)

        rnn_input = torch.cat((rnn_input, c), dim=2)
        output, hidden = self.gru(rnn_input, last_hidden)

        output = output.squeeze(1) # B x hidden_size
        output = torch.cat((output, rnn_input.squeeze()), dim=1)
        output = self.maxout(output)
        output = self.linear(output)
        output = F.log_softmax(output, dim=1)

        return output, hidden, None


class DecoderRNN_Attention(nn.Module):
    def __init__(self, output_size, emb_dim, hidden_size, n_layers=1, pre_embedding=None, 
                 dropout_p=0.1, device=DEVICE, method="dot"):
        super(DecoderRNN_Attention, self).__init__()
        
        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.device = device
        self.attn = Attention(hidden_size, method=method)
        # Define layers
        self.embedding = nn.Embedding(output_size, emb_dim, padding_idx=PAD)
        if pre_embedding is not None:
            self.embedding.weight = nn.Parameter(torch.FloatTensor(pre_embedding))
        
        
        self.gru = nn.GRU(self.hidden_size + emb_dim, self.hidden_size, 
                          self.n_layers, batch_first=True)#, dropout=self.dropout_p)
        self.maxout = Maxout(hidden_size + hidden_size + emb_dim, hidden_size, 2)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, word_input, last_hidden, c, 
                encoder_outputs, encoder_output_lengths):

        word_embedded = self.embedding(word_input)
        attn_context, attn_weights = self.attn(encoder_outputs, last_hidden, encoder_output_lengths, self.device)
        
        rnn_input = torch.cat([attn_context, word_embedded], dim=2)
        output, hidden = self.gru(rnn_input, last_hidden)
        
        output = output.squeeze(1) # B x hidden_size
        output = torch.cat((output, rnn_input.squeeze()), dim=1)
        output = self.maxout(output)
        output = self.linear(output)
        output = F.log_softmax(output, dim=1)
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights

    
class Attention(nn.Module):
    def __init__(self, hidden_size, method="cat"):
        super().__init__()
        self.hidden_size = hidden_size
        self.method = method
        if method == "cat":
            self.energy = nn.Sequential(nn.Linear(hidden_size*2, hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(hidden_size, 1))
            
    
    def set_mask(self, encoder_output_lengths, device):
        seq_len = max(encoder_output_lengths).item()
        mask = (torch.arange(seq_len).expand(len(encoder_output_lengths), seq_len) > \
                    encoder_output_lengths.unsqueeze(1)).to(device)
        return mask
    
    def forward(self, encoder_outputs, last_hidden, encoder_output_lengths, device):
        encoder_outputs = encoder_outputs[:, :, 0, :] + encoder_outputs[:, :, 1, :]

        if self.method == "cat":
            last_hidden = last_hidden.transpose(0, 1).expand_as(encoder_outputs)
            energy = self.energy(torch.cat([last_hidden.squeeze(), encoder_outputs], dim=2))
        elif self.method == "dot":
            energy = torch.bmm(encoder_outputs, last_hidden.permute(1, 2, 0))
        
        # (batch_size, seq_len, 1)
        energy = energy.squeeze(2)
        mask = self.set_mask(encoder_output_lengths, device)

        energy.data.masked_fill_(mask, -float('inf'))
        attn = F.softmax(energy, dim=1).unsqueeze(1) # (batch_size, 1, seq_len)
        attn_context = torch.bmm(attn, encoder_outputs)
        # (batch_size, 1, seq_len) * (batch_size, seq_len, hidden_size)
        return attn_context, attn
        
        
class Maxout(nn.Module):

    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)


    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m