import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from tools.Constants import *
import numpy as np


class EncoderRNN(nn.Module):
    def __init__(self, input_size, emb_dim, hidden_size, num_layers, pre_embedding=None, device=DEVICE):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, emb_dim, padding_idx=PAD)
        if pre_embedding is not None:
            self.embedding.weight = nn.Parameter(torch.FloatTensor(pre_embedding))
        # TODO: load from pretrain
        self.gru = nn.GRU(emb_dim, hidden_size, num_layers=num_layers, batch_first=True)
        
        self.decoder_c = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.decoder_h0 = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())
        
        self.device = device

    def forward(self, source, hidden, lengths):
        embedded = self.embedding(source) # (batch_sz, seq_len, emb_dim)
        packed = rnn.pack_padded_sequence(embedded, lengths.cpu().numpy(), batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = rnn.pad_packed_sequence(outputs, batch_first=True)
        
        c = self.decoder_c(hidden) # (num_layers, batch_sz, hidden_size) -> (num_layers, batch_sz, hidden_size)
        hidden = self.decoder_h0(c) # (num_layers, batch_sz, hidden_size) -> (num_layers, batch_sz, hidden_size)
        return c, hidden

    def initHidden(self, batch_size):
        return nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(self.num_layers, batch_size,
                                                       self.hidden_size).type(torch.FloatTensor).to(self.device)), requires_grad=False)
#         return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=self.device)
    
    
class DecoderRNN(nn.Module):
    def __init__(self, output_size, emb_dim, hidden_size, num_layers=1, pre_embedding=None, dropout_p=0.1, device=DEVICE):
        super(DecoderRNN, self).__init__()
        
        # Define parameters
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
        self.maxout = Maxout(hidden_size, hidden_size, 2)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, word_input, last_hidden, encoder_outputs):
        """
        Note that we will only be running forward for a single decoder time step, but will use all encoder outputs
        Get the embedding of the current input word (last output word)
        @ word_input: (batch, 1)
        @ last_hidden: (num_layers, batch, hidden_size)
        """
        rnn_input = self.embedding(word_input)  # B x 1 x emb_dim     
        c = encoder_outputs.transpose(0, 1).contiguous().view(word_input.size(0), 1, -1)

        rnn_input = torch.cat((rnn_input, c), dim=2)
        output, hidden = self.gru(rnn_input, last_hidden)

        output = output.squeeze(1) # B x hidden_size
        output = self.maxout(output)
        output = self.linear(output)
        output = F.log_softmax(output, dim=1)

        return output, hidden


class DecoderRNN_Attention(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1, device=DEVICE):
        super(DecoderRNN_Attention, self).__init__()
        
        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.device = device
        
        # Define layers
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size * 2, self.hidden_size, self.n_layers, dropout=self.dropout_p)
        self.out = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, word_input, last_hidden, encoder_outputs):
        # Note that we will only be running forward for a single decoder time step, but will use all encoder outputs
        
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N
        word_embedded = self.dropout(word_embedded)
        ##############################
        # TODO: Implement Attention  #
        ##############################
        #encoder_outputs = encoder_outputs.view(1, 1, -1)
        rnn_input = torch.cat((word_embedded, encoder_outputs), 2)
        rnn_input = word_embedded
        attn_weights = F.softmax(self.attn(torch.cat((word_embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))
        output, hidden = self.gru(rnn_input, last_hidden)
        output = torch.cat((word_embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        # Final output layer
        output = output.squeeze(0) # B x N ???
        output = F.log_softmax(self.out(output), dim=1)
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

    
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