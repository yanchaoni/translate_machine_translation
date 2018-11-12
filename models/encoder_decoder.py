import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device=device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self): # (num_layers * num_directions, batch, hidden_size)
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        
        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        # self.dropout = nn.Dropout(dropout_p)
        # self.attn = nn.Linear(hidden_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, word_input, last_hidden, encoder_outputs):
        # Note that we will only be running forward for a single decoder time step, but will use all encoder outputs
        
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N
        # word_embedded = self.dropout(word_embedded)
        ##############################
        # TODO: Implement Attention  #
        ##############################
        encoder_outputs = encoder_outputs.view(1, 1, -1)
        rnn_input = torch.cat((word_embedded, encoder_outputs), 2)
        # Note!!!! hidden initialization is related to num layers!!!
        output, hidden = self.gru(rnn_input, last_hidden)
        
        # Final output layer
        output = output.squeeze(0) # B x N
        output = F.log_softmax(self.out(output), dim=1)
        
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden


class DecoderRNN_Attention(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        
        # Define parameters
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
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
        return torch.zeros(1, 1, self.hidden_size, device=device)

