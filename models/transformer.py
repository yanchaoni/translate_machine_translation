import torch
import torch.nn as nn
import math,copy
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn
from tools.Constants import *
import numpy as np

# check all the sizes!!!
# embd might need change for self attention, HarvardNLP multiply those weights by math.sqrt(self.emd_size)

## self-attention code adapted from https://github.com/harvardnlp/annotated-transformer/blob/master/The%20Annotated%20Transformer.ipynb

######################################################################
# Encoder Components 
######################################################################

# x -> embd -> multijhead attention -> layer norm -> feed forward -> layer norm -> sum_attn
# (sum_attn -> multijhead attention -> layer norm -> feed forward -> layer norm -> sum_attn )^N


"""
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
"""

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1) # dim_emd_size // num_head
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
             # (batch_size, target_len, d_k) * (batch_size, d_k, source_len)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # after softmax, we can calculate hom much each word will be expressed at this position
    prob_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        prob_attn = dropout(prob_attn)
    sum_attn = torch.matmul(prob_attn, value)
    # sum is like the context vector, which will be sent to feed forward NN, and then sent to decoder
    return sum_attn #, prob_attn

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

"""
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout) 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
"""

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_head, emb_size, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.emb_size = emb_size
        self.num_head = num_head
        self.d_k = emb_size // num_head
        # self.linears = clones(nn.Linear(emb_size, emb_size), 4)
        self.linear = nn.Linear(emb_size, emb_size)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        @query: (batch_size, target_len, emb_size)
        @key: (batch_size, source_len, emb_size)
        @value: (batch_size, source_len, emb_size)
        @mask: mask future information
        """
        batch_size, target_len, source_len = query.size(0), query.size(1), key.size(1)
        
        # do all the linear projections in batch from emb_size
        Q = self.linear(query).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)
        K = self.linear(key).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)
        V = self.linear(value).view(batch_size, -1, self.num_head, self.d_k).transpose(1, 2)
        # Q = self.linear(query).view(batch_size*self.num_head, target_len, self.d_k).transpose(1, 2)
        # K = self.linear(key).view(batch_size*self.num_head, source_len, self.d_k).transpose(1, 2)
        # V = self.linear(value).view(batch_size*self.num_head, source_len, self.d_k).transpose(1, 2)

        # compute 'scaled dot product attention' 
        sum_attn = attention(query, key, value, mask = mask, drop = self.dropout)

        # concat
        sum_attn = sum_attn.transpose(1,2).contiguous().view(batch_size, -1, self.num_head * self.d_k)
        sum_attn = self.linear(sum_attn)

        return sum_attn

class FeedForwardSublayer(nn.Module):
    "Implements FFN equation."
    def __init__(self, emd_size, dim_ff, dropout=0.1):
        super(FeedForwardSublayer, self).__init__()
        self.linear1 = nn.Linear(emd_size, dim_ff)
        self.linear2 = nn.Linear(dim_ff, emd_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sum_attn):
        
        out = self.linear1(sum_attn)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)

        return out


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout = 0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class SelfAttentionEncoderLayer(nn.Module):
    def __init__(self, embd_size, self_attn, feed_forward, dropout=0.1):
        
        super(SelfAttentionEncoderLayer, self).__init__()
        self.self_attn = self_attn          # MultiHeadedAttention
        self.feed_forward = feed_forward    # FeedForwardSublayer
        self.embd_size = embd_size
        self.layernorm = LayerNorm(embd_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):

        # x is the source input
        residual = x

        x = self.self_attn(x, x, x, mask)
        x = residual + x
        x = self.layernorm(x)
        x = self.dropout(x)

        residual = x
        x = x + residual
        x = self.feed_forward(x)
        x = residual + x
        x = self.layernorm(x)

        return x

class SelfAttentionEncoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(SelfAttentionEncoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

"""
Self attention based enccoder needs to be rebuilt, following below ideas
think about how to deal with the hidden state in this case

c = copy.deepcopy
attn = MultiHeadedAttention(h, embd_size)
ff = FeedForwardSublayer(embd_size, d_ff, dropout)
position = PositionalEncoding(embd_size, dropout)
SelfAttention(SelfAttentionEncoderLayer(embd_size, c(attn), c(ff), dropout), N)
"""


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    "Thanks to https://arxiv.org/abs/1607.06450 and http://nlp.seas.harvard.edu/2018/04/03/attention.html#model-architecture"
    
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

######################################################################
# Decoder Components
######################################################################
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

######################################################################
# Full-model ensembling 
######################################################################

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module): 
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1) 