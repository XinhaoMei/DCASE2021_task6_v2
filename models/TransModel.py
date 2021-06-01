#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @ CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from models.Encoder import init_layer, Cnn10, Cnn14
from models.LinearModule import AudioLinear


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """ Container module with an Cnn encoder and a Transformer decoder."""

    def __init__(self, config, words_list, pretrained_cnn=None, pretrained_word_embedding=None):
        super(TransformerModel, self).__init__()
        self.model_type = 'Cnn+Transformer'

        ntoken = len(words_list)

        # setting for CNN
        if config.encoder.model == 'Cnn10':
            self.feature_extractor = Cnn10(config)
        elif config.encoder.model == 'Cnn14':
            self.feature_extractor = Cnn14(config)
        else:
            raise NameError('No such enocder model')

        if pretrained_cnn is not None:
            dict_trained = pretrained_cnn
            dict_new = self.feature_extractor.state_dict().copy()
            trained_list = [i for i in pretrained_cnn.keys()
                            if not ('fc' in i or i.startswith('spec') or i.startswith('logmel'))]
            for i in range(len(trained_list)):
                dict_new[trained_list[i]] = dict_trained[trained_list[i]]
            self.feature_extractor.load_state_dict(dict_new)
        if config.encoder.freeze:
            for name, p in self.feature_extractor.named_parameters():
                if 'fc' not in name:
                    p.requires_grad = False

        # decoder settings
        self.decoder_only = config.decoder.decoder_only
        nhead = config.decoder.nhead       # number of heads in Transformer
        self.nhid = config.decoder.nhid         # number of expected features in decoder inputs
        nlayers = config.decoder.nlayers   # number of sub-decoder-layer in the decoder
        dim_feedforward = config.decoder.dim_feedforward   # dimension of the feedforward model
        activation = config.decoder.activation     # activation function of decoder intermediate layer
        dropout = config.decoder.dropout   # the dropout value

        self.pos_encoder = PositionalEncoding(self.nhid, dropout)
        self.audio_linear = AudioLinear(self.nhid)

        if not self.decoder_only:
            ''' Including transfomer encoder '''
            encoder_layers = TransformerEncoderLayer(self.nhid,
                                                     nhead,
                                                     dim_feedforward,
                                                     dropout,
                                                     activation)
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        decoder_layers = TransformerDecoderLayer(self.nhid,
                                                 nhead,
                                                 dim_feedforward,
                                                 dropout,
                                                 activation)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)

        # linear layers
        self.dec_fc = nn.Linear(self.nhid, ntoken)
        self.generator = nn.Softmax(dim=-1)
        self.word_emb = nn.Embedding(ntoken, self.nhid)

        self.init_weights()

        # setting for pretrained word embedding
        if config.word_embedding.freeze:
            self.word_emb.weight.requires_grad = False
        if pretrained_word_embedding is not None:
            self.word_emb.weight.data = pretrained_word_embedding

    def init_weights(self):
        initrange = 0.1
        self.word_emb.weight.data.uniform_(-initrange, initrange)
        self.dec_fc.bias.data.zero_()
        self.dec_fc.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, src):

        src = self.feature_extractor(src)
        src = self.audio_linear(src)

        if not self.decoder_only:
            src = src * math.sqrt(self.nhid)
            src = self.pos_encoder(src)
            src = self.transformer_encoder(src, None)

        return src

    def decode(self, mem, tgt, mixup_param=None, input_mask=None, target_mask=None, target_padding_mask=None):
        # tgt:(batch_size, T_out)
        # mem:(T_mem, batch_size, nhid)

        tgt = tgt.transpose(0, 1)
        if target_mask is None or target_mask.size()[0] != len(tgt):
            device = tgt.device
            target_mask = self.generate_square_subsequent_mask(len(tgt)).to(device)

        tgt = self.word_emb(tgt) * math.sqrt(self.nhid)

        if self.training and mixup_param is not None:
            lam, index = mixup_param
            tgt = lam * tgt + (1 - lam) * tgt[:, index]

        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, mem, 
                                          memory_mask=input_mask, 
                                          tgt_mask=target_mask, 
                                          tgt_key_padding_mask=target_padding_mask)
        output = self.dec_fc(output)

        return output

    def forward(self, src, tgt, mixup_param=None, input_mask=None, target_mask=None, target_padding_mask=None):

        mem = self.encode(src)
        output = self.decode(mem, tgt, mixup_param,
                            input_mask=input_mask,
                            target_mask=target_mask,
                            target_padding_mask=target_padding_mask)
        return output





