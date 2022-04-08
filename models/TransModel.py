#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @ CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer,\
TransformerDecoder, TransformerDecoderLayer
from models.Encoder import Cnn10, Cnn14
from tools.file_io import load_pickle_file
from tools.utils import align_word_embedding
from models.net_vlad import NetVLAD


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).

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
            x: the sequence fed to the positional audio_encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """ Container module with an Cnn audio_encoder and a Transformer decoder."""

    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.model_type = 'Cnn+Transformer'

        vocabulary = load_pickle_file(config.path.vocabulary.format(config.dataset))
        ntoken = len(vocabulary)

        # setting for CNN
        if config.encoder.model == 'Cnn10':
            self.feature_extractor = Cnn10(config)
        elif config.encoder.model == 'Cnn14':
            self.feature_extractor = Cnn14(config)
        else:
            raise NameError('No such enocder model')

        if config.encoder.pretrained:
            pretrained_cnn = torch.load('pretrained_models/audio_encoder/{}.pth'.
                                        format(config.encoder.model))['model']
            dict_new = self.feature_extractor.state_dict().copy()
            trained_list = [i for i in pretrained_cnn.keys()
                            if not ('fc' in i or i.startswith('spec') or i.startswith('logmel'))]
            for i in range(len(trained_list)):
                dict_new[trained_list[i]] = pretrained_cnn[trained_list[i]]
            self.feature_extractor.load_state_dict(dict_new)
        if config.encoder.freeze:
            for name, p in self.feature_extractor.named_parameters():
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

        if not self.decoder_only:
            ''' Including transfomer audio_encoder '''
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
        self.audio_linear = nn.Linear(1024, self.nhid, bias=True)
        self.dec_fc = nn.Linear(self.nhid, ntoken)
        self.generator = nn.Softmax(dim=-1)
        self.word_emb = nn.Embedding(ntoken, self.nhid)

        self.is_vlad = config.training.vlad
        if self.is_vlad:
            self.net_vlad = NetVLAD(cluster_size=20, feature_size=128)

        self.init_weights()

        # setting for pretrained word embedding
        if config.word_embedding.freeze:
            self.word_emb.weight.requires_grad = False
        if config.word_embedding.pretrained:
            self.word_emb.weight.data = align_word_embedding(config.path.vocabulary.format(config.dataset),
                                                             config.path.word2vec, config.decoder.nhid)

    def init_weights(self):
        initrange = 0.1
        self.word_emb.weight.data.uniform_(-initrange, initrange)
        init_layer(self.audio_linear)
        init_layer(self.dec_fc)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, src):

        src = self.feature_extractor(src)  # (time, batch, feature)
        src = F.relu_(self.audio_linear(src))
        src = F.dropout(src, p=0.2, training=self.training)

        if self.is_vlad:
            src = src.transpose(1, 0)
            src = self.net_vlad(src)
            src = src.transpose(1, 0)

        if not self.decoder_only:
            src = src * math.sqrt(self.nhid)
            src = self.pos_encoder(src)
            src = self.transformer_encoder(src, None)

        return src

    def decode(self, mem, tgt, input_mask=None, target_mask=None, target_padding_mask=None):
        # tgt:(batch_size, T_out)
        # mem:(T_mem, batch_size, nhid)

        tgt = tgt.transpose(0, 1)
        if target_mask is None or target_mask.size()[0] != len(tgt):
            device = tgt.device
            target_mask = self.generate_square_subsequent_mask(len(tgt)).to(device)

        tgt = self.word_emb(tgt) * math.sqrt(self.nhid)

        tgt = self.pos_encoder(tgt)
        output = self.transformer_decoder(tgt, mem,
                                          memory_mask=input_mask,
                                          tgt_mask=target_mask,
                                          tgt_key_padding_mask=target_padding_mask)
        output = self.dec_fc(output)

        return output

    def forward(self, src, tgt, input_mask=None, target_mask=None, target_padding_mask=None):

        mem = self.encode(src)
        output = self.decode(mem, tgt,
                             input_mask=input_mask,
                             target_mask=target_mask,
                             target_padding_mask=target_padding_mask)
        return output





