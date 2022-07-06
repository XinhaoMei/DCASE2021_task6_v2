#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import torch
import torch.nn as nn
from models.Encoder import Cnn14_kw, Cnn10_kw
from tools.file_io import load_pickle_file


class Tag_model(nn.Module):

    def __init__(self, config):
        super(Tag_model, self).__init__()

        self.feature_extractor = Cnn14_kw(config)
        pretrained_cnn = torch.load('pretrained_models/audio_encoder/Cnn14.pth')['model']
        dict_new = self.feature_extractor.state_dict().copy()
        trained_list = [i for i in pretrained_cnn.keys()
                        if not ('fc' in i or i.startswith('spec') or i.startswith('logmel'))]
        for i in range(len(trained_list)):
            dict_new[trained_list[i]] = pretrained_cnn[trained_list[i]]
        self.feature_extractor.load_state_dict(dict_new)

        self.vocab = load_pickle_file(f'data/Clotho/pickles/456/tag_vocab_456.p')
        ntokens = len(self.vocab)
        self.audio_linear = nn.Linear(1024, ntokens)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.audio_linear(x)

        return x
