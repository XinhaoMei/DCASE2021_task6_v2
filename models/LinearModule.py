#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk

import torch.nn as nn
import torch.nn.functional as F


def init_layer(layer):
        """ Initialize a Linear or Convolutional layer. """
        nn.init.xavier_uniform_(layer.weight)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                layer.bias.data.fill_(0.)

class AudioLinear(nn.Module):

    def __init__(self, nhid):
        super(AudioLinear, self).__init__()

        # self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc2 = nn.Linear(512, nhid, bias=True)

        self.init_weights()

    def init_weights(self):
        # init_layer(self.fc1)
        init_layer(self.fc2)

    def forward(self, x):
        # input: batch x channel x time_frames x frequency

        # x = input.permute(1, 0, 2)
        # x = F.relu_(self.fc1(x))
        # x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu_(self.fc2(x))  # time x batch x nhid
        # x = x.permute(1, 0, 2)
        return x
 