#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import random
import torch
import torch.nn as nn
import numpy as np
from tools.file_io import load_pickle_file
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer


class WordTokenizer(nn.Module):
    """
    Tokenizer using own vocabulary.
    Convert each word to its index and pad them as a batch.
    """

    def __init__(self, config):
        super(WordTokenizer, self).__init__()
        dataset = config.dataset
        self.is_sos_token = config.sos_token  # append '<sos>' at the begging of each sentence
        self.vocabulary = load_pickle_file(config.path.vocabulary.format(dataset))
        self.eos_index = self.vocabulary.index('<eos>')
        self.bert_pretrain = config.bert_pretrain
        if self.bert_pretrain:
            self.mask_token = self.vocabulary.index('<mask>')

    def forward(self, inputs):
        # inputs: captions, [str, str, str, ...]
        batch_size = len(inputs)
        if self.is_sos_token:
            inputs = ['<sos> {} <eos>'.format(cap) for cap in inputs]
        else:
            inputs = ['{} <eos>'.format(cap) for cap in inputs]
        captions = [cap.strip().split() for cap in inputs]
        captions_ind = []
        caption_lengths = []
        captions_target = []
        for cap in captions:
            cap_index = [self.vocabulary.index(word) if word in self.vocabulary
                         else self.vocabulary.index('<ukn>')
                         for word in cap]
            caption_lengths.append(len(cap_index))
            if self.bert_pretrain:
                cap_index, target = self.mask_word(cap_index)
                captions_ind.append(torch.tensor(cap_index))
                captions_target.append(torch.tensor(target))
            else:
                captions_ind.append(torch.tensor(cap_index))
        # padding it captions to a tensor of batch
        index_tensor = pad_sequence(captions_ind,
                                    batch_first=True,
                                    padding_value=self.eos_index)

        return index_tensor, caption_lengths


# class BERTTokenizer(nn.Module):
#
#     def __init__(self):
#         super(BERTTokenizer, self).__init__()
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
#     def forward(self, captions):
#         tokenized = self.tokenizer(captions, add_special_tokens=True,
#                                    padding=True, return_tensors='pt')
#         input_ids = tokenized['input_ids']
#         attention_mask = tokenized['attention_mask']
#
#         return input_ids, attention_mask
