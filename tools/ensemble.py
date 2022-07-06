#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import torch
import operator
import torch.nn as nn
from loguru import logger
import torch.nn.functional as F
from queue import PriorityQueue
from tools.beam import BeamSearchNode
from torch.nn.utils.rnn import pad_sequence


class EnsembleModel(nn.Module):

    def __init__(self, model_list, sos_ind, eos_ind):
        super(EnsembleModel, self).__init__()

        self.model_nums = len(model_list)
        self.model_list = model_list
        self.sos_ind = sos_ind
        self.eos_ind = eos_ind
        self.max_len = 30
        self.logger = logger.bind(indent=1)
        self.logger.info(f'{self.model_nums} models for ensemble.')

    def greedy_decode(self, x, keyword=None):

        device = x.device
        encoded_feats = []
        # get audio features extracted by each model
        for i, model in enumerate(self.model_list):
            model.eval()
            x_feats = model.encode(x)
            encoded_feats.append(x_feats)
        batch_size = x.shape[0]
        ys = torch.ones(batch_size, 1).fill_(self.sos_ind).long().to(device)

        for t in range(self.max_len):
            word_probs = 0.
            for i, model in enumerate(self.model_list):
                feats = encoded_feats[i]
                if model.is_keywords:
                    kw = keyword
                else:
                    kw = None
                out = model.decode(feats, ys, keyword=kw)  # T_out, batch_size, ntoken
                log_prob = F.log_softmax(out[-1, :], dim=-1)
                prob = torch.exp(log_prob)
                word_probs += prob
            word_probs = word_probs / self.model_nums
            word = torch.argmax(word_probs, dim=1).unsqueeze(1)
            ys = torch.cat([ys, word], dim=1)
        return ys

    def beam_decode(self, x, keyword=None, beam_width=3, top_k=1):
        device = x.device
        encoded_feats = []
        # get audio features extracted by each model
        for i, model in enumerate(self.model_list):
            x_feats = model.encode(x)
            encoded_feats.append(x_feats)
        batch_size = x.shape[0]

        decoded_batch = []

        for idx in range(batch_size):

            decoder_input = torch.LongTensor([[self.sos_ind]]).to(device)

            endnodes = []

            # starting nodel
            start_node = BeamSearchNode(None, decoder_input, 0, 1)
            nodes = PriorityQueue()
            temp_nodes = PriorityQueue()

            nodes.put((-start_node.eval(), start_node))

            keeped_node_width = beam_width
            time_step = 0

            while time_step < self.max_len:
                if len(endnodes) >= beam_width:
                    break

                # remove and get the best node from the queue
                # best means with the - log_p is lowest
                score, n = nodes.get()
                decoder_input = n.wordid  # (1, seq_len) words

                word_probs = 0.

                for i, model in enumerate(self.model_list):
                    feats = encoded_feats[i][:, idx, :].unsqueeze(1)
                    if model.is_keywords:
                        kw = keyword[idx].unsqueeze(0)
                    else:
                        kw = None
                    out = model.decode(feats, decoder_input, keyword=kw)  # T_out, batch_size, ntoken
                    log_prob = F.log_softmax(out[-1, :], dim=-1)
                    prob = torch.exp(log_prob)
                    word_probs += prob
                word_probs = word_probs / self.model_nums
                word_log_probs = torch.log(word_probs)
                log_prob, indexes = torch.topk(word_log_probs, beam_width)

                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(1, -1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(n, torch.cat((decoder_input, decoded_t), dim=1), n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    temp_nodes.put((score, node))

                if nodes.qsize() == 0:
                    for _ in range(beam_width):
                        score, node = temp_nodes.get()
                        if node.wordid[0, -1].item() == self.eos_ind and n.prevNode is not None:
                            endnodes.append((score, node))
                            keeped_node_width -= 1
                        else:
                            nodes.put((score, node))
                    time_step += 1
                    if time_step == self.max_len and keeped_node_width != 0:
                        for _ in range(keeped_node_width):
                            endnodes.append(nodes.get())
                    temp_nodes = PriorityQueue()
                else:
                    continue

            utterances = []
            for score, n in sorted(endnodes, key=operator.itemgetter(0)):
                utterances.append(n.wordid[0, :])
            for i in range(top_k):
                decoded_batch.append(utterances[i])

        return pad_sequence(decoded_batch, batch_first=True, padding_value=self.eos_ind)