#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


"""
Adapted from https://github.com/budzianowski/PyTorch-Beam-Search-Decoding,
and https://github.com/haantran96/wavetransformer/blob/main/modules/beam.py
"""

import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue
from torch.nn.utils.rnn import pad_sequence


class BeamSearchNode(object):
    def __init__(self, previousNode, wordId, logProb, length):
        """
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        """
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def __lt__(self, other):
        return self.logp < other.logp

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


def beam_decode(src, model, sos_ind, eos_ind, beam_width=5, top_k=1, max_len=30):

    decoded_batch = []

    device = src.device
    batch_size = src.shape[0]

    # audio feature sequece extracted by the audio_encoder
    encoded_feats = model.encode(src)

    # decoding goes sentence by sentence
    for idx in range(batch_size):

        encoded_feat = encoded_feats[:, idx, :].unsqueeze(1)

        decoder_input = torch.LongTensor([[sos_ind]]).to(device)

        endnodes = []

        # starting nodel
        start_node = BeamSearchNode(None, decoder_input, 0, 1)
        nodes = PriorityQueue()
        temp_nodes = PriorityQueue()

        nodes.put((-start_node.eval(), start_node))

        keeped_node_width = beam_width
        time_step = 0

        # start beam search
        while time_step < max_len:

            if len(endnodes) >= beam_width:
                break

            # remove and get the best node from the queue
            # best means with the - log_p is lowest
            score, n = nodes.get()
            decoder_input = n.wordid  # (1, seq_len) words

            decoder_output = model.decode(encoded_feat, decoder_input)
            log_prob = F.log_softmax(decoder_output[-1, :], dim=-1)

            log_prob, indexes = torch.topk(log_prob, beam_width)

            for new_k in range(beam_width):
                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()

                node = BeamSearchNode(n, torch.cat((decoder_input, decoded_t), dim=1), n.logp + log_p, n.leng + 1)
                score = -node.eval()
                temp_nodes.put((score, node))

            if nodes.qsize() == 0:
                for _ in range(beam_width):
                    score, node = temp_nodes.get()
                    if node.wordid[0, -1].item() == eos_ind and n.prevNode is not None:
                        endnodes.append((score, node))
                        keeped_node_width -= 1
                    else:
                        nodes.put((score, node))
                time_step += 1
                if time_step == max_len and keeped_node_width != 0:
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

    return pad_sequence(decoded_batch, batch_first=True, padding_value=eos_ind)
