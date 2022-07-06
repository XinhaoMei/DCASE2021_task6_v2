#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey, Gengyun Chen @ NJPTU
# @E-mail  : x.mei@surrey.ac.uk

''' Code for rl adapted from https://github.com/wsntxxn/DCASE2020T6 '''

import torch
import numpy as np


def scst_sample(model, src, keyword=None, sos_ind=0, eos_ind=9, max_len=30):
    batch_size = src.size()[0]
    encoded_feats = model.encode(src)
    ys = torch.ones(batch_size, 1).fill_(sos_ind).long().to(src.device)

    sampled_logprobs = torch.zeros(batch_size, max_len).to(src.device)

    for i in range(max_len):
        out = model.decode(encoded_feats, ys, keyword=keyword)
        logprobs = torch.log_softmax(out[i], dim=1)
        prob_prev = torch.exp(logprobs)
        w_t = torch.multinomial(prob_prev, 1)
        logprobs = logprobs.gather(1, w_t).squeeze(1)
        sampled_logprobs[:, i] = logprobs
        ys = torch.cat((ys, w_t), dim=1)
    return ys, sampled_logprobs


def pack_sample(captions, sos_ind, eos_ind):

    true_lengths = []
    max_length = 0

    if (np.array(captions[:, 0].cpu()) == sos_ind).all():
        captions = captions[:, 1:]
    for i in range(captions.shape[0]):  # batch
        for j in range(captions.shape[1]):
            if captions[i][j] == eos_ind:
                captions[i, j:] = eos_ind
                true_lengths.append(j + 1)
                if j >= max_length:
                    max_length = j + 1
                break
            elif j == captions.shape[1] - 1 and captions[i][j] != eos_ind:
                true_lengths.append(j + 1)
                # captions[i][j] = eos_ind
                max_length = j + 1

    # captions = captions[:, :max_length]
    return captions, true_lengths


def get_self_critical_reward(greedy_sample, sampled, tgt, words_list, sos_ind, eos_ind):
    greedy_sample = greedy_sample.long().cpu().numpy()

    sampled = sampled.long().cpu().numpy()

    greedy_score = compute_batch_score(greedy_sample, tgt, words_list, sos_ind, eos_ind)
    sampled_score = compute_batch_score(sampled, tgt, words_list, sos_ind, eos_ind)

    reward = sampled_score - greedy_score

    return reward, sampled_score


def compute_batch_score(sample, tgt, words_list, sos_ind=0, eos_ind=9):
    from coco_caption.pycocoevalcap.cider.cider import Cider
    scorer = Cider()
    # tgt = tgt[:, 1:]
    dict4pred = {}
    dict4tgt = {}
    batch_size = len(tgt)
    for i in range(batch_size):
        candidate = []
        for idx in sample[i]:
            if idx == eos_ind:  # <eos>
                break
            elif idx == sos_ind:
                continue
            candidate.append(words_list[idx])
        dict4pred[i] = [" ".join(candidate)]
    for i in range(batch_size):
        dict4tgt[i] = [tgt[i]]

    score, scores = scorer.compute_score(dict4tgt, dict4pred)
    return scores
