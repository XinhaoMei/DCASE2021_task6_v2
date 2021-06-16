#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey, Gengyun Chen @ NJPTU
# @E-mail  : x.mei@surrey.ac.uk

''' Code for rl adapted from https://github.com/wsntxxn/DCASE2020T6 '''

import torch


def scst_sample(model, src, tgt, eos_ind=9, sos_ind=0, max_len=30):

    batch_size = src.size()[0]
    mem = model.encode(src)
    ys = torch.ones(batch_size, 1).fill_(sos_ind).long().to(src.device)

    sampled_logprobs = torch.zeros(batch_size, max_len).to(src.device)

    for i in range(max_len - 1):
        target_mask = model.generate_square_subsequent_mask(ys.shape[1]).to(src.device)

        out = model.decode(mem, ys, target_mask=target_mask)
        logprobs = torch.log_softmax(out[i], dim=1)
        prob_prev = torch.exp(logprobs)
        w_t = torch.multinomial(prob_prev, 1)
        logprobs = logprobs.gather(1, w_t).squeeze(1)
        sampled_logprobs[:, i] = logprobs
        ys = torch.cat((ys, w_t), dim=1)
    return ys, sampled_logprobs


def get_self_critical_reward(greedy_sample, sampled, tgt, words_list, sos_ind, eos_ind):
    greedy_sample = greedy_sample.long().cpu().numpy()

    sampled = sampled.long().cpu().numpy()

    greedy_score = compute_batch_score(greedy_sample, tgt, words_list, eos_ind, sos_ind)
    sampled_score = compute_batch_score(sampled, tgt, words_list, eos_ind, sos_ind)

    reward = sampled_score - greedy_score

    return reward, sampled_score


def compute_batch_score(sample, tgt, words_list, eos_ind=9, sos_ind=0):
    from coco_caption.pycocoevalcap.cider.cider import Cider
    scorer = Cider()
    tgt = tgt[:, 1:]
    dict4pred = {}
    dict4tgt = {}
    batch_size = tgt.size()[0]
    for i in range(batch_size):
        candidate = []
        for idx in sample[i]:
            if idx == eos_ind:  # <eos>
                break
            elif idx == sos_ind:
                continue
            candidate.append(words_list[idx])
        dict4pred[i] = [" ".join(candidate)]
    # print("dict4pred:",dict4pred)
    for i in range(batch_size):
        caption = []
        for idx in tgt[i]:
            if idx == eos_ind:  # <eos>
                break
            caption.append(words_list[idx])
        dict4tgt[i] = [" ".join(caption)]

    score, scores = scorer.compute_score(dict4tgt, dict4pred)
    return scores
