#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from loguru import logger
from gensim.models.word2vec import Word2Vec


def setup_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def mixup_data(src, tgt, alpha=1):

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    index = torch.randperm(src.size()[0])
    mixed_src = lam * src + (1 - lam) * src[index, :]
    tgt_a, tgt_b = tgt, tgt[index]
    return mixed_src, tgt_a, tgt_b, lam, index


def rotation_logger(x, y):
    """Callable to determine the rotation of files in logger.

    :param x: Str to be logged.
    :type x: loguru._handler.StrRecord
    :param y: File used for logging.
    :type y: _io.TextIOWrapper
    :return: Shall we switch to a new file?
    :rtype: bool
    """
    return 'Captions start' in x


def set_tgt_padding_mask(tgt, tgt_len):
    # tgt: (batch_size, max_len)
    # tgt_len: list() length for each caption in the batch
    batch_size = tgt.shape[0]
    max_len = tgt.shape[1]
    mask = torch.zeros(tgt.shape).type_as(tgt).to(tgt.device)
    for i in range(batch_size):
        num_pad = max_len - tgt_len[i]
        mask[i][max_len - num_pad:] = 1

    mask = mask.float().masked_fill(mask == 1, True).masked_fill(mask == 0, False).bool()

    return mask


def greedy_decode(model, src, max_len=30, sos_ind=0):

    batch_size = src.shape[0]
    mem = model.encode(src)

    ys = torch.ones(batch_size, 1).fill_(sos_ind).long().to(src.device)

    for i in range(max_len - 1):
        target_mask = model.generate_square_subsequent_mask(ys.shape[1]).to(src.device)
        out = model.decode(mem, ys, target_mask=target_mask)  # T_out, batch_size, ntoken
        prob = model.generator(out[-1, :])
        next_word = torch.argmax(prob, dim=1)
        next_word = next_word.unsqueeze(1)
        ys = torch.cat([ys, next_word], dim=1)
    return ys


def decode_output(predicted_output, ref_captions, file_names, words_list, output_dir, beam=False):

    if beam:
        caption_logger = logger.bind(indent=3)
        caption_logger.info('Captions start')
        caption_logger.info('Beam search:')
    else:
        caption_logger = logger.bind(indent=2)
        caption_logger.info('Captions start')
        caption_logger.info('Greedy search:')

    captions_pred, captions_gt, f_names = [], [], []


    for pred_words, ref_cap, f_name in zip(predicted_output, ref_captions, file_names):
        pred_cap = [words_list[i] for i in pred_words]

        ref_cap = ref_cap.strip().split()
        ref_cap = ref_cap[1:]
        ref_cap = ref_cap[:-1]

        try:
            pred_cap = pred_cap[:pred_cap.index('<eos>')]
        except ValueError:
            pass

        pred_cap = ' '.join(pred_cap)
        gt_cap = ' '.join(ref_cap)

        if f_name not in f_names:
            f_names.append(f_name)
            captions_pred.append({'file_name':f_name,'caption_predicted': pred_cap})
            captions_gt.append({'file_name':f_name,'caption_1': gt_cap})
        else:
            for index, gt_dict in enumerate(captions_gt):
                if f_name == gt_dict['file_name']:
                    len_captions = len([i_c for i_c in gt_dict.keys()
                                        if i_c.startswith('caption_')]) + 1
                    gt_dict.update({f'caption_{len_captions}': gt_cap})
                    captions_gt[index] = gt_dict
                    break

        log_strings = [f'Captions for file {f_name}:',
                       f'\t Predicted caption: {pred_cap}',
                       f'\t Original caption: {gt_cap}\n\n']

        [caption_logger.info(log_string)
         for log_string in log_strings]

    return captions_pred, captions_gt


def align_word_embedding(words_list, model_path, nhid):

    w2v_model = Word2Vec.load(model_path)
    ntoken = len(words_list)
    weights = torch.randn(ntoken, nhid)
    for i, word in enumerate(words_list):
        embedding = w2v_model.wv[word]
        weights[i] = torch.from_numpy(embedding).float()
    return weights


# https://github.com/pytorch/pytorch/issues/7455#issuecomment-513062631
# When smoothing=0.0, the output is almost the same as nn.CrossEntropyLoss
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, ignore_index=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            if self.ignore_index:
                true_dist[:, self.ignore_index] = 0
                mask = torch.nonzero(target.data == self.ignore_index)
                if mask.dim() > 0:
                    true_dist.index_fill_(0, mask.squeeze(), 0.0)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class Beam:
    """
    The beam class for handling beam search.
    partly adapted from
    https://github.com/OpenNMT/OpenNMT-py/blob/195f5ae17f572c22ff5229e52c2dd2254ad4e3db/onmt/translate/beam.py
    There are some place which needs improvement:
    1. The prev_beam should be separated as prev_beam and beam_score.
    The prev_beam should be a tensor and beam_score should be a numpy array,
    such that the beam advance() method could speeds up.
    2. Do not support advance function like length penalty.
    3. If the beam is done searching, it could quit from further computation.
    In here, an eos is simply appended and still go through the model in next iteration.
    """

    def __init__(self, beam_size, device, start_symbol_ind, end_symbol_ind):
        self.device = device
        self.beam_size = beam_size
        self.prev_beam = [[torch.ones(1).fill_(start_symbol_ind).long().to(device), 0]]
        self.start_symbol_ind = start_symbol_ind
        self.end_symbol_ind = end_symbol_ind
        self.eos_top = False
        self.finished = []
        self.first_time = True

    def advance(self, word_probs, first_time):  # word_probs: (beam_size, ntoken) or (1, ntoken) for the first time.

        if self.done():
            # if current beam is done, just add eos to the beam.
            for b in self.prev_beam:
                b[0] = torch.cat([b[0], torch.tensor(self.end_symbol_ind).unsqueeze(0).to(self.device)])
            return

        # in first time, the beam need not to align with each index.
        if first_time:  # word_probs:(1, ntoken)
            score, index = word_probs.squeeze(0).topk(self.beam_size, 0, True, True)  # get the initial topk
            self.prev_beam = []
            for s, ind in zip(score, index):
                # initialize each beam
                self.prev_beam.append([torch.tensor([self.start_symbol_ind, ind]).long().to(self.device), s.item()])
                self.prev_beam = self.sort_beam(self.prev_beam)
        else:  # word_probs:(beam_size, ntoken)
            score, index = word_probs.topk(self.beam_size, 1, True, True)  # get topk
            current_beam = [[b[0].clone().detach(), b[1]] for b in self.prev_beam for i in range(self.beam_size)]
            # repeat each beam beam_size times for global score comparison, need to detach each tensor copied.
            i = 0
            for score_beam, index_beam in zip(score, index):  # get topk scores and corresponding index for each beam
                for s, ind in zip(score_beam, index_beam):
                    current_beam[i][0] = torch.cat([current_beam[i][0], ind.unsqueeze(0)])
                    # append current index to beam
                    current_beam[i][1] += s.item()  # add the score
                    i += 1

            current_beam = self.sort_beam(current_beam)  # sort current beam
            if current_beam[0][0][-1] == self.end_symbol_ind:  # check if the top beam ends with eos
                self.eos_top = True

            # check for eos node and added them to finished beam list.
            # In the end, delete those nodes and do not let them have child note.
            delete_beam_index = []
            for i in range(len(current_beam)):
                if current_beam[i][0][-1] == self.end_symbol_ind:
                    delete_beam_index.append(i)
            for i in sorted(delete_beam_index, reverse=True):
                self.finished.append(current_beam[i])
                del current_beam[i]

            self.prev_beam = current_beam[:self.beam_size]  # get top beam_size beam
            # print(self.prev_beam)

    def done(self):
        # check if current beam is done searching
        return self.eos_top and len(self.finished) >= 1

    def get_current_state(self):
        # get current beams
        # print(self.prev_beam)
        return torch.stack([b[0] for b in self.prev_beam])

    def get_output(self):
        if len(self.finished) > 0:
            # sort the finished beam and return the sentence with the highest score.
            self.finished = self.sort_beam(self.finished)
            return self.finished[0][0]
        else:
            self.prev_beam = self.sort_beam(self.prev_beam)
            return self.prev_beam[0][0]

    def sort_beam(self, beam):
        # sort the beam according to the score
        return sorted(beam, key=lambda x: x[1], reverse=True)


def beam_search(model, src, max_len=30, start_symbol_ind=0, end_symbol_ind=9, beam_size=1):
    device = src.device  # src:(batch_size,T_in,feature_dim)
    batch_size = src.size()[0]
    memory = model.encode(src)  # memory:(T_mem,batch_size,nhid)
    # ys = torch.ones(batch_size, 1).fill_(start_symbol_ind).long().to(device)  # ys_0: (batch_size,T_pred=1)

    first_time = True

    beam = [Beam(beam_size, device, start_symbol_ind, end_symbol_ind) for _ in range(batch_size)]  # a batch of beams

    for i in range(max_len):
        # end if all beams are done, or exceeds max length
        if all((b.done() for b in beam)):
            break

        # get current input
        ys = torch.cat([b.get_current_state() for b in beam], dim=0).to(device).requires_grad_(False)

        # get input mask
        target_mask = model.generate_square_subsequent_mask(ys.size()[1]).to(device)
        out = model.decode(memory, ys, target_mask=target_mask)  # (T_out, batch_size, ntoken) for first time,
        # (T_out, batch_size*beam_size, ntoken) in other times
        out = F.log_softmax(out[-1, :], dim=-1)  # (batch_size, ntoken) for first time,
        # (batch_size*beam_size, ntoken) in other times

        beam_batch = 1 if first_time else beam_size
        # in the first run, a slice of 1 should be taken for each beam,
        # later, a slice of [beam_size] need to be taken for each beam.
        for j, b in enumerate(beam):
            b.advance(out[j * beam_batch:(j + 1) * beam_batch, :], first_time)  # update each beam

        if first_time:
            first_time = False  # reset the flag
            # after the first run, the beam expands, so the memory needs to expands too.
            memory = memory.repeat_interleave(beam_size, dim=1)

    output = [b.get_output() for b in beam]
    return output

