#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import torch
import librosa
from re import sub
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tools.file_io import load_picke_file, load_csv_file


def _sentence_process(sentence, add_specials=False):

    # transform to lower case
    sentence = sentence.lower()

    if add_specials:
        sentence = '<sos> {} <eos>'.format(sentence)

    # remove any forgotten space before punctuation and double space
    sentence = sub(r'\s([,.!?;:"](?:\s|$))', r'\1', sentence).replace('  ', ' ')

    # remove punctuations
    sentence = sub('[,.!?;:\"]', ' ', sentence).replace('  ', ' ')

    return sentence


class AudioCapsDataset(Dataset):

    def __init__(self, split):
        super(AudioCapsDataset, self).__init__()
        self.wav_path = '/vol/research/AAC_CVSSP_research/AudioCaps/data/' + split
        csv_path = 'audiocaps/csv_files/new_' + split + '.csv'
        vocabulary_path = 'audiocaps/pickles/words_list.p'

        self.vocabulary = load_picke_file(vocabulary_path)
        self.examples = load_csv_file(csv_path)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):

        ex = self.examples[item]
        file_name = ex['file_name']
        audio_path = self.wav_path + '/' + file_name
        audio = librosa.load(audio_path, sr=32000)[0]
        caption = _sentence_process(ex['caption'], add_specials=True)
        words = caption.strip().split()
        words_indexs = np.array([self.vocabulary.index(word) for word in words])
        caption_len = len(words_indexs)

        return audio, words_indexs, file_name, caption_len, caption


def get_audiocaps_loader(split,
                         batch_size,
                         shuffle=False,
                         drop_last=False,
                         num_workers=1):
    dataset = AudioCapsDataset(split)

    return DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=shuffle, drop_last=drop_last,
                      num_workers=num_workers, collate_fn=audiocaps_collate_fn)


def audiocaps_collate_fn(batch):

    max_audio_time_steps = 32000 * 10
    max_caption_length = max(i[1].shape[0] for i in batch)

    eos_token = batch[0][1][-1]

    audio_tensor, words_tensor = [], []

    for audio, words_indexs, _, _, _ in batch:
        if max_audio_time_steps >= audio.shape[0]:
            padding = torch.zeros(max_audio_time_steps - audio.shape[0]).float()
            data = [torch.from_numpy(audio).float()]
            data.append(padding)
            temp_audio = torch.cat(data)
        else:
            temp_audio = torch.from_numpy(audio[:max_audio_time_steps]).float()
        audio_tensor.append(temp_audio.unsqueeze_(0))

        if max_caption_length >= words_indexs.shape[0]:
            padding = torch.ones(max_caption_length - len(words_indexs)).mul(eos_token).long()
            data = [torch.from_numpy(words_indexs).long()]
            data.append(padding)
            tmp_words_indexs = torch.cat(data)
        else:
            tmp_words_indexs = torch.from_numpy(words_indexs[:max_caption_length]).long()
        words_tensor.append(tmp_words_indexs.unsqueeze_(0))

    audio_tensor = torch.cat(audio_tensor)
    words_tensor = torch.cat(words_tensor)

    file_names = [i[2] for i in batch]
    caption_lens = [i[3] for i in batch]
    captions = [i[4] for i in batch]

    return audio_tensor, words_tensor, file_names, caption_lens, captions
