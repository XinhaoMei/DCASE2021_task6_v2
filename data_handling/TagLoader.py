#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import torch
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from tools.file_io import load_pickle_file


class AudioCaptionDataset(Dataset):

    def __init__(self, dataset='Clotho', split='train', task='tagging'):
        """
        load audio clip's waveform and corresponding caption
        Args:
            dataset: 'AudioCaps', 'Clotho
            split: 'train', 'val', 'test'
        """
        super(AudioCaptionDataset, self).__init__()
        self.dataset = dataset
        self.split = split
        self.h5_path = f'data/{dataset}/hdf5s/{split}/{split}.h5'
        with h5py.File(self.h5_path, 'r') as hf:
            self.audio_keys = [audio_name.decode() for audio_name in hf['audio_name'][:]]
            self.audio_lengths = [length for length in hf['audio_length'][:]]

        self.target_dict = load_pickle_file(f'data/Clotho/pickles/456/{self.split}_keywords_dict_456.p')
        self.vocab = load_pickle_file(f'data/Clotho/pickles/456/tag_vocab_456.p')

    def __len__(self):
        return len(self.audio_keys)

    def __getitem__(self, index):

        audio_idx = index
        audio_name = self.audio_keys[audio_idx]
        with h5py.File(self.h5_path, 'r') as hf:
            waveform = hf['waveform'][audio_idx]
        kws = self.target_dict[audio_name]
        target = np.zeros(len(self.vocab))
        if kws != []:
            kws_index = np.array([self.vocab.index(kw) for kw in kws])
            target[kws_index] = 1.
        length = self.audio_lengths[audio_idx]
        return waveform, target, audio_idx, length, audio_name


def collate_fn(batch_data):
    """

    Args:
        batch_data:

    Returns:

    """

    max_audio_length = max([i[3] for i in batch_data])

    # max_audio_length = 44100 * 15

    wav_tensor = []
    for waveform, _, _, _, _ in batch_data:
        if max_audio_length > waveform.shape[0]:
            padding = torch.zeros(max_audio_length - waveform.shape[0]).float()
            temp_audio = torch.cat([torch.from_numpy(waveform).float(), padding])
        else:
            temp_audio = torch.from_numpy(waveform[:max_audio_length]).float()
        wav_tensor.append(temp_audio.unsqueeze_(0))

    wavs_tensor = torch.cat(wav_tensor)
    target = torch.tensor(np.array([i[1] for i in batch_data]))
    audio_ids = torch.Tensor([i[2] for i in batch_data])
    audio_names = [i[4] for i in batch_data]

    return wavs_tensor, target, audio_ids, audio_names


def get_dataloader(split, config):
    dataset = AudioCaptionDataset(config.dataset, split, 'tagging')
    if split == 'train':
        shuffle = True
        drop_last = False
    else:
        shuffle = False
        drop_last = False

    return DataLoader(dataset=dataset,
                      batch_size=config.data.batch_size,
                      shuffle=shuffle,
                      drop_last=drop_last,
                      num_workers=config.data.num_workers,
                      collate_fn=collate_fn)
