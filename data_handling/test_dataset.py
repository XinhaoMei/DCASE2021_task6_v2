#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import torch
import librosa
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


class TestDataset(Dataset):

    def __init__(self, load_into_memory):
        super(TestDataset, self).__init__()
        data_dir = Path('data/test')

        self.examples = sorted(data_dir.iterdir())
        self.load_into_memory = load_into_memory

        if self.load_into_memory:
            self.examples = [(librosa.load(file, sr=44100)[0], str(file).split('/')[-1]) for file in self.examples]

    def __len__(self):

        return len(self.examples)

    def __getitem__(self, index):
        item = self.examples[index]
        if not self.load_into_memory:
            item = [librosa.load(item, sr=44100)[0], str(item).split('/')[-1]]

        feature = item[0]
        file_name = item[1]

        return feature, file_name


def get_test_loader(load_into_memory,
                    batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=1):
    dataset = TestDataset(load_into_memory)

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      drop_last=drop_last,
                      num_workers=num_workers,
                      collate_fn=test_collate_fn)

def test_collate_fn(batch):

    max_audio_time_steps = max(i[0].shape[0] for i in batch)

    audio_tensor = []

    for audio, _ in batch:
        if max_audio_time_steps > audio.shape[0]:
            padding = torch.zeros(max_audio_time_steps - audio.shape[0]).float()
            data = [torch.from_numpy(audio).float()]
            data.append(padding)
            temp_audio = torch.cat(data)
        else:
            temp_audio = torch.from_numpy(audio[:max_audio_time_steps]).float()
        audio_tensor.append(temp_audio.unsqueeze_(0))

    audio_tensor = torch.cat(audio_tensor)

    file_names = [i[1] for i in batch]

    return audio_tensor, file_names

