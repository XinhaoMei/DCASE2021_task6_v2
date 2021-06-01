#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


class ClothoDataset(Dataset):

    def __init__(self, split,
                 input_field_name,
                 load_into_memory):

        super(ClothoDataset, self).__init__()
        split_dir = Path('data/data_splits', split)

        self.examples = sorted(split_dir.iterdir())
        self.input_field_name = input_field_name
        self.output_field_name = 'words_indexs'
        self.load_into_memory = load_into_memory

        if load_into_memory:
            self.examples = [np.load(str(file), allow_pickle=True) for file in self.examples]

    def __len__(self):

        return len(self.examples)

    def __getitem__(self, index):

        item = self.examples[index]
        if not self.load_into_memory:
            item = np.load(str(item), allow_pickle=True)

        feature = item[self.input_field_name].item()  # waveform or log melspectorgram
        words_indexs = item[self.output_field_name].item()
        file_name = str(item['file_name'].item())
        caption_len = len(words_indexs)
        caption = str(item['caption'].item())

        return feature, words_indexs, file_name, caption_len, caption


def get_clotho_loader(split,
                      input_field_name,
                      load_into_memory,
                      batch_size,
                      shuffle=False,
                      drop_last=False,
                      num_workers=1):
    dataset = ClothoDataset(split, input_field_name, load_into_memory)
    if input_field_name == 'audio_data':
        return DataLoader(dataset=dataset, batch_size=batch_size,
                          shuffle=shuffle, drop_last=drop_last,
                          num_workers=num_workers, collate_fn=clotho_collate_fn_audio)
    else:
        return DataLoader(dataset=dataset, batch_size=batch_size,
                          shuffle=shuffle, drop_last=drop_last,
                          num_workers=num_workers, collate_fn=clotho_collate_fn)


def clotho_collate_fn(batch):

    max_feature_time_steps = max(i[0].shape[0] for i in batch)
    max_caption_length = max(i[1].shape[0] for i in batch)

    feature_number = batch[0][0].shape[-1]
    eos_token = batch[0][1][-1]

    feature_tensor, words_tensor = [], []

    for feature, words_indexs, _, _, _ in batch:
        if max_feature_time_steps > feature.shape[0]:
            padding = torch.zeros(max_feature_time_steps - feature.shape[0], feature_number).float()
            data = [torch.from_numpy(feature).float()]
            data.append(padding)
            temp_feature = torch.cat(data)
        else:
            temp_feature = torch.from_numpy(feature[:max_feature_time_steps, :]).float()
        feature_tensor.append(temp_feature.unsqueeze_(0))

        if max_caption_length > words_indexs.shape[0]:
            padding = torch.ones(max_caption_length - len(words_indexs)).mul(eos_token).long()
            data = [torch.from_numpy(words_indexs).long()]
            data.append(padding)
            tmp_words_indexs = torch.cat(data)
        else:
            tmp_words_indexs = torch.from_numpy(words_indexs[:max_caption_length]).long()
        words_tensor.append(tmp_words_indexs.unsqueeze_(0))

    feature_tensor = torch.cat(feature_tensor)
    words_tensor = torch.cat(words_tensor)

    file_names = [i[2] for i in batch]
    caption_lens = [i[3] for i in batch]
    captions = [i[4] for i in batch]

    return feature_tensor, words_tensor, file_names, caption_lens, captions

def clotho_collate_fn_audio(batch):

    max_audio_time_steps = max(i[0].shape[0] for i in batch)
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
