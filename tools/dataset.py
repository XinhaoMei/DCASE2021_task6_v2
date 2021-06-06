#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import os
import argparse
import numpy as np
import librosa
from tqdm import tqdm
from loguru import logger
from pathlib import Path
from itertools import chain
from re import sub
from tools.file_io import load_csv_file, write_pickle_file


parser = argparse.ArgumentParser(description='Setting for dataset creation')

parser.add_argument('--sr', type=int, default=32000, help="Sampling rate for the audio.")
parser.add_argument('--n_fft', type=int, default=1024, help="Length of the FFT window.")
parser.add_argument('--hop_length', type=int, default=320, help="Number of samples between successive frames.")
parser.add_argument('--n_mels', type=int, default=64, help="Number of mel bins.")
parser.add_argument('--window', type=str, default='hann', help='Type of window.')


def create_dataset():

    inner_logger = logger

    inner_logger.info('Loading csv files and process each caption.')

    dev_csv = load_csv_file('data/csv_files/clotho_captions_development.csv')
    val_csv = load_csv_file('data/csv_files/clotho_captions_validation.csv')
    eval_csv = load_csv_file('data/csv_files/clotho_captions_evaluation.csv')

    caption_fields = ['caption_{}'.format(i) for i in range(1, 6)]

    for csv_item in chain(dev_csv, val_csv, eval_csv):
        ''' Process each caption'''
        captions = [_sentence_process(csv_item[caption_field], add_specials=True) for caption_field in caption_fields]

        [csv_item.update({caption_field: caption})
         for caption_field, caption in zip(caption_fields, captions)]
    inner_logger.info('Done!')

    # all captions in dev set
    dev_captions = [csv_entry.get(caption_field)
                        for csv_entry in dev_csv
                        for caption_field in caption_fields]

    inner_logger.info('Creating vocabulary and counting words frequency...')
    words_list, words_freq = _create_vocabulary(dev_captions)
    pickles_path = Path('data/pickles')
    pickles_path.mkdir(parents=True, exist_ok=True)
    write_pickle_file(words_list, str(pickles_path.joinpath('words_list.p')))
    write_pickle_file(words_freq, str(pickles_path.joinpath('words_freq.p')))
    inner_logger.info(f'Done. Total {len(words_list)} words in the vocabulary.')

    for split_data in [(dev_csv, 'development'), (val_csv, 'validation'), (eval_csv, 'evaluation')]:

        split_csv = split_data[0]
        split_name = split_data[1]

        split_dir = Path('data/data_splits', split_name)
        split_dir.mkdir(parents=True, exist_ok=True)

        audio_dir = Path('data', split_name)

        inner_logger.info(f'Creating the {split_name} split.')
        _create_split_data(split_csv, split_dir, audio_dir, words_list)
        inner_logger.info('Done')

        audio_number = len(os.listdir(str(audio_dir)))
        data_number = len(os.listdir(str(split_dir)))

        inner_logger.info('{} audio files in {}.'.format(audio_number, split_name))
        inner_logger.info('{} data files in {}'.format(data_number, split_name))
        inner_logger.info('{} data files per audio.'.format(data_number / audio_number))

    inner_logger.info('Dataset created')


def _create_vocabulary(captions):
    words_list = []
    vocabulary = []
    for caption in captions:
        caption_words = caption.strip().split()
        vocabulary.extend(caption_words)
    words_list = list(set(vocabulary))
    words_list.sort(key=vocabulary.index)
    words_freq = [vocabulary.count(word) for word in words_list]

    return words_list, words_freq


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


def _create_split_data(split_csv, split_dir, audio_dir, words_list):

    args = parser.parse_args()

    sr = args.sr
    n_fft = args.n_fft
    hop_length = args.hop_length
    n_mels = args.n_mels
    window = args.window

    caption_fields = ['caption_{}'.format(i) for i in range(1, 6)]
    file_name_template = 'clotho_file_{audio_file_name}_{caption_index}.npy'

    for csv_entry in tqdm(split_csv, total=len(split_csv)):

        audio_file_name = csv_entry['file_name']

        audio, _ = librosa.load(audio_dir.joinpath(audio_file_name), sr=sr)

        feature = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                                 n_mels=n_mels, window=window)
        feature = librosa.power_to_db(feature).T

        for caption_index, caption_field in enumerate(caption_fields):

            caption = csv_entry[caption_field]

            caption_words = caption.strip().split()

            words_indexs = [words_list.index(word) for word in caption_words]

            np_rec_array = np.rec.array(np.array(
                (audio_file_name, audio, feature, caption, caption_index, np.array(words_indexs)),
                dtype=[
                    ('file_name', 'U{}'.format(len(audio_file_name))),
                    ('audio_data', np.dtype(object)),
                    ('feature', np.dtype(object)),
                    ('caption', 'U{}'.format(len(caption))),
                    ('caption_index', 'i4'),
                    ('words_indexs', np.dtype(object))
                ]
            ))

            # save the numpy object
            file_name = str(split_dir.joinpath(file_name_template.format(
                                audio_file_name=audio_file_name, caption_index=caption_index)))
            np.save(file_name, np_rec_array)


if __name__ == '__main__':
    create_dataset()
