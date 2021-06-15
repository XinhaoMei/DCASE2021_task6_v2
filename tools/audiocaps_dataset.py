#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


from loguru import logger
from pathlib import Path
from itertools import chain
from re import sub
from tools.file_io import load_csv_file, write_pickle_file


def create_dataset():

    inner_logger = logger

    inner_logger.info('Dataset processing for AudioCaps.')
    inner_logger.info('Loading csv files and process each caption.')

    train_csv = load_csv_file('audiocaps/csv_files/new_train.csv')
    val_csv = load_csv_file('audiocaps/csv_files/new_val.csv')
    test_csv = load_csv_file('audiocaps/csv_files/new_test.csv')

    all_captions = []
    for csv_item in chain(train_csv, val_csv, test_csv):
        caption = _sentence_process(csv_item['caption'], add_specials=True)
        csv_item['caption'] = caption
        all_captions.append(caption)
    inner_logger.info('Done.')

    inner_logger.info('Creating vocabulary and counting words frequency')
    words_list, words_freq = _create_vocabulary(all_captions)
    pickles_path = Path('audiocaps/pickles')
    pickles_path.mkdir(parents=True, exist_ok=True)
    write_pickle_file(words_list, str(pickles_path.joinpath('words_list.p')))
    write_pickle_file(words_freq, str(pickles_path.joinpath('words_freq.p')))
    inner_logger.info(f'Done. Total {len(words_list)} words in the vocabulary.')


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
