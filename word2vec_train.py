#!/usr/bin/env python
# coding: utf-8

from gensim.models.word2vec import Word2Vec
from re import sub
from pathlib import Path
import csv
from itertools import chain


def load_csv(file_name):
    with open(file_name, 'r') as f:
        csv_reader = csv.DictReader(f)
        csv_obj = [csv_line for csv_line in csv_reader]
    return csv_obj


dev_csv_file = 'data/csv_files/clotho_captions_development.csv'
val_csv_file = 'data/csv_files/clotho_captions_validation.csv'
eval_csv_file = 'data/csv_files/clotho_captions_evaluation.csv'
ac_train_csv_file = 'audiocaps/csv_files/new_train.csv'
ac_val_csv_file = 'audiocaps/csv_files/new_val.csv'
ac_test_csv_file = 'audiocaps/csv_files/new_test.csv'

dev_csv = load_csv(dev_csv_file)
val_csv = load_csv(val_csv_file)
eval_csv = load_csv(eval_csv_file)
ac_train_csv = load_csv(ac_train_csv_file)
ac_val_csv = load_csv(ac_val_csv_file)
ac_test_csv = load_csv(ac_test_csv_file)

print(f'Total {len(dev_csv) + len(ac_train_csv)} audios in development set')
print(f'Total {len(val_csv) + len(ac_val_csv)} audios in validation set')
print(f'Total {len(eval_csv) + len(ac_test_csv)} audios in evaluation set')

clotho_captions = []
field_caption = 'caption_{}'
for item in chain(dev_csv, val_csv, eval_csv):

    for cap_ind in range(1, 6):
        sentence = item[field_caption.format(cap_ind)].lower()
        # remove fogotten space before punctuation and double space
        sentence = sub(r'\s([,.!?;:"](?:\s|$))', r'\1', sentence).replace('  ', ' ')
        sentence = sub('[,.!?;:\"]', ' ', sentence).replace('  ', ' ')
        sentence = '<sos> {} <eos>'.format(sentence).strip().split()
        clotho_captions.append(sentence)

print(f'{len(clotho_captions)} captions in clotho dataset.')

ac_captions = []
for ac_item in chain(ac_train_csv, ac_val_csv, ac_test_csv):
    sentence = ac_item['caption'].lower()
    # remove fogotten space before punctuation and double space
    sentence = sub(r'\s([,.!?;:"](?:\s|$))', r'\1', sentence).replace('  ', ' ')
    sentence = sub('[,.!?;:\"]', ' ', sentence).replace('  ', ' ')
    sentence = '<sos> {} <eos>'.format(sentence).strip().split()
    ac_captions.append(sentence)

clotho_captions.extend(ac_captions)
print(f'{len(clotho_captions)} captions(sentences) in total to be trained.')

print('Start training the model')
# train the model
model = Word2Vec(clotho_captions, size=128, min_count=1, window=3, iter=1000)
print('Training finished.\n')
model.save('w2v.model')
