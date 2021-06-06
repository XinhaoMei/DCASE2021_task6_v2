#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


"""
Remove wav files that are less than 5 seconds in audiocaps dataset
"""

import glob
import librosa
import subprocess
import pickle
from tqdm import tqdm

wav_files = glob.glob("train/*.wav")

wav_names = []

for wav_file in tqdm(wav_files, total=len(wav_files)):
    audio = librosa.load(wav_file, sr=32000)
    time = len(audio) / 32000
    if time < 5.0:
        wav_names.append(wav_file)
print(f'Total {len(wav_names)} files is less than 5 seconds')
for name in wav_names:
    subprocess.call('rm "%s" ' % name, shell=True)
with open('wav_names.p', 'wb') as f:
    pickle.dump(wav_names, f)
