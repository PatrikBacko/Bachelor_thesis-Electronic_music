#!/usr/bin/env python3

import math
import os

import numpy as np
import librosa as lb
import soundfile as sf
import pyaudio

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse

def pad_or_trim(mfcc, length = 100):
    '''
    pads or trims mfcc to given length, default is 100 ! (cca 1 second with 256 hop length and 512 n_fft and 44100 sr) !

    params:
        mfcc - mfcc to pad or trim
        length - length to pad or trim to (default is 100)

    returns:
        mfcc - padded or trimmed mfcc
    '''

    if mfcc.shape[1] > length:
        return mfcc[:, :length]
    else:
        last_column = mfcc[:, -1:]
        padding = np.repeat(last_column, length - mfcc.shape[1], axis=1)
        return np.concatenate((mfcc, padding), axis=1)

def prepare_data(source_dir, log_file, batch_size = 32, length = 100):
    '''
    gets path to directory with samples, and returns a dataloader with padded or trimmed mfccs of the samples
    ! padded or trimmed to cca 1 second with 256 hop length and 512 n_fft and 44100 sr !

    params:
        source_dir - path to directory with samples
        length - length to pad or trim to (default is 100)

    returns:
        train_loader - dataloader with padded or trimmed mfccs of the samples
    '''
    print(f'Preparing data from {source_dir}...', file=log_file)

    paths_to_samples = [os.path.join(source_dir, path) for path in os.listdir(source_dir)]

    mfccs = []

    for path in paths_to_samples:
            array, sr = lb.load(path)

            mfcc = lb.feature.mfcc(y=array, sr=sr, n_mfcc=512, n_fft=512, hop_length=256, lifter=0, dct_type=3, n_mels = 256)
            mfcc_pad_or_trim = pad_or_trim(mfcc, 100)

            mfccs.append(mfcc_pad_or_trim)

    mfccs_tensor = torch.tensor(np.array(mfccs)).view(-1, 1, 256, 100)
    train_loader = torch.utils.data.DataLoader(mfccs_tensor, batch_size=batch_size, shuffle=True)

    print('Data prepared.', file=log_file)

    return train_loader
