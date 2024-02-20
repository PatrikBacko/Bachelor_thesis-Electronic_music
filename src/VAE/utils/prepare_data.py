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

MFCC_KWARGS = {
    'n_mfcc': 512,
    'dct_type': 2,
    'norm': "ortho",
    'lifter': 0,

    #mel spectrogram kwargs
    'n_fft': 512,  
    'hop_length': 256,
    'win_length': 512,
    'window': "hann",
    'center': True,
    'pad_mode': "constant",
    'power': 2.0,

    #mel filterbank kwargs
    'n_mels': 256,
    'fmin': 0.0,
    'fmax': None,
    'htk': False,
    'dtype': np.float32
    }



def pad_or_trim(mfcc, length):
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
    
def get_paths_to_samples(data_dir, sample_groups_list):
    paths_to_samples = []
    for sample_group in sample_groups_list:
        sample_group_path = os.path.join(data_dir, sample_group, f'{sample_group}_samples')
        paths_to_samples.extend([os.path.join(sample_group_path, sample)for sample in os.listdir(sample_group_path)])

    return paths_to_samples
    
def convert_to_mfcc(paths_to_samples, length = 100):
    '''
    Gets list of paths to samples and returns list of converted samples to mfcc (and padded or trimmed to given length, default is 100)

    params:
        paths_to_samples - list of paths to samples
        length - length to pad or trim to (default is 100)

    returns:
        mfccs - list of mfccs of the samples
    '''
    mfccs = []

    for path in paths_to_samples:
            array, sr = lb.load(path)
            #TODO: MFCC **kwargs to be set in the config file, or at least in constant in this script
            mfcc = lb.feature.mfcc(y=array, sr=sr, **MFCC_KWARGS)
            mfcc_pad_or_trim = pad_or_trim(mfcc, length)

            mfccs.append(mfcc_pad_or_trim)
    return mfccs

def return_data_loader(mfccs_list, batch_size = 32):
    '''
    gets list of mfccs, and returns a torch dataloader with them

    params:
        mfccs_list - list of mfccs
        log_file - file to log the process
        batch_size - batch size for the dataloader (default is 32)

    returns:
        train_loader - dataloader with mfccs
    '''

    mfccs_tensor = torch.tensor(np.array(mfccs_list)).view(-1, 1, mfccs_list[0].shape[0], mfccs_list[0].shape[1])
    train_loader = torch.utils.data.DataLoader(mfccs_tensor, batch_size=batch_size, shuffle=True)

    return train_loader

def prepare_data(data_dir, sample_groups_list, length = 100, batch_size = 32):
    '''
    prepares the data for training

    params:
        source_dir - path to directory with samples
        log_file - file to log the process
        length - length to pad or trim to (default is 100)
        batch_size - batch size for the dataloader (default is 32)

    returns:
        train_loader - dataloader with padded or trimmed mfccs of the samples
    '''

    paths_to_samples = get_paths_to_samples(data_dir, sample_groups_list)
    mfccs = convert_to_mfcc(paths_to_samples, length)
    train_loader = return_data_loader(mfccs, batch_size)

    return train_loader
