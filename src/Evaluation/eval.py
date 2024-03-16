import os
from pathlib import Path

import numpy as np
import librosa as lb
# import soundfile as sf
# import pyaudio
import sounddevice as sd

import torch
import torch.nn as nn
import torch.nn.functional as F

import sklearn
import sklearn.decomposition

import matplotlib.pyplot as plt
%matplotlib inline
# %matplotlib notebook

from models.VAE_1 import VAE_1
from utils.prepare_data import pad_or_trim

from utils.config import load_config



def load_wave(path):
    sample, sr = lb.load(path)
    return sample, sr

def load_random_wave(sample_type, path_to_samples = r"C:\Users\llama\Desktop\cuni\bakalarka\Bachelor_thesis-Electronic_music\data\drums-one_shots"):
    # load random wave
    path = os.path.join(path_to_samples, sample_type, f'{sample_type}_samples')

    sample_name = np.random.choice(os.listdir(path))
    sample, sr = load_wave(os.path.join(path, sample_name))
    return sample, sr

def load_all_waves(sample_types: list, path_to_samples = r"C:\Users\llama\Desktop\cuni\bakalarka\Bachelor_thesis-Electronic_music\data\drums-one_shots", return_sample_groups = False):
    waves = []
    for sample_type in sample_types:
        path = os.path.join(path_to_samples, sample_type, f'{sample_type}_samples')
        for sample_name in os.listdir(path):
            sample, sr = load_wave(os.path.join(path, sample_name))
            if return_sample_groups:
                waves.append((sample, sr, sample_type))
            else:
                waves.append((sample, sr))
    return waves

def convert_to_mfcc(wave, sr, mfcc_kwargs):
    mfcc = lb.feature.mfcc(y=wave, sr=sr, **mfcc_kwargs)
    return mfcc

def convert_to_wave(mfcc, sr, inverse_mfcc_kwargs):
    return lb.feature.inverse.mfcc_to_audio(mfcc=mfcc, sr=sr, **inverse_mfcc_kwargs)


def sample_from_latent_space(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

def fuse_sampled_samples(samples, weights = None):
    if weights is None:
        weights = np.ones(len(samples)) / len(samples)
    return np.average(samples, axis=0, weights=weights).astype(np.float32)


def inverse_mfcc_kwargs(mfcc_kwargs):
    return {
        'n_mels': mfcc_kwargs['n_mels'],
        'dct_type': mfcc_kwargs['dct_type'],
        'norm': mfcc_kwargs['norm'],
        'lifter': mfcc_kwargs['lifter'],
        'n_fft': mfcc_kwargs['n_fft'],
        'hop_length': mfcc_kwargs['hop_length'],
        'win_length': mfcc_kwargs['win_length'],
        'window':  mfcc_kwargs['window'],
        'center': mfcc_kwargs['center'],
        'pad_mode': mfcc_kwargs['pad_mode'],
        'power': mfcc_kwargs['power'],


        'ref': 1.0,
        'n_iter': 32,
        'length': None,
        'dtype': np.float32
    }




def load_model(model_path, device = 'cpu'):
    # sys.path.append(r'C:\Users\llama\Desktop\cuni\bakalarka\Bachelor_thesis-Electronic_music')
    # sys.path.append(r'/storage/praha1/home/buciak/bachelors_thesis/bachelor_thesis-Electronic_music')

    model_dir_path = Path(r'C:\Users\llama\Desktop\cuni\bakalarka\Bachelor_thesis-Electronic_music\trained_models\model_kick,clap,hat,snare_latent-dims=32_noise=multiplicative')
    # model_dir_path = Path(r'/storage/praha1/home/buciak/bachelors_thesis/test_results/test_no_noise')

    model_name = os.path.basename(model_dir_path)
    config = load_config(model_dir_path / f'{model_name}_config.json')

    model = VAE_1(config.latent_dim)
    model.load_state_dict(torch.load(model_dir_path / f'model_{model_name}.pkl', map_location=torch.device('cpu')))

    return model
