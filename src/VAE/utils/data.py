#!/usr/bin/env python3
import os
import numpy as np
import librosa as lb
import soundfile as sf
import scipy.io.wavfile as wav

from pathlib import Path

import torch

from src.VAE.utils.scaler import load_scaler
from src.VAE.exceptions.InvalidSamplingException import InvalidInverseConversionException

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
    'htk': False
    }

def to_numpy(tensor):
    '''
    Converts a tensor to numpy array

    params:
        tensor - tensor to convert

    returns:
        np.ndarray - converted tensor
    '''
    return tensor.detach().cpu().numpy()


def get_inverse_mfcc_kwargs(mfcc_kwargs = MFCC_KWARGS):
    '''
    Gets a list of mfccs and returns the inverse mfcc kwargs

    params:
        mfccs (list) - list of mfccs
        inverse_mfcc_kwargs (dict) - kwargs for the inverse mfcc

    returns:
        dict - inverse mfcc kwargs
    '''

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


def trim_wave(wave, sr, length):
    '''
    Trims wave to given length in seconds, if the wave is shorter than the given length, it returns the original wave

    params:
        wave - wave to trim
        sr - sample rate of the wave
        length - length to trim to

    returns:
        np.array - trimmed wave
    '''
    duration = wave.size / sr
    if duration > length:
        return wave[:int(sr * length)]
    else:
        return wave
    

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
    

def load_wave(path_to_sample):
    '''
    Gets a path to a sample and returns tupble, loaded wave with its sample rate

    params:
        paths_to_sample - path to sample
    
    returns:
        np.array - loaded wave
        int - sample rate of the wave
    '''
    wave, sr = lb.load(path_to_sample)

    if sr != 44100:
        wave = lb.resample(wave, orig_sr=sr, target_sr=44100)
        sr = 44100

    return wave, sr

def load_random_wave(data_path, sample_group = None, seed = None, test_samples = False):
        '''
        Loads a random wave from the data_path with a given sample_group, if not given, it chooses a random sample_group

        params:
            data_path (str | Path) - path to the data
            sample_group (str) - sample group to choose from
            seed (int) - seed for the random generator
            test_samples (bool) - whether to load test samples or not

        returns:
            np.array - loaded wave
        '''
        
        data_path = Path(data_path)

        rng = np.random.default_rng(seed)

        if sample_group is None:
            sample_group = rng.choice([group for group in os.listdir(data_path) if os.path.isdir(data_path / group)])
        
        path_sample_group = data_path / sample_group / f'{sample_group}_samples'

        if test_samples:
            path_sample_group = data_path / sample_group / f'{sample_group}_test_samples'

        sample_name = rng.choice(os.listdir(path_sample_group))
        path_to_wave = path_sample_group / sample_name


        wave, sr = load_wave(path_to_wave)
    
        return wave, sr, sample_name


def save_wave(wave, sr, path_to_save):
    '''
    Saves a wave to a given path

    params:
        wave (np.array) - wave to save
        sr (int) - sample rate of the wave
        path_to_save (str) - path to save the wave
    '''
    # sf.write(path_to_save, wave, sr, subtype='PCM_24')
    wav.write(path_to_save, sr, wave)
    

def convert_to_mfcc(wave, sr, mfcc_kwargs = MFCC_KWARGS):
    '''
    Gets a wave and its sample rate, then returns its mfcc

    params:
        wave (np.array) - wave to convert to mfcc
        sr (int) - sample rate of the wave

    returns:
        np.ndarray - mfcc of the wave
    '''

    return lb.feature.mfcc(y=wave, sr=sr, **mfcc_kwargs)

def get_wave_from_mfcc(mfcc, sr = 44100, inverse_mfcc_kwargs = get_inverse_mfcc_kwargs()):
    '''
    Gets a mfcc and returns its inverse

    params:
        mfcc (np.ndarray) - mfcc to get inverse of

    returns:
        np.ndarray - inverse of the mfcc

    raises:
        InvalidSamplingException - if there is an error with mfcc inverse conversion
    '''

    try:
        return lb.feature.inverse.mfcc_to_audio(mfcc = mfcc, sr = sr, **inverse_mfcc_kwargs)

    except:
        raise InvalidInverseConversionException('Error with mfcc conversion')



def prepare_wave_for_model(wave, sr, config):
    '''
    prepares a wave for the model (mfcc conversion, padding or trimming, reshaping to torch tensor)

    params:
        wave (np.array) - wave to prepare
        sr (int) - sample rate of the wave
        config (utils.Config) - config of the model

    returns:
        torch.tensor - prepared wave
    '''

    mfcc = convert_to_mfcc(wave, sr, mfcc_kwargs=config.mfcc_kwargs)
    mfcc = pad_or_trim(mfcc, config.pad_or_trim_length)

    if config.scaler is not None:
        mfcc = load_scaler(config.scaler).transform(mfcc.reshape(1, -1)).reshape(config.mfcc_kwargs['n_mels'], config.pad_or_trim_length)

    tensor = torch.tensor(mfcc).view(-1, 1, config.mfcc_kwargs['n_mels'], config.pad_or_trim_length)

    return tensor


def tensor_to_mfcc(tensor, config):
    '''
    Converts a tensor to mfcc

    params:
        tensor (torch.tensor) - tensor to convert
        config (utils.Config) - config of the model

    returns:
        np.ndarray - converted tensor
    '''
    mfcc = to_numpy(tensor).reshape(config.mfcc_kwargs['n_mels'], config.pad_or_trim_length)

    if config.scaler is not None:
        mfcc = load_scaler(config.scaler).inverse_transform(mfcc.reshape(1, -1)).reshape(config.mfcc_kwargs['n_mels'], config.pad_or_trim_length)

    return mfcc

def tensor_to_wave(tensor, sr, config):
    '''
    Converts a tensor to wave

    params:
        tensor (torch.tensor) - tensor to convert
        sr (int) - sample rate of the wave
        config (utils.Config) - config of the model

    returns:
        np.array - converted tensor to wave

    raises:
        InvalidSamplingException - if there is an error with spectogram inverse conversion
    '''

    mfcc = tensor_to_mfcc(tensor, config)

    return get_wave_from_mfcc(mfcc, sr, get_inverse_mfcc_kwargs(config.mfcc_kwargs))
