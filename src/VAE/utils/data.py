#!/usr/bin/env python3
import numpy as np
import librosa as lb
import soundfile as sf

import torch

from src.VAE.utils.scaler import load_scaler

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
    return lb.load(path_to_sample)


def save_wave(wave, sr, path_to_save):
    '''
    Saves a wave to a given path

    params:
        wave (np.array) - wave to save
        sr (int) - sample rate of the wave
        path_to_save (str) - path to save the wave
    '''
    sf.write(path_to_save, wave, sr, subtype='PCM_24')
    

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
    '''

    return lb.feature.inverse.mfcc_to_audio(mfcc = mfcc, sr = sr, **inverse_mfcc_kwargs)


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
        mfcc = load_scaler(config.scaler).transform(mfcc)

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
        mfcc = load_scaler(config.scaler).inverse_transform(mfcc)

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
    '''
    mfcc = tensor_to_mfcc(tensor, config)
    return get_wave_from_mfcc(mfcc, sr, get_inverse_mfcc_kwargs(config.mfcc_kwargs))

