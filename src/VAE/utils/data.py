#!/usr/bin/env python3
import numpy as np
import librosa as lb
import torch


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
        return wave[:np.round(sr * length)]
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
    lb.output.write_wav(path_to_save, wave, sr)
    

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
