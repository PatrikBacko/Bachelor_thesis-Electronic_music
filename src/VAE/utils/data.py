#!/usr/bin/env python3
import numpy as np
import librosa as lb
import soundfile as sf
from pathlib import Path
import os
import scipy.io.wavfile as wav

import torch

from src.VAE.utils.conversion import convert_wave_to_spectogram, convert_spectogram_to_wave, pad_or_trim
from src.VAE.utils.scaler import load_scaler
from src.VAE.exceptions.InvalidSamplingException import InvalidInverseConversionException




def to_numpy(tensor):
    '''
    Converts a tensor to numpy array

    params:
        tensor - tensor to convert

    returns:
        np.ndarray - converted tensor
    '''
    return tensor.detach().cpu().numpy()


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
        path_to_save (str | Path) - path to save the wave
    '''
    wav.write(str(path_to_save), sr, wave)


def get_wave_from_spectogram(spectogram, sr, conversion_config):
    '''
    Gets a spectogram and returns its inverse

    params:
        spectogram (np.ndarray) - spectogram to get inverse of

    returns:
        np.ndarray - wave, inverse of the spectogram

    raises:
        InvalidSamplingException - if there is an error with spectogram inverse conversion
    '''

    return convert_spectogram_to_wave(spectogram, sr, conversion_config)
    

def tensor_to_spectogram(tensor, config):
    '''
    Converts a tensor to spectogram

    params:
        tensor (torch.tensor) - tensor to convert
        config (utils.Config) - config of the model

    returns:
        np.ndarray - converted tensor to spectogram
    '''
    shape = tensor.shape[1:]
    if shape[0] == 1:
        shape = shape[1:]

    spectogram = to_numpy(tensor).reshape(shape)

    if config.scaler is not None:
        scaler = load_scaler(config.scaler)
        spectogram = scaler.inverse_transform(spectogram.reshape(1, -1)).reshape(shape)

    return spectogram



def get_spectogram(wave, sr, conversion_config):
    '''
    Gets a wave and its sample rate, then returns its spectogram

    params:
        wave (np.array) - wave to convert to spectogram
        sr (int) - sample rate of the wave
        conversion_config (dict) - spectogram conversion config

    returns:
        np.ndarray - spectogram of the wave with given config
    '''

    return convert_wave_to_spectogram(wave, sr, conversion_config)



def prepare_wave_for_model(wave, sr, config):
    '''
    prepares a wave for the model (spectogram conversion, padding or trimming, reshaping to torch tensor)

    params:
        wave (np.array) - wave to prepare
        sr (int) - sample rate of the wave
        config (utils.Config) - config of the model

    returns:
        torch.tensor - prepared wave
    '''

    spectogram = get_spectogram(wave, sr, conversion_config=config.conversion_config)
    spectogram = pad_or_trim(spectogram, config.pad_or_trim_length, config.conversion_config)

    if config.scaler is not None:
        scaler = load_scaler(config.scaler)
        spectogram = scaler.transform(spectogram.reshape(1, -1)).reshape(spectogram.shape)

    tensor = torch.tensor(spectogram).view(-1, config.conversion_config['channels'], config.conversion_config['height'], config.pad_or_trim_length)

    return tensor

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

    spectogram = tensor_to_spectogram(tensor, config)

    try:
        return get_wave_from_spectogram(spectogram, sr, config.conversion_config)
    except Exception as e:
        raise InvalidInverseConversionException(f'Error with inverse conversion: {e}')
