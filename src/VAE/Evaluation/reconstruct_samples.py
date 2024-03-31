import torch
import numpy as np
import os
from pathlib import Path


from src.VAE.utils.data import load_wave, save_wave, convert_to_mfcc, get_wave_from_mfcc, pad_or_trim, to_numpy, get_inverse_mfcc_kwargs, trim_wave


def load_random_wave(data_path, sample_group = None, seed = None):
        '''
        Loads a random wave from the data_path with a given sample_group, if not given, it chooses a random sample_group

        params:
            data_path (str | Path) - path to the data
            sample_group (str) - sample group to choose from
            seed (int) - seed for the random generator

        returns:
            np.array - loaded wave
        '''
        rng = np.random.default_rng(seed)

        data_path = Path(data_path)

        if sample_group is None:
            sample_group = rng.choice([group for group in os.listdir(data_path) if os.path.isdir(data_path / group)])
        

        path_sample_group = data_path / sample_group / f'{sample_group}_samples'
        sample_name = rng.choice(os.listdir(path_sample_group))
        path_to_wave = path_sample_group / sample_name


        wave, sr = load_wave(path_to_wave)
    
        return wave, sr, sample_name


def reconstruct_wave(wave, sr, model, config):
    '''
    Reconstructs a given wave using the model

    params:
        wave (np.array) - wave to reconstruct
        sr (int) - sample rate of the wave
        model (torch.nn.Module) - model to use for reconstruction
        config (utils.Config) - config of the model

    returns:
        np.array - reconstructed wave
    '''
    mfcc = convert_to_mfcc(wave, sr)
    mfcc = pad_or_trim(mfcc, config.pad_or_trim_length)

    x = torch.tensor(mfcc).view(-1, 1, config.mfcc_kwargs['n_mels'], config.pad_or_trim_length)

    reconstructed_x, _, _ = model(x)
    reconstructed_mfcc = to_numpy(reconstructed_x).reshape(config.mfcc_kwargs['n_mels'], config.pad_or_trim_length)

    inverse_mfcc_kwargs = get_inverse_mfcc_kwargs(config.mfcc_kwargs)

    reconstructed_wave = get_wave_from_mfcc(reconstructed_mfcc, sr, inverse_mfcc_kwargs)

    return reconstructed_wave


def reconstruct_random_samples(model, config, output_path, n_samples, data_path, original_wave_trim_length, seed = None):
    '''
    Selects n_samples random samples from the data_path and reconstructs them using the model for each sample_group the model was trained on. 
    The reconstructed samples and original ones (trimmed to a given length) are saved to the output_path

    params:
        model (torch.nn.Module) - model to use for reconstruction
        config (utils.Config) - config of the model
        output_path (str | Path) - path to the directory to save the reconstructed samples
        n_samples (int) - number of samples to reconstruct
        data_path (str | Path) - path to the data
        original_wave_trim_length (float) - length to trim the original wave to
        seed (int) - seed for the random generator

    returns:
        None
    '''
    for group in config.sample_group:
        for _ in range(n_samples):
            wave, sr, wave_name = load_random_wave(data_path, group, seed=seed)
            reconstructed_wave = reconstruct_wave(wave, sr, model, config)

            trimmed_wave = trim_wave(wave, sr, original_wave_trim_length)
            wave_name = wave_name.replace('.wav', '')

            #TODO: not sure about the names of the waves
            save_wave(trimmed_wave, sr, os.path.join(output_path, f'{wave_name}_original.wav'))
            save_wave(reconstructed_wave, sr, os.path.join(output_path, f'{wave_name}_reconstructed.wav'))


def reconstruct_test_samples(model, config, output_path, data_path, original_wave_trim_length):
    '''
    reconstructs all the test samples from the data_path using the model and saves the reconstructed samples and original ones (trimmed to a given length) to the output_path

    params:
        model (torch.nn.Module) - model to use for reconstruction
        config (utils.Config) - config of the model
        output_path (str | Path) - path to the directory to save the reconstructed samples
        data_path (str | Path) - path to the data
        original_wave_trim_length (float) - length to trim the original wave to

    returns:
        None
    '''
    data_path = Path(data_path)
    for group in config.sample_group:
        path_to_group = data_path / group / f'{group}_test_samples'

        for sample_name in os.listdir(path_to_group):
            sample_path = path_to_group / sample_name

            wave, sr = load_wave(sample_path)
            reconstructed_wave = reconstruct_wave(wave, sr, model, config)

            trimmed_wave = trim_wave(wave, sr, original_wave_trim_length)


            #TODO: not sure about the names of the waves
            save_wave(trimmed_wave, sr, os.path.join(output_path, f'{sample_name}'))
            save_wave(reconstructed_wave, sr, os.path.join(output_path, f'reconstructed_{sample_name}'))


def reconstruct_samples(model, config, output_path, data_path = 'data/drums-one_shots', n_samples = 10, seed = None):
    output_path = Path(output_path)

    original_wave_trim_length = 1.5

    reconstruct_random_samples(model, config, output_path / 'random', n_samples, data_path, original_wave_trim_length, seed)
    reconstruct_test_samples(model, config, output_path / 'test', data_path, original_wave_trim_length)
