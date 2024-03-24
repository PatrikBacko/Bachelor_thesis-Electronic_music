import torch
import numpy as np
import os
from pathlib import Path


from utils.data import load_wave, save_wave, convert_to_mfcc, get_wave_from_mfcc, pad_or_trim, to_numpy, get_inverse_mfcc_kwargs, trim_wave


def load_random_wave(data_path, sample_group = None, seed = None):
        rng = np.random.default_rng(seed)

        data_path = Path(data_path)

        if sample_group is None:
            sample_group = rng.choice(os.listdir(data_path))
        
        path_sample_group = data_path / sample_group
        sample_name = rng.choice(os.listdir(path_sample_group))
        path_to_wave = path_sample_group / sample_name

        wave, sr = load_wave(path_to_wave)
    
        return wave, sr, sample_name


def reconstruct_wave(wave, sr, model, config):
    mfcc = convert_to_mfcc(wave, sr)
    mfcc = pad_or_trim(mfcc, config.pad_or_trim_length)

    x = torch.tensor(mfcc).view(-1, 1, 256, config.pad_or_trim_length)

    reconstructed_x, mu, logvar = model(x)
    reconstructed_mfcc = to_numpy(reconstructed_x).reshape(256, config.pad_or_trim_length)

    inverse_mfcc_kwargs = get_inverse_mfcc_kwargs(config.mfcc_kwargs)

    reconstructed_wave = get_wave_from_mfcc(reconstructed_mfcc, sr, inverse_mfcc_kwargs)

    return reconstructed_wave


def reconstruct_random_samples(model, config, output_path, n_samples, data_path, original_wave_trim_length, seed = None):
    for group in config.sample_groups:
        for _ in range(n_samples):
            wave, sr, wave_name = load_random_wave(data_path, group, seed)
            reconstructed_wave = reconstruct_wave(wave, model, config)

            trimmed_wave = trim_wave(wave, sr, original_wave_trim_length)

            #TODO: not sure about the names of the waves
            save_wave(trimmed_wave, sr, os.path.join(output_path, f'{wave_name}'))
            save_wave(reconstructed_wave, sr, os.path.join(output_path, f'reconstructed_{wave_name}'))


def reconstruct_test_samples(model, config, output_path, data_path, original_wave_trim_length):
    data_path = Path(data_path)
    for group in config.sample_groups:
        path_to_group = data_path / group / f'{group}_test_samples'

        for sample_name in os.listdir(path_to_group):
            sample_path = path_to_group / sample_name

            wave, sr = load_wave(sample_path)
            reconstructed_wave = reconstruct_wave(wave, model, config)

            trimmed_wave = trim_wave(wave, sr, original_wave_trim_length)


            #TODO: not sure about the names of the waves
            save_wave(trimmed_wave, sr, os.path.join(output_path, f'{sample_name}'))
            save_wave(reconstructed_wave, sr, os.path.join(output_path, f'reconstructed_{sample_name}'))


def reconstruct_samples(model, config, output_path, data_path = 'data/drums-one_shots', n_samples = 10, seed = None):
    output_path = Path(output_path)

    original_wave_trim_length = 1.5

    reconstruct_random_samples(model, config, output_path / 'random', n_samples, data_path, original_wave_trim_length. seed)
    reconstruct_test_samples(model, config, output_path / 'test', data_path, original_wave_trim_length)