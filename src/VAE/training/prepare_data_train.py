import torch
import numpy as np
import os
import gc

from src.VAE.utils.data import load_wave, get_spectrogram
from src.VAE.utils.conversion import pad_or_trim


def _get_paths_to_samples(data_dir, sample_groups_list):
    '''
    Gets list of sample groups and data directory, then returns list of paths to samples
    
    params:
        data_dir - path to directory with samples
        sample_groups_list - list of sample groups

    returns:
        paths_to_samples - list of paths to samples
    '''
    paths_to_samples = []
    for sample_group in sample_groups_list:
        sample_group_path = os.path.join(data_dir, sample_group, f'{sample_group}_samples')
        paths_to_samples.extend([os.path.join(sample_group_path, sample)for sample in os.listdir(sample_group_path)])

    return paths_to_samples


def prepare_train_loader(data_dir, sample_groups_list, length, batch_size, conversion_config, scaler = None):
    '''prepares the data for training

    params:
        source_dir - path to directory with samples
        length - length to pad or trim to
        batch_size - batch size for the dataloader

    returns:
        train_loader - pytorch dataloader with padded or trimmed specgtograms of the samples
    '''

    paths_to_samples = _get_paths_to_samples(data_dir, sample_groups_list)
    padded_spectrograms = [pad_or_trim(spectrogram, length, conversion_config) for spectrogram in 
                          [get_spectrogram(wave, sr, conversion_config) for wave, sr in 
                           [load_wave(path) for path in paths_to_samples]]]

    if scaler:
        raveled_spectrograms = np.array([spectrogram.ravel() for spectrogram in padded_spectrograms])
        scaler = scaler.fit(raveled_spectrograms)

        transformed_spectrograms = scaler.transform(raveled_spectrograms)
        transformed_spectrograms = transformed_spectrograms.reshape(-1, conversion_config['channels'], conversion_config['height'], length)
        del raveled_spectrograms
    else:
        transformed_spectrograms = np.array(padded_spectrograms).reshape(-1, conversion_config['channels'], conversion_config['height'], length)

    del padded_spectrograms
    gc.collect()

    spectrogram_tensor = torch.tensor(transformed_spectrograms)
    del transformed_spectrograms
    gc.collect()
        
    train_loader = torch.utils.data.DataLoader(spectrogram_tensor, batch_size=batch_size, shuffle=True)
    
    return train_loader
