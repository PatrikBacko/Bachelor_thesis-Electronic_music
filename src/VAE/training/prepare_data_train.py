import torch
import numpy as np
import os
import gc

from src.VAE.utils.data import load_wave, get_spectogram
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
    padded_spectograms = [pad_or_trim(spectogram, length, conversion_config) for spectogram in 
                          [get_spectogram(wave, sr, conversion_config) for wave, sr in 
                           [load_wave(path) for path in paths_to_samples]]]

    if scaler:
        raveled_spectograms = np.array([spectogram.ravel() for spectogram in padded_spectograms])
        scaler = scaler.fit(raveled_spectograms)

        transformed_spectograms = scaler.transform(raveled_spectograms)
        transformed_spectograms = transformed_spectograms.reshape(-1, conversion_config['channels'], conversion_config['height'], length)
        del raveled_spectograms
    else:
        transformed_spectograms = np.array(padded_spectograms).reshape(-1, conversion_config['channels'], conversion_config['height'], length)

    spectogram_tensor = torch.tensor(transformed_spectograms)
    del padded_spectograms
    del transformed_spectograms
    gc.collect()
        
    train_loader = torch.utils.data.DataLoader(spectogram_tensor, batch_size=batch_size, shuffle=True)
    
    return train_loader
