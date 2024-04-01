import torch
import numpy as np
import os

from src.VAE.utils.data import load_wave, convert_to_mfcc, pad_or_trim

def get_paths_to_samples(data_dir, sample_groups_list):
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

def return_data_loader(mfccs_list, batch_size):
    '''
    gets list of mfccs, and returns a torch dataloader with them

    params:
        mfccs_list - list of mfccs
        log_file - file to log the process
        batch_size - batch size for the dataloader

    returns:
        train_loader - dataloader with mfccs
    '''

    mfccs_tensor = torch.tensor(np.array(mfccs_list)).view(-1, 1, mfccs_list[0].shape[0], mfccs_list[0].shape[1])
    train_loader = torch.utils.data.DataLoader(mfccs_tensor, batch_size=batch_size, shuffle=True)

    return train_loader


def prepare_train_loader(data_dir, sample_groups_list, length, batch_size, scaler = None):
    '''prepares the data for training

    params:
        source_dir - path to directory with samples
        log_file - file to log the process
        length - length to pad or trim to
        batch_size - batch size for the dataloader

    returns:
        train_loader - dataloader with padded or trimmed mfccs of the samples
    '''

    paths_to_samples = get_paths_to_samples(data_dir, sample_groups_list)
    waves = [load_wave(path) for path in paths_to_samples]
    mfccs = [convert_to_mfcc(wave, sr) for wave, sr in waves]
    padded_mfccs = [pad_or_trim(mfccs, length) for mfccs in mfccs]

    if scaler:
        scaler = scaler.fit(padded_mfccs)
        transformed_mfccs = [scaler.transform(mfccs) for mfccs in padded_mfccs]
    else:
        transformed_mfccs = padded_mfccs

    train_loader = return_data_loader(transformed_mfccs, batch_size)

    return train_loader
