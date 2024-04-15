import numpy as np
import os
from pathlib import Path


from src.VAE.utils.data import load_wave, save_wave, tensor_to_wave, prepare_wave_for_model, trim_wave


def load_random_wave(data_path, sample_group = None, seed = None, test_samples = False):
        '''
        Loads a random wave from the data_path with a given sample_group, if not given, it chooses a random sample_group

        params:
            data_path (Path) - path to the data
            sample_group (str) - sample group to choose from
            seed (int) - seed for the random generator
            test_samples (bool) - whether to load test samples or not

        returns:
            np.array - loaded wave
        '''
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
    x = prepare_wave_for_model(wave, sr, config)

    reconstructed_x, _, _ = model(x)

    return tensor_to_wave(reconstructed_x, sr, config)


def reconstruct_random_train_samples(model, config, output_path, n_samples, data_path, original_wave_trim_length, seed = None):
    '''
    Selects n_samples random samples from the data_path and reconstructs them using the model for each sample_group the model was trained on. 
    The reconstructed samples and original ones (trimmed to a given length) are saved to the output_path

    params:
        model (torch.nn.Module) - model to use for reconstruction
        config (utils.Config) - config of the model
        output_path (Path) - path to the directory to save the reconstructed samples
        n_samples (int) - number of samples to reconstruct
        data_path (Path) - path to the data
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

            save_wave(trimmed_wave, sr, output_path / f'{wave_name}_original.wav')
            save_wave(reconstructed_wave, sr, output_path / f'{wave_name}_reconstructed.wav')


def reconstruct_test_samples(model, config, output_path, data_path, original_wave_trim_length):
    '''
    reconstructs all the test samples from the data_path using the model and saves the reconstructed samples and original ones (trimmed to a given length) to the output_path

    params:
        model (torch.nn.Module) - model to use for reconstruction
        config (utils.Config) - config of the model
        output_path (Path) - path to the directory to save the reconstructed samples
        data_path (Path) - path to the data
        original_wave_trim_length (float) - length to trim the original wave to

    returns:
        None
    '''
    for group in config.sample_group:
        path_to_group = data_path / group / f'{group}_test_samples'

        for sample_name in os.listdir(path_to_group):
            sample_path = path_to_group / sample_name

            wave, sr = load_wave(sample_path)
            reconstructed_wave = reconstruct_wave(wave, sr, model, config)

            trimmed_wave = trim_wave(wave, sr, original_wave_trim_length)

            sample_name = sample_name.replace('.wav', '')
            save_wave(trimmed_wave, sr, os.path.join(output_path, f'{sample_name}_original.wav'))
            save_wave(reconstructed_wave, sr, os.path.join(output_path, f'{sample_name}_reconstructed.wav'))


def reconstruct_samples(model, config, output_path, data_path = 'data/drums-one_shots', n_samples = 10, seed = None):
    '''
    Takes test samples, and n_samples of random samples from the data_path for each sample_group the model was trained on and reconstructs them using the model.
    Then saves the reconstructed samples and original ones (trimmed to a given length) to the output_path

    params:
        model (torch.nn.Module) - model to use for reconstruction
        config (utils.Config) - config of the model
        output_path (str | Path) - path to the directory to save the reconstructed samples
        data_path (str | Path) - path to the data
        n_samples (int) - number of samples to reconstruct
        seed (int) - seed for the random generator

    returns:
        None
    '''

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    data_path = Path(data_path)

    original_wave_trim_length = 1.5

    output_path_test = output_path / 'reconstructed_test'
    output_path_test.mkdir(exist_ok=True)
    reconstruct_test_samples(model, config, output_path_test, data_path, original_wave_trim_length)

    output_path_random = output_path / 'reconstructed_train_random'
    output_path_random.mkdir(exist_ok=True)
    reconstruct_random_train_samples(model, config, output_path_random, n_samples, data_path, original_wave_trim_length, seed)
