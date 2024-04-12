import torch
from pathlib import Path
import numpy as np

from src.VAE.evaluation.jobs.reconstruct_samples import load_random_wave
from src.VAE.utils.data import prepare_wave_for_model, tensor_to_wave, to_numpy, save_wave



def fuse_sampled_waves(samples, weights = None):
    """
    Fuses the given samples encoded into latent space by averaging them with given weights (default is uniform).

    params:
        samples (list of np.array) - samples to fuse
        weights (np.array) - weights for averaging

    returns:
        np.array - fused samples
    """
    if weights is None:
        weights = np.ones(len(samples)) / len(samples)
    return np.average(samples, axis=0, weights=weights).astype(np.float32)


def fuse_and_decode_wave(model, config, wave_1, sr_1, wave_2, sr_2):
    """
    Gets two waves, encodes them into latent space, fuses them and decodes the fused latent space into a wave.

    params:
        model (torch.nn.Module) - model to use for encoding and decoding
        config (utils.Config) - config of the model
        wave_1 (np.array) - wave to fuse
        sr_1 (int) - sample rate of the wave
        wave_2 (np.array) - wave to fuse
        sr_2 (int) - sample rate of the wave

    returns:
        np.array - fused wave
    """
    x_1 = prepare_wave_for_model(wave_1, sr_1, config)
    x_2 = prepare_wave_for_model(wave_2, sr_2, config)

    z_1, _ = model.encode(x_1)
    z_2, _ = model.encode(x_2)

    z_1 = to_numpy(z_1)
    z_2 = to_numpy(z_2)

    z = fuse_sampled_waves([z_1, z_2])

    z = torch.tensor(z)

    reconstructed_x = model.decode(z)

    return tensor_to_wave(reconstructed_x, sr_1, config)


def reconstruct_and_save_fused_wave(sample_type_1, sample_type_2, model, config, data_path, output_path, test_samples = False, seed = None):
    """
    Loads two random waves of the given sample types, fuses them and saves the fused wave.

    params:
        sample_type_1 (str) - first sample type to fuse
        sample_type_2 (str) - second sample type to fuse
        model (torch.nn.Module) - model to use for encoding and decoding
        config (utils.Config) - config of the model
        data_path (Path) - path to the data
        output_path (Path) - path to save the fused wave
        test_samples (bool) - whether to load test samples or not
        seed (int) - seed for the random generator

    returns:
        None
    """

    wave_1, sr_1, sample_name_1 = load_random_wave(data_path, sample_type_1, test_samples = test_samples, seed = seed)
    wave_2, sr_2, sample_name_2 = load_random_wave(data_path, sample_type_2, test_samples = test_samples, seed = seed)

    fused_wave = fuse_and_decode_wave(model, config, wave_1, sr_1, wave_2, sr_2)

    save_wave(fused_wave, sr_1, output_path / f'{sample_name_1}_{sample_name_2}_fused.wav')



def generate_convex_combinations(model, config, data_path, output_path, test_samples = False, seed = None):
    """
    Generates convex combinations of the given sample types and saves them.

    params:
        model (torch.nn.Module) - model to use for encoding and decoding
        config (utils.Config) - config of the model
        data_path (Path) - path to the data
        output_path (Path) - path to save the convex combinations
        test_samples (bool) - whether to load test samples or not
        seed (int) - seed for the random generator

    returns:
        None
    """
    data_path = Path(data_path)
    output_path = Path(output_path)

    output_path.mkdir(exist_ok=True, parents=True)

    for sample_type in config.sample_group:
        reconstruct_and_save_fused_wave(sample_type, sample_type, model, config, data_path, output_path)

    reconstruct_and_save_fused_wave('kick', 'clap', model, config, data_path, output_path, test_samples = test_samples, seed = seed)
    reconstruct_and_save_fused_wave('kick', 'crash', model, config, data_path, output_path, test_samples = test_samples, seed = seed)
    reconstruct_and_save_fused_wave('kick', 'tom', model, config, data_path, output_path, test_samples = test_samples, seed = seed)
    reconstruct_and_save_fused_wave('kick', 'snare', model, config, data_path, output_path, test_samples = test_samples, seed = seed)

    reconstruct_and_save_fused_wave('clap', 'crash', model, config, data_path, output_path, test_samples = test_samples, seed = seed)
    reconstruct_and_save_fused_wave('clap', 'tom', model, config, data_path, output_path, test_samples = test_samples, seed = seed)
    reconstruct_and_save_fused_wave('clap', 'snare', model, config, data_path, output_path, test_samples = test_samples, seed = seed)

    reconstruct_and_save_fused_wave('crash', 'tom', model, config, data_path, output_path, test_samples = test_samples, seed = seed)
    reconstruct_and_save_fused_wave('crash', 'snare', model, config, data_path, output_path, test_samples = test_samples, seed = seed)

    reconstruct_and_save_fused_wave('tom', 'snare', model, config, data_path, output_path, test_samples = test_samples, seed = seed)
