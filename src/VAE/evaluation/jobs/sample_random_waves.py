import torch

from src.VAE.utils.data import tensor_to_wave, save_wave
from src.VAE.exceptions.InvalidSamplingException import InvalidInverseConversionException



def sample_random_wave(model, config, mean, scale, seed = None, sr = 44_100):
    """
    Samples a random wave from the latent space of the model and reconstructs it

    params:
        model (torch.nn.Module) - model to sample from
        config (utils.Config) - config of the model
        seed (int) - seed for the random generator

    returns:
        np.array - reconstructed wave

    raises:
        InvalidSamplingException - if there is an error with sampling (inverse of the spectogram is not possible)
    """
    
    if seed is not None:
        torch.manual_seed(seed)

    normal = torch.distributions.Normal(mean, scale)


    z = normal.sample((1, config.latent_dim))

    reconstructed_x = model.decode(z)

    return tensor_to_wave(reconstructed_x, 44_100, config)


def sample_and_save_random_wave(model, config, output_path, mean, scale, seed = None, sr = 44_100, i=""):
    """
    Samples a random wave from the model and saves it to the output_path
    If there is an error with sampling, it prints the error and continues

    params:
        model (torch.nn.Module) - model to sample from
        config (utils.Config) - config of the model
        output_path (Path) - path to the directory to save the sample
        seed (int) - seed for the random generator

    returns:
        None
    """
    try:
        reconstructed_wave = sample_random_wave(model, config, mean, scale, seed, sr)
        save_wave(reconstructed_wave, sr, output_path / f'sample_{i}_mean={mean}_scale={scale}.wav')

    except InvalidInverseConversionException as e:
        print(f'Error with sampling a wave with mean={mean} and scale={scale}')
        print(f'\t{e}')


def sample_and_save_random_waves(model, config, output_path, n_samples=5, means = [0], scales = [1, 2, 3, 4, 5, 10], seed = None, sr = 44_100):
    """
    Samples random waves from the model and saves them to the output_path

    params:
        model (torch.nn.Module) - model to sample from
        config (utils.Config) - config of the model
        output_path (Path) - path to the directory to save the samples
        seed (int) - seed for the random generator

    returns:
        None
    """

    output_path.mkdir(exist_ok=True, parents=True)
    for mean in means:
        for scale in scales:
            for i in range(n_samples):
                sample_and_save_random_wave(model, config, output_path, mean, scale, seed, sr)