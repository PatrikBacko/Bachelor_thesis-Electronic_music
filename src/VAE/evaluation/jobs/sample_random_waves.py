import torch

from src.VAE.utils.data import tensor_to_wave, save_wave



def sample_random_wave(model, config, mean, scale, seed = None, sr = 44_100):
    """
    Samples a random wave from the latent space of the model and reconstructs it

    params:
        model (torch.nn.Module) - model to sample from
        config (utils.Config) - config of the model
        seed (int) - seed for the random generator

    returns:
        np.array - reconstructed wave
    """
    
    if seed is not None:
        torch.manual_seed(seed)

    normal = torch.distributions.Normal(mean, scale)


    z = normal.sample((1, config.latent_dim))

    reconstructed_x = model.decode(z)

    return tensor_to_wave(reconstructed_x, 44_100, config)




def sample_and_save_random_waves(model, config, output_path, n_samples=5, means = [0], scales = [1, 2, 5, 10], seed = None, sr = 44_100):
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
                reconstructed_wave = sample_random_wave(model, config, mean, scale, sr)

                save_wave(reconstructed_wave, sr, output_path / f'sample_scale={scale}_{i}.wav')