import numpy as np
import torch
from pathlib import Path


from src.VAE.evaluation.jobs.reconstruct_samples import load_random_wave
from src.VAE.evaluation.jobs.pca import get_fitted_pca
from src.VAE.utils.data import prepare_wave_for_model, tensor_to_wave, to_numpy, save_wave
from src.VAE.exceptions.InvalidSamplingException import InvalidInverseConversionException


def pca_shift_and_save_wave(wave, sr, samle_name, model, config, pca, output_path, alphas):
    """
    Takes a wave, encodes it and then shifts it in the direction of the first 3 principal components of the data multiplied by alphas.
    Then saves the reconstructed wave to the output_path with the name of the sample and the shift value.

    params:
        wave (np.array) - wave to shift
        sr (int) - sample rate of the wave
        samle_name (str) - name of the sample
        model (torch.nn.Module) - model to use for reconstruction
        config (utils.Config) - config of the model
        pca (sklearn.decomposition.PCA) - pca object fitted on the data
        output_path (Path) - path to save the reconstructed waves
        alphas (list) - list of values to multiply the principal components with

    returns:
        None
    """

    comp = pca.components_

    vectors = {
        'vector_1' : (comp[0] / np.linalg.norm(comp[0])).astype(np.float32),
        'vector_2' : (comp[1] / np.linalg.norm(comp[1])).astype(np.float32),
        'vector_3' : (comp[2] / np.linalg.norm(comp[2])).astype(np.float32),
        'vector_4' : (comp[3] / np.linalg.norm(comp[3])).astype(np.float32),
        'vector_5' : (comp[4] / np.linalg.norm(comp[4])).astype(np.float32)
    }

    for vector_name, vector in vectors.items():
        vector_output_path = output_path / vector_name
        vector_output_path.mkdir(exist_ok=True, parents=True)

        mean, _ = model.encode(prepare_wave_for_model(wave, sr, config))
        mean = to_numpy(mean)

        for alpha in alphas:
            z = mean + alpha * vector
            z = torch.tensor(z).view(1, -1)

            reconstructed_x = model.decode(z)

            try:
                reconstructed_wave = tensor_to_wave(reconstructed_x, sr, config)
                save_wave(reconstructed_wave, sr, vector_output_path / f'{samle_name}__vector={vector_name}_shift={alpha}.wav')

            except InvalidInverseConversionException as e:
                print(f'\tError with shifting a wave with with alpha={alpha} and vector={vector_name}')
                print(f'\t\t{e}')



def generate_samples_with_pca_shift(model, config, means_logvars_dict, data_path, output_path, test_samples = False, seed = None):
    """
    Generates and saves samples with pca shift for a given model and data.

    params:
        model (torch.nn.Module) - model to use for reconstruction
        config (utils.Config) - config of the model
        means_logvars_dict (dict) - dictionary of means and logvars of the data
        data_path (Path) - path to the data
        output_path (Path) - path to save the reconstructed waves
        test_samples (bool) - whether to use test samples or not
        seed (int) - seed for the random generator

    returns:
        None
    """
    data_path = Path(data_path)
    output_path = Path(output_path)

    output_path.mkdir(exist_ok=True, parents=True)


    pca = get_fitted_pca(means_logvars_dict, 5)

    alphas = [ -4,-3,-2,-1,-0.5, 0 , 0.5, 1, 2, 3, 4]

    for sample_type in ['kick', 'tom', 'crash', 'clap', 'snare']:
        wave, sr, sample_name = load_random_wave(data_path, sample_type, test_samples = test_samples, seed = seed)

        pca_shift_and_save_wave(wave, sr, sample_name, model, config, pca, output_path, alphas)
