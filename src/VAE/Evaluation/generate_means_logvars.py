from pathlib import Path
import numpy as np
import json

from src.VAE.utils.data import load_wave, convert_to_mfcc, pad_or_trim, to_numpy, get_inverse_mfcc_kwargs, trim_wave


def return_mean_logvar(wave, sr, model, config):
    '''
    Returns mean and logvar of the given wave in the latent space of the model

    params:
        wave (np.array) - wave to encode
        sr (int) - sample rate of the wave
        model (torch.nn.Module) - model to encode the wave with
        config (utils.Config) - config of the model

    returns:
        np.array - mean of the wave in the latent space
        np.array - logvar of the wave in the latent space
    '''
    x = pad_or_trim(convert_to_mfcc(wave, sr, config.mfcc_kwargs), config.pad_or_trim_length)
    return model.encode(x)


def generate_and_save_means_and_logvars(model, config, output_path, data_path):
    '''
    generates means and logvars of all samples the model was trained on and saves them to a json file

    json file structure:
    {
        'sample_group_1': [
            {
                'sample_name': 'sample_name_1',
                'mean': [mean_1],
                'logvar': [logvar_1]
            },
            {
                'sample_name': 'sample_name_2',
                'mean': [mean_2],
                'logvar': [logvar_2]
            },
            ...
        ],
        'sample_group_2': [
            ...
        ],
        ...
    }

    params:
        model (torch.nn.Module) - model to encode the samples with
        config (utils.Config) - config of the model
        output_path (str | Path) - path to the directory where to save the means and logvars
        data_path (str | Path) - path to the directory with the samples

    returns:
        None
    '''
    data_path = Path(data_path)
    output_path = Path(output_path)

    means_logvars = {}
    for group in config.sample_groups:
        group_path = data_path / group / f'{group}_samples'
        means_logvars[group] = []

        for sample_path in group_path.iterdir():
            wave, sr = load_wave(sample_path)
            mean, logvar = return_mean_logvar(wave, sr, model, config)

            means_logvars[group].append({'sample_name': sample_path.name, 'mean': mean.tolist(), 'logvar': logvar.tolist()})


    json.dump(means_logvars, open(output_path / 'means_logvars.json', 'w'), indent=4)


def load_means_logvars_json(path):
    '''
    loads name, sample_group, means and logvars of samples from a json file for further processing

    json file structure:
    {
        'sample_group_1': [
            {
                'sample_name': 'sample_name_1',
                'mean': [mean_1],
                'logvar': [logvar_1]
            },
            {
                'sample_name': 'sample_name_2',
                'mean': [mean_2],
                'logvar': [logvar_2]
            },
            ...
        ],
        'sample_group_2': [
            ...
        ],
        ...
    }

    params:
        path (str | Path) - path to the json file

    returns:
        dict - dictionary with the loaded means and logvars
    '''


    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f'File {path} does not exist.')

    mean_logvars = json.load(open(path, 'r'))
    for group in mean_logvars:
        for item in mean_logvars[group]:
            item['mean'] = np.array(item['mean'])
            item['logvar'] = np.array(item['logvar'])

    return mean_logvars
