from src.VAE.models.VAE_1 import VAE_1
from src.VAE.models.VAE_2 import VAE_2
from src.VAE.models.VAE_3 import VAE_3
from src.VAE.models.VAE_4 import VAE_4
from src.VAE.models.VAE_3_old import VAE_3_old
from src.VAE.models.VAE_3_dropout import VAE_3_dropout
from src.VAE.models.VAE_4_dropout import VAE_4_dropout
from src.VAE.models.VAE_5_tied_weights import VAE_5_tied_weights
from src.VAE.models.VAE_6 import VAE_6

import torch

MODELS = {
        'VAE_1': VAE_1,
        'VAE_2': VAE_2,
        'VAE_3': VAE_3,
        'VAE_3_old': VAE_3_old,
        'VAE_4': VAE_4,
        'VAE_3_dropout': VAE_3_dropout,
        'VAE_4_dropout': VAE_4_dropout,
        'VAE_5_tied_weights': VAE_5_tied_weights,
        'VAE_6' : VAE_6
        }

def get_models_list():
    '''
    Returns the list of available models

    returns:
        list, list of available models
    '''
    return MODELS.keys()

def check_model_validity(model, conversion_config):
    '''
    checks if the model is valid based for chosen conversion_config

    parameters:
        model: str, model type
        conversion_config: dict, configuration for the conversion

    returns:
        bool, True if model is valid, False otherwise
    '''
    if model not in MODELS:
        raise ValueError(f"Model {model} not found. Available models are {MODELS.keys()}")

    input_shape = return_input_shape(model)

    if input_shape[1] != conversion_config['channels']:
        raise ValueError(f"Model {model} expects {input_shape[1]} channels, but {conversion_config['channels']} channels were given")

    if input_shape[2] != conversion_config['height']:
        raise ValueError(f"Model {model} expects {input_shape[2]} height, but {conversion_config['height']} height was given")

    else:
        return True

def create_model(model, latent_dim):
    '''
    Returns the VAE model based on the model_type
    
    parameters:
        model_type: str, type of the model
        latent_dim: int, latent dimension of the model

    returns:
        model: torch.nn.Module, model of the VAE
    '''
    return MODELS[model](latent_dim)

def return_input_shape(model):
    '''
    Returns the input shape based on the model type

    parameters:
        model: str, type of the model

    returns:
        input_shape: tuple, input shape of the model
    '''
    return MODELS[model].input_shape

def return_pad_or_trim_len(model):
    '''
    Returns the padding or trimming size based on the model type

    parameters:
        model: str, type of the model

    returns:
        pad_or_trim_len: int, padding or trimming size of
    '''
    return MODELS[model].input_shape[3]

def load_model(model_path, model_type, latend_dim, device='cpu'):
    '''
    Loads the model from the model_path
    
    parameters:
        model_path: str, path to the model
        model: str, type of the model
        latent_dim: int, latent dimension of the model

    returns:
        model: torch.nn.Module, model of the VAE
    '''
    model_type = create_model(model_type, latend_dim)
    model_type.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    return model_type