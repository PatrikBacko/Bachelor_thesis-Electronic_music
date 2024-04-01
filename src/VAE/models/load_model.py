from src.VAE.models.VAE_1 import VAE_1
from src.VAE.models.VAE_2 import VAE_2
from src.VAE.models.VAE_3 import VAE_3
from src.VAE.models.VAE_4 import VAE_4
from src.VAE.models.VAE_3_old import VAE_3_old

import torch

MODELS = ['VAE_1', 'VAE_2', 'VAE_3', 'VAE_3_old', 'VAE_4']

def create_model(model, latent_dim):
    '''
    Returns the VAE model based on the model_type
    
    parameters:
        model_type: str, type of the model
        latent_dim: int, latent dimension of the model

    returns:
        model: torch.nn.Module, model of the VAE
    '''
    if model == 'VAE_1':
        return VAE_1(latent_dim)
    elif model == 'VAE_2':
        return VAE_2(latent_dim)
    elif model == 'VAE_3':
        return VAE_3(latent_dim)
    elif model == 'VAE_3_old':
        return VAE_3_old(latent_dim)
    elif model == 'VAE_4':
        return VAE_4(latent_dim)
    else:
        raise ValueError('Model type not found')

    

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
