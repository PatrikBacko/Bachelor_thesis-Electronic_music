from src.VAE.models.VAE_1 import VAE_1
from src.VAE.models.VAE_2 import VAE_2
from src.VAE.models.VAE_3 import VAE_3

import torch

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
    

def load_model(model_path, model, latend_dim):
    '''
    Loads the model from the model_path
    
    parameters:
        model_path: str, path to the model
        model: str, type of the model
        latent_dim: int, latent dimension of the model

    returns:
        model: torch.nn.Module, model of the VAE
    '''
    model = create_model(model, latend_dim)
    model.load_state_dict(torch.load(model_path))
    return model
