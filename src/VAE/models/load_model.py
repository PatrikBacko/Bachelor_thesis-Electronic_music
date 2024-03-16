from src.VAE.models.VAE_1 import VAE_1
from src.VAE.models.VAE_1 import VAE_2

def load_model(model, **model_kwargs):
    '''
    Returns the VAE model based on the model_type
    
    parameters:
        model_type: str, type of the model
        latent_dim: int, latent dimension of the model

    returns:
        model: torch.nn.Module, model of the VAE
    '''
    if model == 'VAE_1':
        return VAE_1(**model_kwargs)
    elif model == 'VAE_2':
        return VAE_2(**model_kwargs)