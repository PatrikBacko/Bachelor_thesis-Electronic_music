from sklearn.preprocessing import StandardScaler
import numpy as np


SCALERS = ['standard']

def create_scaler(scaler_type):
    '''
    creates a scaler object

    params:
        scaler_type - type of scaler

    returns:
        scaler - scaler object
    '''
    if scaler_type == 'standard':
        return StandardScaler()
    
    else:
        raise ValueError(f'Unknown scaler type: {scaler_type}')
    

def save_scaler_to_config(scaler):
    '''
    saves a scaler object to a config

    params:
        scaler - scaler object

    returns:
        scaler_config - config of the scaler
    '''
    if scaler is None:
        return None


    if isinstance(scaler, StandardScaler):
        scaler_config = {
            'type': 'standard',
            'mean': list(scaler.mean_),
            'scale': list(scaler.scale_)
        }
    else:
        raise ValueError(f'Unknown scaler type: {scaler.__class__.__name__}')

    return scaler_config
    

def load_scaler(scaler_config):
    '''
    loads a scaler object from a config

    params:
        scaler_config (dict) - config of the scaler

    returns:
        scaler - scaler object or None if no config is provided
    '''
    if scaler_config is None:
        return None

    if scaler_config['type'] == 'standard':
        scaler = create_scaler(scaler_config['type'])
        scaler.mean_ = np.array(scaler_config['mean'])
        scaler.scale_ = np.array(scaler_config['scale'])
        return scaler
    
    else:
        raise ValueError(f'Unknown scaler type: {scaler_config["type"]}')