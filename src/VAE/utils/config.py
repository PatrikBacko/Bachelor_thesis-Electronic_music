'''
script for logging the config of the model 
'''

import os
import json
from datetime import datetime

from src.VAE.utils.scaler import save_scaler_to_config

class Config():
    '''
    class for logging the config of the model into a json file
    '''
    def __init__(self, 
                model_name='', 
                sample_group='all', 
                model='VAE_1', 
                latent_dim=32, 
                epochs=100,
                batch_size=32, 
                pad_or_trim_length=None,
                kl_regularisation=1.0,
                learning_rate=0.001,

                noise=False, 
                variance=0.0, 
                mean=0.0, 
                distribution='constant', 
                operation='additive', 
                scope='pixel', 

                scaler=None,

                date_time= None, 
                mfcc_kwargs=None):
        
        self.model_name = model_name
        self.sample_group = sample_group
        self.model = model
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.pad_or_trim_length = pad_or_trim_length
        self.kl_regularisation = kl_regularisation
        self.learning_rate = learning_rate


        if noise:
            self.noise = {
                'variance': variance,
                'mean': mean,
                'distribution': distribution,
                'operation': operation,
                'scope': scope
            }
        else:
            self.noise = None

        self.scaler = scaler

        self.mfcc_kwargs = mfcc_kwargs
        self.date_time = date_time


    def to_json(self):
        config_dict = self.__dict__
        config_dict['date_time'] = self.date_time.strftime("%d/%m/%Y %H:%M")

        return json.dumps(config_dict)
    
    @staticmethod
    def from_json(json_str):

        config_dict = json.loads(json_str)

        if config_dict['date_time'] is not None:
            config_dict['date_time'] = datetime.strptime(config_dict['date_time'], "%d/%m/%Y %H:%M")

        return Config(**config_dict)



def save_config(path, args, mfcc_kwargs):
    '''
    saves the config of the model to a json file

    params:
        path - path to the directory where to save the config
        args - arguments of the model
        mfcc_kwargs - kwargs for mfcc conversion config
    '''
    scaler = save_scaler_to_config(args.scaler)

    config = Config(model_name=args.model_name, 
                    sample_group=args.sample_group, 
                    model=args.model, 
                    latent_dim=args.latent_dim, 
                    epochs=args.epochs, 
                    batch_size=args.batch_size, 
                    pad_or_trim_length=args.pad_or_trim_length,
                    kl_regularisation=args.kl_regularisation,
                    learning_rate=args.learning_rate,
                    
                    noise=args.noise, 
                    variance=args.variance, 
                    mean=args.mean, 
                    distribution=args.distribution, 
                    operation=args.operation, 
                    scope=args.scope,

                    scaler=scaler,

                    date_time=datetime.now(), 
                    mfcc_kwargs=mfcc_kwargs)

    with open(os.path.join(path, f'config.json'), 'w') as config_file:
        config_file.write(config.to_json())


def load_config(path):
    '''
    loads the config of the model from a json file

    params:
        path - path to the directory where to save the config
    '''
    with open(path, 'r') as config_file:
        json_str = config_file.read()
    
    return Config.from_json(json_str)
