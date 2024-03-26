'''
script for logging the config of the model 
'''
import os

import json
from datetime import datetime

class Config():
    '''
    class for logging the config of the model into a pkl file
    '''
    def __init__(self, 
                model_name='', 
                sample_group='all', 
                model='VAE_1', 
                latent_dim=32, 
                epochs=100,
                batch_size=32, 
                noise=False, 
                variance=0.0, 
                mean=0.0, 
                distribution='constant', 
                operation='additive', 
                scope='pixel', 
                date_time= None, 
                mfcc_kwargs=None, 
                pad_or_trim_length=None,
                kl_regularisation=1.0):
        
        self.model_name = model_name
        self.sample_group = sample_group
        self.model = model
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.pad_or_trim_length = pad_or_trim_length
        self.kl_regularisation = kl_regularisation

        self.date_time = date_time

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

        self.mfcc_kwargs = mfcc_kwargs

    def to_json(self):
        config_dict = self.__dict__
        config_dict['date_time'] = self.date_time.strftime("%d/%m/%Y %H:%M")

        return json.dumps(config_dict)
    
    @staticmethod
    def from_json(json_str):
        def object_hook(d):
            if 'date_time' in d:
                d['date_time'] = datetime.strptime(d['date_time'], "%d/%m/%Y %H:%M")
            if 'noise' in d and d['noise'] is None:
                d['noise'] = None
            return Config(**d)

        return json.loads(json_str, object_hook=object_hook)



def save_config(path, args, mfcc_kwargs):
    '''
    saves the config of the model to a json file

    params:
        path - path to the directory where to save the config
        args - arguments of the model
        mfcc_kwargs - kwargs for mfcc conversion config
    '''
    config = Config(model_name=args.model_name, 
                    sample_group=args.sample_group, 
                    model=args.model, 
                    latent_dim=args.latent_dim, 
                    epochs=args.epochs, 
                    batch_size=args.batch_size, 
                    noise=args.noise, 
                    variance=args.variance, 
                    mean=args.mean, 
                    distribution=args.distribution, 
                    operation=args.operation, 
                    scope=args.scope, 
                    date_time=datetime.now(), 
                    mfcc_kwargs=mfcc_kwargs,
                    pad_or_trim_length=args.pad_or_trim_length,
                    kl_regularisation=args.kl_regularisation)

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
