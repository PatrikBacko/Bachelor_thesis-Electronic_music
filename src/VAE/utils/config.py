'''
script for logging the config of the model 
'''

import os
import json
from datetime import datetime

from src.VAE.utils.scaler import save_scaler_to_config
from src.VAE.utils.conversion import get_default_conversion_config

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

                conversion_config = get_default_conversion_config('mfcc'),
                scaler=None,

                date_time= None
                ):
        
        self.model_name = model_name
        self.sample_group = sample_group
        self.model = model
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.pad_or_trim_length = pad_or_trim_length
        self.kl_regularisation = kl_regularisation
        self.learning_rate = learning_rate
        self.conversion_config = conversion_config


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

        
        self.date_time = date_time
        self.scaler = scaler


    def _to_json(self, indent=None):
        config_dict = self.__dict__

        return json.dumps(config_dict, indent=indent)
    
    def save_config(self, path):
        '''
        saves the config of the model to a json file

        params:
            path - path to the directory where to save the config
            args - arguments of the model
            mfcc_kwargs - kwargs for mfcc conversion config
        '''
        self.scaler = save_scaler_to_config(self.scaler)

        with open(os.path.join(path, f'config.json'), 'w') as config_file:
            config_file.write(self._to_json())
    
    @staticmethod
    def _from_json(json_str):

        config_dict = json.loads(json_str)

        if config_dict['date_time'] is not None:
            config_dict['date_time'] = datetime.strptime(config_dict['date_time'], "%d/%m/%Y %H:%M")
        
        # convert legacy config to new config
        if 'mfcc_kwargs' in config_dict:
            config_dict = Config._legacy_congig_to_new(config_dict)

        return Config(**config_dict)
    
    @staticmethod
    def _legacy_congig_to_new(config_dict):
        
        mfcc_kwargs = config_dict['mfcc_kwargs']
        del config_dict['mfcc_kwargs']
        conversion_config = get_default_conversion_config('mfcc')
        conversion_config['kwargs'] = mfcc_kwargs
        config_dict['conversion_config'] = conversion_config

        return config_dict

    @staticmethod
    def create_config(args, conversion_config, pad_or_trim_length):
        '''
        creates the config of the model

        params:
            args - arguments of the model
            conversion_config - config for the conversion
        
        returns:
            Config - config of the model
        '''

        return Config(  sample_group=args.sample_group, 
                        model=args.model, 
                        latent_dim=args.latent_dim, 
                        epochs=args.epochs, 
                        batch_size=args.batch_size, 
                        pad_or_trim_length=pad_or_trim_length,
                        kl_regularisation=args.kl_regularisation,
                        learning_rate=args.learning_rate,
                        
                        noise=args.noise, 
                        variance=args.variance, 
                        mean=args.mean, 
                        distribution=args.distribution, 
                        operation=args.operation, 
                        scope=args.scope,

                        conversion_config=conversion_config,
                        scaler=args.scaler,

                        date_time=datetime.now().strftime("%d/%m/%Y %H:%M"))


    @staticmethod
    def load(path):
        '''
        loads the config of the model from a json file

        params:
            path - path to the directory where to save the config

        returns:
            Config - config of the model
        '''
        with open(path, 'r') as config_file:
            json_str = config_file.read()
        
        return Config._from_json(json_str)
