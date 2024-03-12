'''
script for logging the config of the model 
'''

import math
import os

import numpy as np
import librosa as lb
# import soundfile as sf
# import pyaudio

import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse

import pickle as pkl
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
                pad_or_trim_length=None):
        
        self.model_name = model_name
        self.sample_group = sample_group
        self.model = model
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.pad_or_trim_length = pad_or_trim_length

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



def save_config(path, args, mfcc_kwargs):
    '''
    saves the config of the model to a pkl file using pickle, and saves the human readable version to a txt file

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
                    pad_or_trim_length=args.pad_or_trim_length)

    pkl.dump(config, open(os.path.join(path, f'{args.model_name}_config.pkl'), 'wb'))

    with open(os.path.join(path, f'{args.model_name}_config.txt'), 'w') as config_file:
        save_human_readable_config(args, mfcc_kwargs, config_file)

def save_human_readable_config(args, mfcc_kwargs, config_file):
    '''
    logs config of the model to a human readable format

    params:
        args - arguments of the model
        config_file - file to write the config to
    '''
    print('******************', file=config_file)
    print(f'Model {args.model_name} config:', file=config_file)
    print('\tModel: ', args.model, file=config_file)
    print('\tLatent dimension: ', args.latent_dim, file=config_file)
    print('\tEpochs: ', args.epochs, file=config_file)
    print('\tBatch size: ', args.batch_size, file=config_file)
    print('\tSample groups: ', args.sample_group, file=config_file)
    print('\tPad or trim length: ', args.pad_or_trim_length, file=config_file)

    print('******************', file=config_file)
    if args.noise:
        print('Noise:', file=config_file)
        print('\tNoise distribution: ', args.distribution, file=config_file)
        print('\tNoise operation: ', args.operation, file=config_file)
        print('\tNoise scope: ', args.scope, file=config_file)
        print('\tNoise variance: ', args.variance, file=config_file)
        print('\tNoise mean: ', args.mean, file=config_file)
    else:
        print('Noise: None', file=config_file)


    print('******************', file=config_file)
    print(f'Date and time: {datetime.now().strftime("%d/%h/%Y %H:%M")}', file=config_file)
    print('******************', file=config_file)
    print('MFCC conversion kwargs:', file=config_file)
    print(f'\tMFCC kwargs: {mfcc_kwargs}', file=config_file)
    print('******************', file=config_file)
    print('machine readable config saved also to .pkl file')

def load_config(path):
    return pkl.load(open(path, 'rb'))
