 #!/usr/bin/env python3
'''
Script for training the model (Variational Autoencoder) on given samples and saving it.
usage: python main.py [data_dir] [output_path] [--model_name]  [-h] [optional args] 
'''

# # TEMPORARY
# import sys
# sys.path.append(r'C:\Users\llama\Desktop\cuni\bakalarka\Bachelor_thesis-Electronic_music')
# # TEMPORARY

import math
import os

import numpy as np
import librosa as lb
import soundfile as sf
import pyaudio

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.VAE.models.VAE_1 import VAE_1
from src.VAE.training.train_model import train
from src.VAE.utils.prepare_data import prepare_data 

from src.VAE.utils.add_noise import generate_noise, NOISE_SCOPE, NOISE_OPERATION_TYPES, NOISE_GENERATING_DISTS

import argparse


def build_arguments():
    '''
    builds arguments for the script
    '''

    parser = argparse.ArgumentParser(description=__doc__)

    #required arguments
    parser.add_argument('data_dir', type=str, help='Required, Path to directory with samples.')
    parser.add_argument('output_path', type=str, help='Required, Path to directory where to save the model and logs etc.')
    parser.add_argument('--model_name', type=str, help='Required, Name of the log file.')

    #optional arguments
    parser.add_argument('--sample_group', type=str, default='all', help='Names of the sample groups to train the model on, seperated by comma.'
                                                            'If used, only the specified sample groups will be used for training.'
                                                            'possible groups: kick, clap, hat, snare, tom, cymbal, crash, ride')
    parser.add_argument('--model', type=str, default='VAE_1',help='Model to train.')
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent dimension of the model.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training the model.')


    #Noise arguments
    parser.add_argument('-n','--noise', action='store_true', help='Add noise to the spectograms. Config of noise can be set with other noise arguments.'
                        'if this switch is not used, other arguments will be ignored.\n')
    
    #argument for noise variance
    parser.add_argument("-v", "--variance", help="noise variance for generating distribution", type=float, default=0.0)
    #argument for noise mean
    parser.add_argument("-m", "--mean", help="noise mean for generating distribution", type=float, default=0.0)

    #argument for noise distribution type
    parser.add_argument("-d", "--distribution", help="noise generating distribution"
                        "normal: normal distribution"
                        "uniform: uniform distribution"
                        "constant: value is equal to the chosen mean"
                        , choices=NOISE_GENERATING_DISTS, default="constant")
    #argument for noise operation type
    parser.add_argument("-o", "--operation", help="noise operation type (how will be noise added to the spectogram)"
                        "additive: add noise to the spectogram"
                        "multiplicative: multiply the spectogram with noise (noise values are coefficients)"
                        , choices=NOISE_OPERATION_TYPES, default="additive")
    #argument for noise scope
    parser.add_argument("-s", "--scope", help="noise scope."
                        "pixel: add noise to each pixel in the spectogram"
                        "column: each column in the spectogram will have the same noise, but different from other columns"
                        "row: each row in the spectogram will have the same noise, but different from other rows"
                        "entire_picture: the entire spectogram will have the same noise"
                        , choices=NOISE_SCOPE, default="pixel")

    return parser.parse_args()



def log_model_config(args, config_file):
    '''
    logs config of the model
    '''
    print('******************', file=config_file)
    print(f'Model {args.model_name} config:', file=config_file)
    print('\tModel: ', args.model, file=config_file)
    print('\tLatent dimension: ', args.latent_dim, file=config_file)
    print('\tEpochs: ', args.epochs, file=config_file)
    print('\tBatch size: ', args.batch_size, file=config_file)
    print('******************', file=config_file)
    print('Noise config:', file=config_file)
    print('\tNoise distribution: ', args.distribution, file=config_file)
    print('\tNoise operation: ', args.operation, file=config_file)
    print('\tNoise scope: ', args.scope, file=config_file)
    print('\tNoise variance: ', args.variance, file=config_file)
    print('\tNoise mean: ', args.mean, file=config_file)
    print('******************', file=config_file)
    print('Sample groups: ', args.sample_group, file=config_file)
    print('******************', file=config_file)
    print(f'command line arguments: {args}', file=config_file)
    

def main(args):
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    with open(os.path.join(args.output_path, f'{args.model_name}_training.log'), 'w') as log_file: 
        
        length = 100  #length of the spectogram to be trimmed or padded to. With the current settings of mfcc conversion, it is around 1 seconds of audio
        if args.sample_group == 'all':
            sample_groups = ['kick', 'clap', 'hat', 'snare', 'tom', 'cymbal', 'crash', 'ride']
        else:
            sample_groups = args.sample_group.split(',')
        train_loader = prepare_data(args.data_dir, sample_groups, length=length, batch_size=args.batch_size)
        print(f'Data prepared for training. Sample groups: {", ".join(sample_groups)}\n, mfcc length: {length}, data directory {args.data_dir}', file=log_file)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Device : {device}\n', file=log_file)

        model = VAE_1(args.latent_dim).to(device)
        print(f'Model {args.model_name} created. (model type: {args.model})\n', file=log_file)

        if args.noise:
            noise_function = generate_noise(args.mean, args.variance, args.distribution, args.scope, args.operation)
            print(f'Noise function created with config: \n'
                    f'\tNoise distribution: {args.distribution}\n'
                    f'\tNoise operation: {args.operation}\n'
                    f'\tNoise scope: {args.scope}\n'
                    f'\tNoise variance: {args.variance}\n'
                    f'\tNoise mean: {args.mean}\n', file=log_file)

        else:
            noise_function = lambda x:x
            print('No noise added to the spectograms.\n', file=log_file)
        
        ##TODO: implement different models

        losses = train(model, train_loader, args.epochs, device, log_file, noise_function=noise_function)

        torch.save(model.state_dict(), os.path.join(args.output_path, f'model_{args.model_name}.pkl'))
        print(f'Model saved to {args.output_path}', file=log_file)


    with open(os.path.join(args.output_path, f'{args.model_name}.config'), 'w') as config_file:
        log_model_config(args, config_file)


if __name__ == '__main__':
    args = build_arguments()
    main(args)
