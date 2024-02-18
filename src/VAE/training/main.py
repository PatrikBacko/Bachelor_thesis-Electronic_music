#!/usr/bin/env python3
'''
Script for training the model (Variational Autoencoder) on given samples and saving it.
'''

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

import argparse


def build_arguments():
    '''
    builds arguments for the script
    '''

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('source_dir', type=str, help='Path to directory with samples.')
    parser.add_argument('output_path', type=str, help='Path to directory where to save the model and logs etc.')

    parser.add_argument('--model_name', type=str, help='Name of the log file.')
    parser.add_argument('--sample_group', type=str, help='Name of the sample group to train on.')

    parser.add_argument('--model', type=str, default='VAE_1',help='Model to train.')
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent dimension of the model.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training the model.')

    return parser.parse_args()



def log_model_config(args, config_file):
    '''
    logs config of the model
    '''
    print('******************', file=config_file)
    print(f'Model {args.model_name} config:', file=config_file)
    print('******************', file=config_file)
    print('Model: ', args.model, file=config_file)
    print('Latent dimension: ', args.latent_dim, file=config_file)
    print('Epochs: ', args.epochs, file=config_file)
    print('Batch size: ', args.batch_size, files=config_file)
    print('******************', file=config_file)
    

# TODO:implement function for loading data from all sample groups at once
def main(args):
    data_path = os.path.join(args.source_dir, args.sample_group)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = prepare_data(data_path, args.batch_size, length = 100)
    model = VAE_1(args.latent_dim).to(device)

    with open(os.path.join(args.output_path, f'{args.model_name}_training.log'), 'w') as log_file: 
        ##TODO: implement different models

        ### debug log ###
        print('$$$$$$$$$$$$$$$$$$$$$$$', file=log_file)
        print('Device: ', device, file=log_file)
        print('$$$$$$$$$$$$$$$$$$$$$$$', file=log_file)
        #################

        losses = train(model, train_loader, args.epochs, device)

        torch.save(model.state_dict(), args.output_path)
        print(f'Model saved to {args.output_path}', file=log_file)


    with open(os.path.join(args.output_path, f'{args.model_name}_config'), 'w') as config_file:
        log_model_config(args, config_file)


if __name__ == '__main__':
    args = build_arguments()
    main(args)
